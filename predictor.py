"""
预测选股模块
功能：加载模型，预测"T日收盘后生成信号、T+1日开盘买入、T+6日收盘卖出"的持有期收益，排序选出 Top N 股票
支持深度学习（DL）和 XGBoost 两种模型
"""

import pandas as pd
import numpy as np
import config
import json
import os
import hashlib
import pickle
from datetime import datetime, timedelta
from data_loader import DataLoader
from dataset import DatasetBuilder
from model import QuantModel
from baseline_models import XGBoostModel


class StockPredictor:
    """股票预测器（支持 DL / XGBoost 双模式）"""

    def __init__(self, model_type='dl'):
        """
        Args:
            model_type: 'dl'=深度学习, 'xgb'=XGBoost（默认）
        """
        self.model_type = model_type
        self.loader = DataLoader()
        self.builder = DatasetBuilder()
        self.model = None
        self.xgb_model = None
        self.tabular_model = None
        self.feature_cols = None
        self.model_meta = {}

        # 加载模型
        self.load_model()

    def load_model(self):
        """加载训练好的模型"""
        try:
            if self.model_type == 'dl':
                self.model = QuantModel()
                self.model.load(config.MODEL_FILE)
                self.builder.load_scaler(config.SCALER_FILE, config.FEATURE_COLS_FILE)
                model_features = self.model.model.input_shape[-1]
                saved_features = len(self.builder.feature_cols)
                if model_features != saved_features:
                    raise ValueError(
                        f"模型与特征清单维度不一致: "
                        f"model={model_features}, feature_cols={saved_features}"
                    )
                self._load_and_validate_meta(
                    config.MODEL_META_FILE,
                    self.builder.feature_cols,
                    expected_model_type='dl'
                )
                print("深度学习模型加载成功")
            elif self.model_type == 'xgb':
                self.xgb_model = XGBoostModel()
                self.xgb_model.load()
                self.feature_cols = self.xgb_model.feature_names
                self._load_and_validate_meta(
                    config.XGB_MODEL_META_FILE,
                    self.feature_cols,
                    expected_model_type='xgb'
                )
                print("XGBoost 模型加载成功")
            elif self.model_type == 'tabular':
                with open(config.TABULAR_MODEL_FILE, 'rb') as file:
                    artifact = pickle.load(file)
                self.tabular_model = artifact['model']
                self.feature_cols = list(artifact['feature_cols'])
                self._load_and_validate_meta(
                    config.TABULAR_MODEL_META_FILE,
                    self.feature_cols,
                    expected_model_type='tabular'
                )
                print(
                    f"表格模型加载成功: {artifact.get('model_name', 'unknown')}"
                )
            else:
                raise ValueError(f"不支持的模型类型: {self.model_type}")

        except Exception as e:
            raise ValueError(f"加载模型失败: {e}，请先训练模型")

    def _load_and_validate_meta(self, meta_file, feature_cols, expected_model_type):
        """强制校验模型时间范围、标签定义和特征哈希。"""
        if not meta_file or not os.path.exists(meta_file):
            raise FileNotFoundError(f"模型元数据不存在: {meta_file}")
        with open(meta_file, 'r', encoding='utf-8') as file:
            meta = json.load(file)

        required = {
            'model_type', 'data_start_date', 'data_end_date',
            'label_entry_shift', 'label_exit_shift', 'label_entry_price',
            'label_exit_price', 'feature_count', 'feature_hash'
        }
        missing = sorted(required - set(meta))
        if missing:
            raise ValueError(f"模型元数据缺少字段: {missing}")
        if meta['model_type'] != expected_model_type:
            raise ValueError("模型类型与元数据不一致")
        if expected_model_type != 'tabular':
            if int(meta['label_entry_shift']) != int(config.LABEL_ENTRY_SHIFT):
                raise ValueError("模型买入日期定义与当前配置不一致")
            if int(meta['label_exit_shift']) != int(config.LABEL_EXIT_SHIFT):
                raise ValueError("模型卖出日期定义与当前配置不一致")
        if meta['label_entry_price'] != 'open' or meta['label_exit_price'] != 'close':
            raise ValueError("旧模型标签价格定义已失效，必须重新训练")
        if int(meta['feature_count']) != len(feature_cols):
            raise ValueError("模型元数据与特征数量不一致")

        actual_hash = hashlib.sha256(
            '\n'.join(feature_cols).encode('utf-8')
        ).hexdigest()
        if meta['feature_hash'] != actual_hash:
            raise ValueError("模型元数据与特征清单哈希不一致")
        self.model_meta = meta

    def get_candidate_stocks(self, trade_date=None):
        """
        获取候选股票

        Args:
            trade_date: 交易日期，默认为最近一个交易日

        Returns:
            tuple: (stock_data_dict, stock_meta_dict)
                - stock_data: {ts_code: DataFrame}
                - stock_meta: {ts_code: {'sector': ..., 'market_cap': ...}}
        """
        if trade_date is None:
            trade_date = self.loader.get_latest_trade_date()

        print(f"\n获取 {trade_date} 的候选股票...")

        # 获取股票列表（含行业/市值）
        stock_list = self.loader.get_stock_list(trade_date)
        pool_limit = int(
            self.model_meta.get('pool_limit', getattr(config, 'STOCK_POOL_LIMIT', 0))
            or 0
        )
        if pool_limit > 0 and len(stock_list) > pool_limit:
            if 'market_cap' in stock_list.columns:
                stock_list = stock_list.nlargest(pool_limit, 'market_cap')
            else:
                stock_list = stock_list.head(pool_limit)
            stock_list = stock_list.reset_index(drop=True)
        sample_count = getattr(config, 'RUNTIME_SAMPLE_STOCK_COUNT', None)
        if sample_count:
            print(f"小样本模式：预测仅使用前 {sample_count} 只候选股票")
            stock_list = stock_list.head(sample_count).reset_index(drop=True)

        # 构建元数据字典
        stock_meta = {}
        if 'sector' in stock_list.columns:
            stock_meta = {
                row['ts_code']: {
                    'sector': row.get('sector', config.DEFAULT_SECTOR),
                    'market_cap': row.get('market_cap', np.nan),
                    'name': row.get('name', '')
                }
                for _, row in stock_list.iterrows()
            }

        # 获取日线数据
        data_start_date = self.get_predict_start_date(trade_date)
        stock_data = self.loader.get_all_stocks_daily(
            stock_list,
            start_date=data_start_date,
            end_date=trade_date
        )

        # 流动性筛选
        stock_data = self.loader.filter_by_liquidity(stock_data, trade_date)

        # 波动率筛选
        stock_data = self.loader.filter_by_volatility(stock_data)

        # 同步过滤 stock_meta
        stock_meta = {k: v for k, v in stock_meta.items() if k in stock_data}

        return stock_data, stock_meta

    def get_predict_start_date(self, trade_date):
        """预测只需要最近一段历史，避免每次全量拉取。"""
        lookback_days = getattr(config, 'PREDICT_LOOKBACK_DAYS', 260)
        start_dt = datetime.strptime(str(trade_date), '%Y%m%d') - timedelta(days=lookback_days)
        return start_dt.strftime('%Y%m%d')

    def predict_returns(self, stock_data, stock_meta=None, trade_date=None):
        """
        预测所有股票的"T+1日开盘买入、T+6日收盘卖出"持有期收益

        Args:
            stock_data: 股票数据字典
            stock_meta: 股票元数据字典（XGBoost 模式需要行业/市值）

        Returns:
            DataFrame: 预测结果（含股票代码、日期、收盘价、预测概率、行业、市值）
        """
        entry_shift = int(self.model_meta.get('label_entry_shift', config.LABEL_ENTRY_SHIFT))
        exit_shift = int(self.model_meta.get('label_exit_shift', config.LABEL_EXIT_SHIFT))
        print(
            f"\n预测 T+{entry_shift}日开盘买入→T+{exit_shift}日收盘卖出 "
            f"（模型: {self.model_type}）..."
        )

        if self.model_type == 'dl':
            return self._predict_dl(stock_data, trade_date)
        else:
            return self._predict_xgb(stock_data, stock_meta)

    def _predict_dl(self, stock_data, trade_date=None):
        """深度学习预测"""
        X, df_info = self.builder.prepare_predict_data(
            stock_data, expected_trade_date=trade_date
        )
        predictions = self.model.predict(X)
        df_info['model_score'] = predictions.flatten()
        df_info['predicted_return'] = df_info['model_score']
        return df_info

    def _predict_xgb(self, stock_data, stock_meta):
        """XGBoost或通用表格模型的截面预测。"""
        # 1. 构建截面数据（最近一天）
        from features import FeatureEngineer
        engineer = FeatureEngineer()

        rows = []
        for ts_code, df in stock_data.items():
            try:
                df = engineer.add_all_features(df)
                df['ts_code'] = ts_code
                df = DatasetBuilder.finalize_feature_frame(
                    df, require_target=False
                )
                if len(df) == 0:
                    continue
                # 取最近一天
                last_row = df.iloc[-1:].copy()
                # 合并元数据
                if stock_meta and ts_code in stock_meta:
                    meta = stock_meta[ts_code]
                    last_row['sector'] = meta.get('sector', config.DEFAULT_SECTOR)
                    last_row['market_cap'] = meta.get('market_cap', np.nan)
                    last_row['name'] = meta.get('name', '')
                rows.append(last_row)
            except Exception as e:
                print(f"预测处理 {ts_code} 失败: {e}")
                continue

        if not rows:
            raise ValueError("没有有效的预测数据")

        df_predict = pd.concat(rows, ignore_index=True)
        df_predict = DatasetBuilder.add_cross_section_features(df_predict)
        df_predict = df_predict.replace([np.inf, -np.inf], np.nan)

        # 2. 提取特征（与训练时一致）
        if self.feature_cols is None:
            raise ValueError("XGBoost 特征列未加载")

        missing = [c for c in self.feature_cols if c not in df_predict.columns]
        if missing:
            raise ValueError(f"预测数据缺少训练特征: {missing[:10]}")

        df_predict[self.feature_cols] = df_predict[self.feature_cols].fillna(0.0)

        X = df_predict[self.feature_cols].values

        # 3. 预测
        if self.model_type == 'tabular':
            preds = self.tabular_model.predict(X)
        else:
            preds = self.xgb_model.predict(X)
        if getattr(config, 'TARGET_TYPE', 'classification') == 'regression':
            df_predict['model_score'] = preds
            df_predict['predicted_return'] = preds
        else:
            df_predict['predicted_prob'] = preds
            df_predict['model_score'] = preds
            df_predict['predicted_return'] = preds

        # 4. 组织输出
        keep_cols = ['ts_code', 'trade_date', 'close', 'model_score', 'predicted_return', 'sector', 'market_cap', 'name']
        if 'predicted_prob' in df_predict.columns:
            keep_cols.append('predicted_prob')
        df_info = df_predict[keep_cols].copy()
        return df_info

    def rank_and_select(self, df_predictions, top_n=None, sector_neutral=True):
        """
        排序选股：按预测收益率从高到低排序，选 Top N
        可选行业中性化权重分配

        Args:
            df_predictions: 预测结果 DataFrame
            top_n: 选择数量
            sector_neutral: 是否做行业中性化权重

        Returns:
            DataFrame: Top N 股票及权重
        """
        if top_n is None:
            top_n = config.TOP_N_STOCKS

        if 'model_score' not in df_predictions.columns and 'predicted_return' in df_predictions.columns:
            df_predictions = df_predictions.copy()
            df_predictions['model_score'] = df_predictions['predicted_return']
        elif 'model_score' not in df_predictions.columns and 'predicted_prob' in df_predictions.columns:
            df_predictions = df_predictions.copy()
            df_predictions['model_score'] = df_predictions['predicted_prob']

        df_sorted = df_predictions.sort_values('model_score', ascending=False).reset_index(drop=True)
        df_sorted['rank'] = np.arange(1, len(df_sorted) + 1)

        if not sector_neutral or 'sector' not in df_sorted.columns:
            df_top = df_sorted.head(top_n).copy()
            df_top['weight'] = 1.0 / len(df_top)
            print(f"Top {top_n} 股票（等权）模型分数范围: {df_top['model_score'].min():.4f} ~ {df_top['model_score'].max():.4f}")
            return df_top

        df_top = self._allocate_sector_neutral(df_sorted, top_n)
        return df_top

    def _allocate_sector_neutral(self, df_sorted, top_n):
        """
        行业中性化权重分配
        简化实现：等权 + 行业偏离截断
        """
        pool = df_sorted.head(top_n * 2).copy()

        selected = []
        deferred = []
        sector_count = {}
        max_per_sector = max(1, int(np.ceil(top_n * 0.3)))

        for _, row in pool.iterrows():
            if len(selected) >= top_n:
                break
            sector = row['sector']
            if sector_count.get(sector, 0) >= max_per_sector:
                deferred.append(row)
                continue
            selected.append(row)
            sector_count[sector] = sector_count.get(sector, 0) + 1

        # 行业不足时按模型分数回补，持仓数量优先，明确放宽约束。
        if len(selected) < top_n:
            for row in deferred:
                selected.append(row)
                if len(selected) >= top_n:
                    break

        df_top = pd.DataFrame(selected)
        if len(df_top) == 0:
            df_top = df_sorted.head(top_n).copy()
        elif len(df_top) < top_n:
            selected_codes = set(df_top['ts_code'])
            fill = df_sorted[~df_sorted['ts_code'].isin(selected_codes)].head(
                top_n - len(df_top)
            )
            df_top = pd.concat([df_top, fill], ignore_index=True)

        df_top['weight'] = 1.0 / len(df_top)

        sector_dist = df_top.groupby('sector')['weight'].sum().sort_values(ascending=False)
        print(f"Top {len(df_top)} 股票（行业中性）模型分数范围: {df_top['model_score'].min():.4f} ~ {df_top['model_score'].max():.4f}")
        print(f"行业分布: {sector_dist.to_dict()}")
        return df_top


    def add_stock_info(self, df_top):
        """
        添加股票名称等信息

        Args:
            df_top: Top N 股票

        Returns:
            DataFrame: 添加信息后的数据
        """
        # 获取股票基本信息
        frames = []
        for status in ('L', 'D', 'P'):
            frame = self.loader.pro.stock_basic(
                exchange='', list_status=status,
                fields='ts_code,symbol,name,industry'
            )
            if frame is not None and not frame.empty:
                frames.append(frame)
        stock_basic = pd.concat(frames, ignore_index=True).drop_duplicates('ts_code')

        # 合并
        merge_base = df_top.drop(columns=[c for c in ['name', 'industry'] if c in df_top.columns])
        df_result = merge_base.merge(
            stock_basic[['ts_code', 'name', 'industry']],
            on='ts_code',
            how='left'
        )

        return df_result

    def format_output(self, df_result):
        """
        格式化输出

        Args:
            df_result: 结果数据

        Returns:
            DataFrame: 格式化后的数据
        """
        df_output = df_result.copy()

        # 选择输出列
        output_cols = [
            'rank',
            'ts_code',
            'name',
            'industry',
            'close',
            'model_score'
        ]
        if 'weight' in df_output.columns:
            output_cols.append('weight')

        df_output = df_output[output_cols]

        # 重命名列
        col_names = [
            '排名',
            '股票代码',
            '股票名称',
            '行业',
            'T日参考价',
            '模型分数'
        ]
        if 'weight' in df_output.columns:
            col_names.append('建议权重')
        df_output.columns = col_names

        # 格式化数值
        df_output['T日参考价'] = df_output['T日参考价'].round(2)
        df_output['模型分数'] = df_output['模型分数'].round(4)
        if '建议权重' in df_output.columns:
            df_output['建议权重'] = (df_output['建议权重'] * 100).round(2).astype(str) + '%'

        return df_output

    def predict_and_select(self, trade_date=None, top_n=None, sector_neutral=True):
        """
        预测并选股（完整流程）

        Args:
            trade_date: 交易日期
            top_n: 选择的股票数量
            sector_neutral: 是否行业中性化

        Returns:
            DataFrame: 推荐股票列表（含权重）
        """
        print("=" * 50)
        print("开始预测和选股")
        print("=" * 50)

        effective_trade_date = trade_date or self.loader.get_latest_trade_date()

        # 获取候选股票
        stock_data, stock_meta = self.get_candidate_stocks(effective_trade_date)

        if not stock_data:
            print("没有符合条件的股票")
            return pd.DataFrame()

        # 预测
        df_predictions = self.predict_returns(
            stock_data, stock_meta, trade_date=effective_trade_date
        )

        # 排序选择（行业中性化）
        df_top = self.rank_and_select(df_predictions, top_n, sector_neutral=sector_neutral)

        if df_top.empty:
            print("没有符合预测标准的股票")
            return pd.DataFrame()

        # 添加股票信息
        df_result = self.add_stock_info(df_top)

        # 格式化输出
        df_output = self.format_output(df_result)

        print("\n" + "=" * 50)
        print("选股完成")
        print("=" * 50)

        return df_output


class BacktestEngine:
    """按真实资金、成交约束和完整费用模拟非重叠五交易日组合。"""

    def __init__(self, model_type='dl'):
        self.loader = DataLoader()
        self.model_type = model_type

    def run_simple_backtest(self, start_date=None, end_date=None):
        """运行非重叠成交模拟，逐日盯市并保留延迟卖出的资金占用。"""
        if start_date is None:
            start_date = config.BACKTEST_START_DATE
        if end_date is None:
            end_date = self.loader.get_latest_trade_date()

        print("\n" + "=" * 50)
        print(f"回测期间: {start_date} - {end_date}")
        print("=" * 50)
        trade_cal = self.loader.get_trade_cal(start_date, end_date)
        trade_dates = sorted(trade_cal['cal_date'].astype(str).tolist())
        predictor = StockPredictor(model_type=self.model_type)
        entry_shift = int(
            predictor.model_meta.get('label_entry_shift', config.LABEL_ENTRY_SHIFT)
        )
        exit_shift = int(
            predictor.model_meta.get('label_exit_shift', config.LABEL_EXIT_SHIFT)
        )
        if len(trade_dates) < exit_shift + 1:
            raise ValueError("回测交易日不足")
        trained_until = str(predictor.model_meta.get('data_end_date', ''))
        if trained_until >= str(start_date):
            raise ValueError(
                f"模型训练截止日 {trained_until} 覆盖回测起点 {start_date}，"
                "禁止前视回测；请使用 walkforward 模式"
            )

        hold_days = exit_shift - entry_shift
        rebalance_days = hold_days
        nav = float(config.BACKTEST_INITIAL_CAPITAL)
        nav_curve = [nav]
        period_returns = []
        filled_orders = 0
        rejected_orders = 0
        delayed_exits = 0
        unclosed_positions = 0

        i = 0
        stop_for_unclosed = False
        while i < len(trade_dates) - hold_days - 1:
            signal_date = trade_dates[i]
            entry_date = trade_dates[i + 1]
            exit_idx = i + 1 + hold_days
            exit_date = trade_dates[exit_idx]
            print(
                f"\n回测: {signal_date}预测 → {entry_date}开盘买入 "
                f"→ {exit_date}收盘卖出"
            )
            df_selected = predictor.predict_and_select(
                trade_date=signal_date,
                top_n=config.TOP_N_STOCKS,
                sector_neutral=True
            )
            if df_selected.empty:
                period_returns.append(0.0)
                nav_curve.extend([nav] * max(rebalance_days, 1))
                i += max(rebalance_days, 1)
                continue

            ts_codes = df_selected['股票代码'].tolist()
            period_start_nav = nav
            slot_value = nav / max(config.TOP_N_STOCKS, 1)
            trades = []
            max_delay = int(getattr(config, 'BACKTEST_MAX_EXIT_DELAY_DAYS', 10))
            end_idx = min(exit_idx + max_delay, len(trade_dates) - 1)
            for ts_code in ts_codes:
                bars = self.loader.get_stock_daily(
                    ts_code, start_date=entry_date, end_date=trade_dates[end_idx]
                )
                trade = self._simulate_stock_trade(
                    bars, entry_date, exit_date, slot_value
                )
                if trade is None:
                    rejected_orders += 1
                    continue
                trades.append(trade)
                filled_orders += 1
                delayed_exits += int(trade['delayed'])
                if not trade['closed']:
                    unclosed_positions += 1
                    stop_for_unclosed = True

            if not trades:
                period_returns.append(0.0)
                nav_curve.extend([nav] * max(rebalance_days, 1))
                i += max(rebalance_days, 1)
                continue

            cash = nav - sum(trade['cash_out'] for trade in trades)
            last_values = {
                id(trade): trade['cash_out'] for trade in trades
            }
            period_end_date = max(trade['exit_date'] for trade in trades)
            period_dates = [
                date for date in trade_dates
                if entry_date <= date <= period_end_date
            ]
            for current_date in period_dates:
                day_nav = cash
                for trade in trades:
                    if current_date >= trade['exit_date']:
                        value = trade['cash_in']
                    elif current_date in trade['daily_values']:
                        value = trade['daily_values'][current_date]
                        last_values[id(trade)] = value
                    else:
                        value = last_values[id(trade)]
                    day_nav += value
                nav_curve.append(day_nav)

            nav = cash + sum(trade['cash_in'] for trade in trades)
            if not np.isclose(nav, nav_curve[-1]):
                nav_curve.append(nav)
            period_return = nav / period_start_nav - 1
            period_returns.append(period_return)
            print(f"组合净收益: {period_return:.2%}, 净值: {nav:,.2f}")

            if stop_for_unclosed:
                print("存在无法卖出的持仓，资金继续占用，停止后续调仓")
                break
            i = trade_dates.index(period_end_date)

        returns = np.asarray(period_returns, dtype=float)
        total_return = nav / config.BACKTEST_INITIAL_CAPITAL - 1
        elapsed_days = max(len(nav_curve) - 1, 1)
        annual_return = (
            (1 + total_return) ** (250 / elapsed_days) - 1
            if total_return > -1 else -1.0
        )
        daily_nav = np.asarray(nav_curve, dtype=float)
        daily_returns = daily_nav[1:] / daily_nav[:-1] - 1
        std = daily_returns.std(ddof=1) if len(daily_returns) > 1 else 0.0
        sharpe_ratio = (
            daily_returns.mean() / std * np.sqrt(250) if std > 0 else 0.0
        )
        max_drawdown = self._calculate_max_drawdown(np.asarray(nav_curve))
        results = {
            '总收益率': total_return,
            '年化收益率': annual_return,
            '调仓次数': len(returns),
            '胜率': float(np.mean(returns > 0)) if len(returns) else 0.0,
            '平均持有期收益率': float(returns.mean()) if len(returns) else 0.0,
            '夏普比率': sharpe_ratio,
            '最大回撤': max_drawdown,
            '成交股票数': filled_orders,
            '未成交股票数': rejected_orders,
            '延迟卖出数': delayed_exits,
            '未平仓股票数': unclosed_positions,
            '存在前视偏差': False,
        }
        return results

    def _simulate_stock_trade(self, bars, entry_date, exit_date, position_value):
        """模拟单只股票，使用复权收益、整手交易和逐日盯市。"""
        if bars is None or bars.empty:
            return None
        bars = bars.sort_values('trade_date').reset_index(drop=True)
        entry_rows = bars[bars['trade_date'].astype(str) == str(entry_date)]
        if entry_rows.empty:
            return None
        entry = entry_rows.iloc[0]
        if self._is_locked_limit(entry, upward=True):
            return None

        daily_amount = float(entry.get('amount', 0.0)) * 1000.0
        max_ratio = float(getattr(config, 'BACKTEST_MAX_AMOUNT_RATIO', 0.01))

        exit_candidates = bars[bars['trade_date'].astype(str) >= str(exit_date)]
        exit_row = None
        for _, candidate in exit_candidates.iterrows():
            if not self._is_locked_limit(candidate, upward=False):
                exit_row = candidate
                break

        raw_entry_price = float(entry.get('raw_open', entry['open']))
        adjusted_entry_price = float(entry['open'])
        slippage = float(config.BACKTEST_SLIPPAGE)
        execution_entry_price = raw_entry_price * (1.0 + slippage)
        if execution_entry_price <= 0 or adjusted_entry_price <= 0:
            return None

        shares = int(position_value // (execution_entry_price * 100)) * 100
        if shares <= 0:
            return None
        buy_notional = shares * execution_entry_price

        commission_rate = float(config.BACKTEST_COMMISSION)
        min_commission = float(getattr(config, 'BACKTEST_MIN_COMMISSION', 0.0))
        transfer_rate = float(getattr(config, 'BACKTEST_TRANSFER_FEE', 0.0))

        buy_commission = max(buy_notional * commission_rate, min_commission)
        buy_transfer = buy_notional * transfer_rate
        cash_out = buy_notional + buy_commission + buy_transfer
        while cash_out > position_value and shares >= 200:
            shares -= 100
            buy_notional = shares * execution_entry_price
            buy_commission = max(buy_notional * commission_rate, min_commission)
            buy_transfer = buy_notional * transfer_rate
            cash_out = buy_notional + buy_commission + buy_transfer
        if cash_out > position_value:
            return None

        if daily_amount <= 0 or buy_notional > daily_amount * max_ratio:
            return None

        daily_values = {}
        for _, row in bars.iterrows():
            trade_date = str(row['trade_date'])
            if trade_date < str(entry_date):
                continue
            adjusted_close = float(row['close'])
            daily_values[trade_date] = (
                buy_notional * adjusted_close
                / (adjusted_entry_price * (1.0 + slippage))
            )

        closed = exit_row is not None
        if closed:
            actual_exit_date = str(exit_row['trade_date'])
            adjusted_exit_price = float(exit_row['close']) * (1.0 - slippage)
            gross_exit_value = (
                buy_notional * adjusted_exit_price
                / (adjusted_entry_price * (1.0 + slippage))
            )
            stamp_rate = self._stamp_duty_rate(actual_exit_date)
            sell_commission = max(gross_exit_value * commission_rate, min_commission)
            sell_fees = (
                gross_exit_value * (transfer_rate + stamp_rate)
                + sell_commission
            )
            cash_in = gross_exit_value - sell_fees
        else:
            if not daily_values:
                return None
            actual_exit_date = max(daily_values)
            cash_in = daily_values[actual_exit_date]

        return {
            'cash_out': cash_out,
            'cash_in': cash_in,
            'exit_date': actual_exit_date,
            'daily_values': daily_values,
            'delayed': actual_exit_date != str(exit_date),
            'closed': closed,
        }

    @staticmethod
    def _is_locked_limit(row, upward):
        high = float(row.get('raw_high', row['high']))
        low = float(row.get('raw_low', row['low']))
        pct = float(row.get('raw_pct_chg', row.get('pct_chg', 0.0)))
        threshold = float(config.LIMIT_THRESHOLD) * 100
        locked = np.isclose(high, low, rtol=0, atol=max(abs(high), 1.0) * 1e-6)
        return bool(locked and (pct >= threshold if upward else pct <= -threshold))

    @staticmethod
    def _stamp_duty_rate(trade_date):
        if trade_date < '20230828':
            return 0.001
        return float(getattr(config, 'BACKTEST_STAMP_DUTY', 0.0005))

    @staticmethod
    def _calculate_max_drawdown(nav_curve):
        """根据复利净值计算最大回撤。"""
        if len(nav_curve) == 0:
            return 0
        running_max = np.maximum.accumulate(nav_curve)
        drawdowns = nav_curve / running_max - 1.0
        return float(np.min(drawdowns))


def main():
    """测试预测功能"""
    # 创建预测器
    predictor = StockPredictor()

    # 预测并选股
    recommendations = predictor.predict_and_select()

    # 输出推荐股票
    if not recommendations.empty:
        print("\n" + "=" * 50)
        print(f"今日推荐买入股票 (Top {len(recommendations)})")
        print("=" * 50)
        print(recommendations.to_string(index=False))
    else:
        print("今日没有推荐股票")


if __name__ == "__main__":
    main()
