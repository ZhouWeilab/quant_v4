"""
预测选股模块
功能：加载模型，对最新数据预测次日涨幅，排序选出 Top N 股票
"""

import pandas as pd
import numpy as np
import config
from data_loader import DataLoader
from dataset import DatasetBuilder
from model import QuantModel


class StockPredictor:
    """股票预测器"""

    def __init__(self):
        self.loader = DataLoader()
        self.builder = DatasetBuilder()
        self.model = QuantModel()

        # 加载模型和标准化器
        self.load_model()

    def load_model(self):
        """加载训练好的模型和标准化器"""
        try:
            # 加载模型
            self.model.load(config.MODEL_FILE)

            # 加载标准化器和特征列
            self.builder.load_scaler(
                config.SCALER_FILE,
                config.FEATURE_COLS_FILE
            )

            print("模型和标准化器加载成功")

        except Exception as e:
            raise ValueError(f"加载模型失败: {e}，请先训练模型")

    def get_candidate_stocks(self, trade_date=None):
        """
        获取候选股票

        Args:
            trade_date: 交易日期，默认为最近一个交易日

        Returns:
            dict: 符合条件的股票数据
        """
        if trade_date is None:
            trade_date = self.loader.get_latest_trade_date()

        print(f"\n获取 {trade_date} 的候选股票...")

        # 获取股票列表
        stock_list = self.loader.get_stock_list(trade_date)

        # 获取日线数据
        stock_data = self.loader.get_all_stocks_daily(
            stock_list,
            start_date=config.START_DATE,
            end_date=trade_date
        )

        # 流动性筛选
        stock_data = self.loader.filter_by_liquidity(stock_data, trade_date)

        # 波动率筛选
        stock_data = self.loader.filter_by_volatility(stock_data)

        return stock_data

    def predict_returns(self, stock_data):
        """
        预测所有股票的次日收益率

        Args:
            stock_data: 股票数据字典

        Returns:
            DataFrame: 预测结果（包含股票代码、日期、收盘价、预测收益率）
        """
        print("\n预测次日收益率...")

        # 准备预测数据
        X, df_info = self.builder.prepare_predict_data(stock_data)

        # 预测
        predictions = self.model.predict(X)

        # 组织结果
        df_info['predicted_return'] = predictions

        return df_info

    def rank_and_select(self, df_predictions, top_n=None):
        if top_n is None:
            top_n = config.TOP_N_STOCKS

    # 直接按预测值排序，不再过滤 < 1%
        df_sorted = df_predictions.sort_values('predicted_return', ascending=False).reset_index(drop=True)
        df_top = df_sorted.head(top_n)
    
        print(f"Top {top_n} 股票预测涨幅范围: {df_top['predicted_return'].min():.4f} ~ {df_top['predicted_return'].max():.4f}")
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
        stock_basic = self.loader.pro.stock_basic(
            exchange='',
            list_status='L',
            fields='ts_code,symbol,name,industry'
        )

        # 合并
        df_result = df_top.merge(
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
            'ts_code',
            'name',
            'industry',
            'close',
            'predicted_return'
        ]

        df_output = df_output[output_cols]

        # 重命名列
        df_output.columns = [
            '股票代码',
            '股票名称',
            '行业',
            '当前价格',
            '预测涨幅'
        ]

        # 格式化数值
        df_output['当前价格'] = df_output['当前价格'].round(2)
        df_output['预测涨幅'] = (df_output['预测涨幅'] * 100).round(2).astype(str) + '%'

        return df_output

    def predict_and_select(self, trade_date=None, top_n=None):
        """
        预测并选股（完整流程）

        Args:
            trade_date: 交易日期
            top_n: 选择的股票数量

        Returns:
            DataFrame: 推荐股票列表
        """
        print("=" * 50)
        print("开始预测和选股")
        print("=" * 50)

        # 获取候选股票
        stock_data = self.get_candidate_stocks(trade_date)

        if not stock_data:
            print("没有符合条件的股票")
            return pd.DataFrame()

        # 预测
        df_predictions = self.predict_returns(stock_data)

        # 排序选择
        df_top = self.rank_and_select(df_predictions, top_n)

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
    """简单回测引擎"""

    def __init__(self):
        self.loader = DataLoader()

    def run_simple_backtest(self, start_date=None, end_date=None):
        """
        运行简单回测

        Args:
            start_date: 回测起始日期
            end_date: 回测结束日期

        Returns:
            dict: 回测结果统计
        """
        if start_date is None:
            start_date = config.BACKTEST_START_DATE

        if end_date is None:
            end_date = self.loader.get_latest_trade_date()

        print("\n" + "=" * 50)
        print(f"回测期间: {start_date} - {end_date}")
        print("=" * 50)

        # 获取交易日历
        trade_cal = self.loader.get_trade_cal(start_date, end_date)
        trade_dates = trade_cal['cal_date'].tolist()

        # 初始化统计
        total_return = 0
        win_count = 0
        loss_count = 0
        daily_returns = []

        predictor = StockPredictor()

        # 遍历每个交易日
        for i in range(len(trade_dates) - 1):
            current_date = trade_dates[i]
            next_date = trade_dates[i + 1]

            print(f"\n回测日期: {current_date}")

            try:
                # 预测选股
                df_selected = predictor.predict_and_select(
                    trade_date=current_date,
                    top_n=config.TOP_N_STOCKS
                )

                if df_selected.empty:
                    continue

                # 计算次日收益
                ts_codes = df_selected['股票代码'].tolist()
                day_returns = []

                for ts_code in ts_codes:
                    # 获取次日数据
                    df = self.loader.get_stock_daily(
                        ts_code,
                        start_date=next_date,
                        end_date=next_date
                    )

                    if df is not None and not df.empty:
                        # 计算收益率
                        actual_return = df.iloc[0]['pct_chg'] / 100
                        day_returns.append(actual_return)

                if day_returns:
                    # 平均收益率（等权重）
                    avg_return = np.mean(day_returns)
                    daily_returns.append(avg_return)
                    total_return += avg_return

                    if avg_return > 0:
                        win_count += 1
                    else:
                        loss_count += 1

                    print(f"当日收益率: {avg_return:.2%}")

            except Exception as e:
                print(f"回测失败 {current_date}: {e}")
                continue

        # 统计结果
        total_days = len(daily_returns)
        win_rate = win_count / total_days if total_days > 0 else 0
        avg_daily_return = total_return / total_days if total_days > 0 else 0

        # 年化收益率（假设250个交易日）
        annual_return = avg_daily_return * 250

        # 夏普比率
        if len(daily_returns) > 0:
            daily_std = np.std(daily_returns)
            sharpe_ratio = avg_daily_return / daily_std * np.sqrt(250) if daily_std > 0 else 0
        else:
            sharpe_ratio = 0

        # 最大回撤
        cumulative_returns = np.cumsum(daily_returns)
        max_drawdown = self._calculate_max_drawdown(cumulative_returns)

        results = {
            '总收益率': total_return,
            '年化收益率': annual_return,
            '交易天数': total_days,
            '胜率': win_rate,
            '平均日收益率': avg_daily_return,
            '夏普比率': sharpe_ratio,
            '最大回撤': max_drawdown
        }

        return results

    def _calculate_max_drawdown(self, cumulative_returns):
        """计算最大回撤"""
        if len(cumulative_returns) == 0:
            return 0

        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        max_drawdown = np.min(drawdowns)

        return max_drawdown


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
