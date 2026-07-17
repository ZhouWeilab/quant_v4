"""
特征工程模块
功能：构建量价特征、均线特征、技术指标特征
"""

import pandas as pd
import numpy as np
import warnings
import re
import config


warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


class FeatureEngineer:
    """特征工程器"""

    def __init__(self):
        pass

    def add_returns(self, df):
        """
        计算收益率特征

        Args:
            df: 日线数据

        Returns:
            DataFrame: 添加收益率特征后的数据
        """
        # 1日收益率
        df['return_1d'] = df['close'].pct_change(1)

        # 多日收益率
        for period in [3, 5, 10, 20]:
            df[f'return_{period}d'] = df['close'].pct_change(period)

        return df

    def add_ma_features(self, df):
        """
        计算均线特征

        Args:
            df: 日线数据

        Returns:
            DataFrame: 添加均线特征后的数据
        """
        # 价格均线
        for period in config.MA_PERIODS:
            df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
            # 价格相对均线的位置
            df[f'close_ma_{period}_ratio'] = df['close'] / df[f'ma_{period}'] - 1

        # 成交量均线
        for period in config.VOLUME_MA_PERIODS:
            df[f'vol_ma_{period}'] = df['vol'].rolling(window=period).mean()
            # 成交量相对均线的比值
            df[f'vol_ma_{period}_ratio'] = df['vol'] / df[f'vol_ma_{period}']

        # 均线多头排列特征
        df['ma_alignment'] = 0
        if len(config.MA_PERIODS) >= 3:
            short_ma = f'ma_{config.MA_PERIODS[0]}'
            mid_ma = f'ma_{config.MA_PERIODS[1]}'
            long_ma = f'ma_{config.MA_PERIODS[2]}'

            df.loc[
                (df[short_ma] > df[mid_ma]) & (df[mid_ma] > df[long_ma]),
                'ma_alignment'
            ] = 1

        return df

    def add_price_features(self, df):
        """
        计算价格相关特征

        Args:
            df: 日线数据

        Returns:
            DataFrame: 添加价格特征后的数据
        """
        # 振幅
        df['amplitude'] = (df['high'] - df['low']) / df['pre_close']

        # 上影线比例
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-6)

        # 下影线比例
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-6)

        # 实体比例
        df['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-6)

        # 收盘价位置（在当日最高最低价之间的位置）
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-6)

        # 多日最高价/最低价
        for period in [5, 10, 20]:
            df[f'high_{period}d'] = df['high'].rolling(window=period).max()
            df[f'low_{period}d'] = df['low'].rolling(window=period).min()
            # 当前价格相对多日区间的位置
            df[f'price_position_{period}d'] = (
                (df['close'] - df[f'low_{period}d']) /
                (df[f'high_{period}d'] - df[f'low_{period}d'] + 1e-6)
            )

        return df

    def add_volume_features(self, df):
        """
        计算成交量相关特征

        Args:
            df: 日线数据

        Returns:
            DataFrame: 添加成交量特征后的数据
        """
        # 成交量变化率
        df['vol_change'] = df['vol'].pct_change(1)

        # 量价关系
        df['price_volume_corr'] = df['close'].rolling(window=20).corr(df['vol'])

        # 成交额变化
        df['amount_change'] = df['amount'].pct_change(1)

        # 换手率（如果有）
        if 'turnover_rate' in df.columns:
            df['turnover_ma5'] = df['turnover_rate'].rolling(window=5).mean()
            df['turnover_std5'] = df['turnover_rate'].rolling(window=5).std()

        return df

    def add_rsi(self, df, period=None):
        """
        计算 RSI 指标

        Args:
            df: 日线数据
            period: RSI 周期

        Returns:
            DataFrame: 添加 RSI 指标后的数据
        """
        if period is None:
            period = config.RSI_PERIOD

        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / (loss + 1e-6)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        return df

    def add_macd(self, df):
        """
        计算 MACD 指标

        Args:
            df: 日线数据

        Returns:
            DataFrame: 添加 MACD 指标后的数据
        """
        # 计算 EMA
        ema_fast = df['close'].ewm(span=config.MACD_FAST, adjust=False).mean()
        ema_slow = df['close'].ewm(span=config.MACD_SLOW, adjust=False).mean()

        # DIF
        df['macd_dif'] = ema_fast - ema_slow

        # DEA
        df['macd_dea'] = df['macd_dif'].ewm(span=config.MACD_SIGNAL, adjust=False).mean()

        # MACD
        df['macd'] = (df['macd_dif'] - df['macd_dea']) * 2

        # MACD 柱状图变化
        df['macd_change'] = df['macd'].diff()

        # 绝对价格口径跨股票、跨年份不可比，训练只使用归一化版本。
        close = df['close'].replace(0, np.nan)
        df['macd_dif_norm'] = df['macd_dif'] / close
        df['macd_dea_norm'] = df['macd_dea'] / close
        df['macd_norm'] = df['macd'] / close
        df['macd_change_norm'] = df['macd_change'] / close

        return df

    def add_boll(self, df):
        """
        计算布林带指标

        Args:
            df: 日线数据

        Returns:
            DataFrame: 添加布林带指标后的数据
        """
        # 中轨
        df['boll_mid'] = df['close'].rolling(window=config.BOLL_PERIOD).mean()

        # 标准差
        std = df['close'].rolling(window=config.BOLL_PERIOD).std()

        # 上轨
        df['boll_upper'] = df['boll_mid'] + config.BOLL_STD * std

        # 下轨
        df['boll_lower'] = df['boll_mid'] - config.BOLL_STD * std

        # 布林带宽度
        df['boll_width'] = (df['boll_upper'] - df['boll_lower']) / df['boll_mid']

        # 价格在布林带中的位置
        df['boll_position'] = (
            (df['close'] - df['boll_lower']) /
            (df['boll_upper'] - df['boll_lower'] + 1e-6)
        )

        return df

    def add_atr(self, df, period=None):
        """
        计算 ATR 指标（真实波动幅度）

        Args:
            df: 日线数据
            period: ATR 周期

        Returns:
            DataFrame: 添加 ATR 指标后的数据
        """
        if period is None:
            period = config.ATR_PERIOD

        # 真实波动幅度
        df['tr'] = df[['high', 'pre_close']].max(axis=1) - df[['low', 'pre_close']].min(axis=1)

        # ATR
        df[f'atr_{period}'] = df['tr'].rolling(window=period).mean()

        # 归一化 ATR
        df[f'atr_{period}_norm'] = df[f'atr_{period}'] / df['close']

        return df

    def add_momentum_features(self, df):
        """
        计算动量特征

        Args:
            df: 日线数据

        Returns:
            DataFrame: 添加动量特征后的数据
        """
        # 价格动量
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
            df[f'momentum_{period}_norm'] = df[f'momentum_{period}'] / df['close'].shift(period)

        # 加速度（动量的变化）
        df['acceleration_5'] = df['momentum_5'].diff()
        df['acceleration_5_norm'] = (
            df['acceleration_5'] / df['close'].replace(0, np.nan)
        )

        return df

    def add_volatility_features(self, df):
        """
        计算波动率特征

        Args:
            df: 日线数据

        Returns:
            DataFrame: 添加波动率特征后的数据
        """
        # 收益率波动率
        returns = df['close'].pct_change()

        for period in [5, 10, 20]:
            df[f'volatility_{period}'] = returns.rolling(window=period).std()

        # 振幅波动率
        for period in [5, 10, 20]:
            df[f'amplitude_volatility_{period}'] = df['amplitude'].rolling(window=period).std()

        return df

    def add_fundamental_features(self, df):
        """
        基本面/情绪指标特征（来自 daily_basic）
        包含：PE、PB、PS、换手率、量比等
        """
        # PE 相关
        if 'pe' in df.columns:
            df['pe_inv'] = 1.0 / df['pe'].replace(0, np.nan)
            df['pe_ttm_inv'] = 1.0 / df['pe_ttm'].replace(0, np.nan)
            df['pe_rank_20'] = df['pe'].rolling(20).apply(
                lambda x: (x.rank().iloc[-1] - 1) / max(len(x) - 1, 1), raw=False
            )

        # PB 相关
        if 'pb' in df.columns:
            df['pb_inv'] = 1.0 / df['pb'].replace(0, np.nan)
            df['pb_rank_20'] = df['pb'].rolling(20).apply(
                lambda x: (x.rank().iloc[-1] - 1) / max(len(x) - 1, 1), raw=False
            )

        # 换手率
        if 'turnover_rate' in df.columns:
            df['turnover_rate'] = pd.to_numeric(df['turnover_rate'], errors='coerce')
            df['turnover_ma5'] = df['turnover_rate'].rolling(5).mean()
            df['turnover_ma20'] = df['turnover_rate'].rolling(20).mean()
            df['turnover_ratio'] = df['turnover_rate'] / (df['turnover_ma20'] + 1e-6)
            df['turnover_zscore'] = (
                (df['turnover_rate'] - df['turnover_ma20']) /
                (df['turnover_rate'].rolling(20).std() + 1e-6)
            )

        # 量比
        if 'volume_ratio' in df.columns:
            df['volume_ratio'] = pd.to_numeric(df['volume_ratio'], errors='coerce')
            df['volume_ratio_ma5'] = df['volume_ratio'].rolling(5).mean()

        return df

    def add_moneyflow_features(self, df):
        """
        资金流向因子（来自 Tushare moneyflow）
        A股最有效的短线因子之一
        """
        # 超大单净流入占比
        if 'buy_elg_vol' in df.columns and 'sell_elg_vol' in df.columns and 'vol' in df.columns:
            df['elg_net_ratio'] = (df['buy_elg_vol'] - df['sell_elg_vol']) / (df['vol'] + 1e-6)
            df['elg_buy_ratio'] = df['buy_elg_vol'] / (df['vol'] + 1e-6)

        # 主力（大单+超大单）净流入占比
        if all(c in df.columns for c in ['buy_elg_vol', 'sell_elg_vol', 'buy_lg_vol', 'sell_lg_vol']):
            df['main_net_ratio'] = (
                (df['buy_elg_vol'] + df['buy_lg_vol'] - df['sell_elg_vol'] - df['sell_lg_vol'])
                / (df['vol'] + 1e-6)
            )

        # 散户（小单）净流入占比（反向指标）
        if 'buy_sm_vol' in df.columns and 'sell_sm_vol' in df.columns:
            df['sm_net_ratio'] = (df['buy_sm_vol'] - df['sell_sm_vol']) / (df['vol'] + 1e-6)

        # 净流入金额占比
        if 'net_mf_amount' in df.columns and 'amount' in df.columns:
            df['net_mf_ratio'] = df['net_mf_amount'] / (df['amount'] + 1e-6)

        # 资金流向 5 日累计（动量）
        for col in ['elg_net_ratio', 'main_net_ratio', 'net_mf_ratio']:
            if col in df.columns:
                df[f'{col}_cum5'] = df[col].rolling(5).sum()
                df[f'{col}_cum10'] = df[col].rolling(10).sum()

        return df

    def add_top_list_features(self, df):
        """
        龙虎榜因子（来自 Tushare top_list）
        """
        # 是否上榜
        if 'net_amount' in df.columns:
            net_amount = pd.to_numeric(df['net_amount'], errors='coerce').fillna(0.0)
            listed = net_amount.abs() > 0
            if 'l_amount' in df.columns:
                listed |= (
                    pd.to_numeric(df['l_amount'], errors='coerce')
                    .fillna(0.0)
                    .gt(0)
                )
            df['is_top_list'] = listed.astype(int)
            # 龙虎榜净买入占成交额比例
            if 'amount' in df.columns:
                df['top_net_ratio'] = df['net_amount'] / (df['amount'] + 1e-6)
            # 龙虎榜成交额占全日比例
            if 'l_amount' in df.columns and 'amount' in df.columns:
                df['top_amount_ratio'] = df['l_amount'] / (df['amount'] + 1e-6)
            # 近 5 日上榜次数
            df['top_list_count_5d'] = df['is_top_list'].rolling(5).sum()

        return df

    def add_limit_features(self, df):
        """
        涨跌停因子（来自 Tushare limit_list）
        """
        # 封单比例（封单量 / 成交量）
        if 'fc_ratio' in df.columns:
            df['fc_ratio'] = pd.to_numeric(df['fc_ratio'], errors='coerce')
            df['fc_ratio_ma3'] = df['fc_ratio'].rolling(3).mean()

        # 封单金额
        if 'fd_amount' in df.columns:
            df['fd_amount'] = pd.to_numeric(df['fd_amount'], errors='coerce')
            if 'amount' in df.columns:
                df['fd_amount_ratio'] = df['fd_amount'] / (df['amount'] + 1e-6)

        # 涨停强度
        if 'strth' in df.columns:
            df['strth'] = pd.to_numeric(df['strth'], errors='coerce')

        # 开板次数（越低越好）
        if 'open_times' in df.columns:
            df['open_times'] = pd.to_numeric(df['open_times'], errors='coerce')

        # 是否涨停/跌停
        if 'limit' in df.columns:
            df['is_limit_up'] = (df['limit'] == 'U').astype(int)
            df['is_limit_down'] = (df['limit'] == 'D').astype(int)

        return df

    def add_alt_features(self, df):
        """
        另类因子（非传统技术指标）
        """
        # 1. 开盘跳空缺口
        df['gap'] = (df['open'] - df['pre_close']) / df['pre_close']

        # 2. 日内趋势强度
        df['intraday_trend'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-6)

        # 3. 量价背离
        df['pv_divergence'] = (
            df['close'].rolling(5).corr(df['vol']) *
            df['close'].rolling(5).std() /
            (df['close'].rolling(5).mean() + 1e-6)
        )

        # 4. 波动率聚集
        returns = df['close'].pct_change()
        df['vol_cluster'] = returns.rolling(5).std() / (returns.rolling(20).std() + 1e-6)

        # 5. 涨跌成交量不对称性
        df['up_volume'] = np.where(df['close'] > df['open'], df['vol'], 0)
        df['down_volume'] = np.where(df['close'] <= df['open'], df['vol'], 0)
        df['volume_asym'] = (
            df['up_volume'].rolling(10).sum() - df['down_volume'].rolling(10).sum()
        ) / (df['vol'].rolling(10).sum() + 1e-6)

        # 6. 连续上涨/下跌天数
        df['consecutive_up'] = 0
        df['consecutive_down'] = 0
        for i in range(1, len(df)):
            if df.loc[i, 'close'] > df.loc[i - 1, 'close']:
                df.loc[i, 'consecutive_up'] = df.loc[i - 1, 'consecutive_up'] + 1
                df.loc[i, 'consecutive_down'] = 0
            elif df.loc[i, 'close'] < df.loc[i - 1, 'close']:
                df.loc[i, 'consecutive_down'] = df.loc[i - 1, 'consecutive_down'] + 1
                df.loc[i, 'consecutive_up'] = 0

        return df

    def add_tradable_alpha_features(self, df):
        """Add short-term tradable alpha features using only current/past bars."""
        close = df['close'].replace(0, np.nan)
        pre_close = df['pre_close'].replace(0, np.nan)
        amount = df['amount'].replace(0, np.nan)
        vol = df['vol'].replace(0, np.nan)
        day_range = (df['high'] - df['low']).replace(0, np.nan)
        ret = df['close'].pct_change()

        df['amount_ma5'] = df['amount'].rolling(5).mean()
        df['amount_ma20'] = df['amount'].rolling(20).mean()
        df['amount_log_change'] = np.log1p(amount).diff()
        df['amount_ratio_5_20'] = df['amount_ma5'] / (df['amount_ma20'] + 1e-6)
        df['amount_zscore_20'] = (df['amount'] - df['amount_ma20']) / (df['amount'].rolling(20).std() + 1e-6)
        df['amount_cv_20'] = df['amount'].rolling(20).std() / (df['amount_ma20'] + 1e-6)
        df['vol_ratio_1_5'] = vol / (df['vol'].rolling(5).mean() + 1e-6)
        df['vol_ratio_5_20'] = df['vol'].rolling(5).mean() / (df['vol'].rolling(20).mean() + 1e-6)

        high20_prev = df['high'].rolling(20).max().shift(1)
        low20_prev = df['low'].rolling(20).min().shift(1)
        high60_prev = df['high'].rolling(60).max().shift(1)
        df['breakout_20'] = close / (high20_prev + 1e-6) - 1
        df['breakout_60'] = close / (high60_prev + 1e-6) - 1
        df['range_position_20'] = (close - low20_prev) / (high20_prev - low20_prev + 1e-6)
        df['pullback_from_20_high'] = close / (high20_prev + 1e-6) - 1
        df['drawdown_10'] = close / (df['close'].rolling(10).max() + 1e-6) - 1
        df['drawdown_20'] = close / (df['close'].rolling(20).max() + 1e-6) - 1

        df['open_gap'] = df['open'] / pre_close - 1
        df['close_to_high'] = close / df['high'].replace(0, np.nan) - 1
        df['close_to_low'] = close / df['low'].replace(0, np.nan) - 1
        df['upper_shadow_ret'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (pre_close + 1e-6)
        df['lower_shadow_ret'] = (df[['open', 'close']].min(axis=1) - df['low']) / (pre_close + 1e-6)
        df['body_to_range'] = (df['close'] - df['open']) / (day_range + 1e-6)
        df['close_strength'] = (df['close'] - df['low']) / (day_range + 1e-6)
        df['fade_from_high'] = (df['close'] - df['high']) / (pre_close + 1e-6)

        df['ret_sum_3'] = ret.rolling(3).sum()
        df['ret_sum_5'] = ret.rolling(5).sum()
        df['ret_std_5'] = ret.rolling(5).std()
        df['ret_skew_10'] = ret.rolling(10).skew()
        df['up_day_ratio_5'] = (ret > 0).rolling(5).mean()
        df['up_day_ratio_10'] = (ret > 0).rolling(10).mean()
        df['max_ret_5'] = ret.rolling(5).max()
        df['min_ret_5'] = ret.rolling(5).min()

        df['dist_to_10pct_limit'] = 0.10 - (close / pre_close - 1)
        df['dist_to_20pct_limit'] = 0.20 - (close / pre_close - 1)
        df['near_limit_up_10'] = ((close / pre_close - 1) > 0.085).astype(int)
        df['near_limit_down_10'] = ((close / pre_close - 1) < -0.085).astype(int)

        if 'turnover_rate' in df.columns:
            turnover = pd.to_numeric(df['turnover_rate'], errors='coerce')
            df['turnover_change_1d'] = turnover.diff()
            df['turnover_ratio_1_5'] = turnover / (turnover.rolling(5).mean() + 1e-6)
            df['turnover_crowding_20'] = (turnover - turnover.rolling(20).mean()) / (turnover.rolling(20).std() + 1e-6)

        return df

    def add_all_features(self, df):
        """
        添加所有特征

        Args:
            df: 日线数据（含 daily_basic / moneyflow / top_list / limit_list 合并后的列）

        Returns:
            DataFrame: 添加所有特征后的数据
        """
        df = df.copy()

        # 确保数据按日期排序
        df = df.sort_values('trade_date').reset_index(drop=True)

        # 基础特征
        df = self.add_returns(df)
        df = self.add_price_features(df)
        df = self.add_volume_features(df)

        # 均线特征
        df = self.add_ma_features(df)

        # 技术指标
        df = self.add_rsi(df)
        df = self.add_macd(df)
        df = self.add_boll(df)
        df = self.add_atr(df)

        # 高级特征
        df = self.add_momentum_features(df)
        df = self.add_volatility_features(df)

        # 基本面/情绪指标
        df = self.add_fundamental_features(df)

        # 资金流向（新增，A股最强短线因子之一）
        df = self.add_moneyflow_features(df)

        # 龙虎榜（新增）
        df = self.add_top_list_features(df)

        # 涨跌停（新增）
        df = self.add_limit_features(df)

        # 另类因子
        df = self.add_alt_features(df)

        # 实盘可交易短线因子
        df = self.add_tradable_alpha_features(df)

        return df

    def get_feature_columns(self, df):
        """
        获取特征列名（排除原始列、目标列、元数据列）

        Args:
            df: 特征数据

        Returns:
            list: 特征列名列表（仅数值型因子）
        """
        # 排除列：原始行情列、目标列、元数据列
        exclude_cols = [
            'ts_code', 'trade_date', 'open', 'high', 'low', 'close',
            'pre_close', 'change', 'pct_chg', 'vol', 'amount',
            'raw_open', 'raw_high', 'raw_low', 'raw_close', 'raw_pre_close',
            'raw_change', 'raw_pct_chg',
            'adj_factor', 'listing_age_days', 'is_st', 'universe_eligible',
            'target', 'target_return', 'target_raw', 'target_excess_return',
            'sector', 'market_cap', 'name', 'historical_name', 'industry',
            'area', 'list_date', 'delist_date', 'list_status'
        ]

        # 仅保留数值列作为特征
        all_cols = df.columns.tolist()
        feature_cols = []
        for col in all_cols:
            if col in exclude_cols:
                continue
            if (
                getattr(config, 'EXCLUDE_NONSTATIONARY_FEATURES', True)
                and self._is_nonstationary_feature(col)
            ):
                continue
            # 额外检查：列必须是数值型
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_cols.append(col)

        return feature_cols

    @staticmethod
    def _is_nonstationary_feature(col):
        """排除绝对价格、绝对成交量和原始规模字段，仅保留可比口径。"""
        exact = {
            'boll_mid', 'boll_upper', 'boll_lower', 'tr',
            'macd_dif', 'macd_dea', 'macd', 'macd_change',
            'acceleration_5', 'up_volume', 'down_volume',
            'amount_ma5', 'amount_ma20',
            'buy', 'sell', 'net_amount', 'l_amount', 'fd_amount',
            'total_share', 'float_share', 'free_share',
            'total_mv', 'circ_mv',
        }
        if col in exact:
            return True
        if re.match(
            r'^(ma_\d+|vol_ma_\d+|high_\d+d|low_\d+d|atr_\d+|momentum_\d+)$',
            col
        ):
            return True
        if col.startswith(('buy_', 'sell_')) and col.endswith(('_vol', '_amount')):
            return True
        if col.endswith(('_vol', '_amount', '_share', '_mv')):
            return not col.endswith(('_ratio', '_cs_rank'))
        return False


def main():
    """测试特征工程功能"""
    from data_loader import DataLoader

    loader = DataLoader()
    engineer = FeatureEngineer()

    # 获取测试数据
    trade_date = loader.get_latest_trade_date()
    stock_list = loader.get_stock_list(trade_date)

    if not stock_list.empty:
        test_code = stock_list.iloc[0]['ts_code']
        test_name = stock_list.iloc[0]['name']

        df = loader.get_stock_daily(test_code)

        if df is not None:
            print(f"原始数据: {len(df)} 行, {len(df.columns)} 列")

            # 添加特征
            df_features = engineer.add_all_features(df)

            print(f"特征数据: {len(df_features)} 行, {len(df_features.columns)} 列")

            # 获取特征列
            feature_cols = engineer.get_feature_columns(df_features)
            print(f"特征数量: {len(feature_cols)}")
            print(f"特征列: {feature_cols[:10]}...")

            # 显示样例
            print(f"\n{test_code} {test_name} 特征样例:")
            print(df_features[['trade_date', 'close'] + feature_cols[:5]].tail())


if __name__ == "__main__":
    main()
