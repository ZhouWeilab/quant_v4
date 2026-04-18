"""
特征工程模块
功能：构建量价特征、均线特征、技术指标特征
"""

import pandas as pd
import numpy as np
import config


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

    def add_all_features(self, df):
        """
        添加所有特征

        Args:
            df: 日线数据

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

        return df

    def get_feature_columns(self, df):
        """
        获取特征列名（排除原始列和目标列）

        Args:
            df: 特征数据

        Returns:
            list: 特征列名列表
        """
        # 原始列（不作为特征）
        exclude_cols = [
            'ts_code', 'trade_date', 'open', 'high', 'low', 'close',
            'pre_close', 'change', 'pct_chg', 'vol', 'amount',
            'target', 'target_return'  # 目标列
        ]

        # 获取所有列
        all_cols = df.columns.tolist()

        # 特征列
        feature_cols = [col for col in all_cols if col not in exclude_cols]

        return feature_cols


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
