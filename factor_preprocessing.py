"""
因子预处理模块
功能：MAD 去极值、滚动标准化、行业/市值中性化
严格遵循时序隔离原则：所有统计量只用过去已知数据计算
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import config


class FactorPreprocessor:
    """因子预处理器"""

    def __init__(self, mad_n=3, zscore_window=60, neutralize_ridge_alpha=1.0):
        """
        初始化

        Args:
            mad_n: MAD 去极值倍数（默认 3）
            zscore_window: 滚动标准化窗口（交易日，默认 60）
            neutralize_ridge_alpha: 中性化回归的 Ridge 正则化强度
        """
        self.mad_n = mad_n
        self.zscore_window = zscore_window
        self.neutralize_ridge_alpha = neutralize_ridge_alpha

        # 记录训练状态（用于预测时的 transform）
        self.fitted_stats = {}

    @staticmethod
    def _mad_winsorize_series(x, n=3):
        """
        对单序列做 MAD 去极值

        Args:
            x: Series 或 ndarray
            n: MAD 倍数

        Returns:
            ndarray: 去极值后的序列
        """
        x = np.asarray(x, dtype=float)
        median = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - median))
        if mad < 1e-10:
            return x
        upper = median + n * 1.4826 * mad
        lower = median - n * 1.4826 * mad
        return np.clip(x, lower, upper)

    def mad_winsorize(self, df, factor_cols, date_col='trade_date'):
        """
        截面 MAD 去极值：每天对每个因子单独处理
        使用 groupby.transform 向量化实现，避免 Python 层循环
        """
        df = df.copy()
        for col in factor_cols:
            def _mad_clip(x):
                x_arr = np.asarray(x, dtype=float)
                median = np.nanmedian(x_arr)
                mad = np.nanmedian(np.abs(x_arr - median))
                if mad < 1e-10:
                    return x_arr
                upper = median + self.mad_n * 1.4826 * mad
                lower = median - self.mad_n * 1.4826 * mad
                return np.clip(x_arr, lower, upper)
            df[col] = df.groupby(date_col)[col].transform(_mad_clip)
        return df

    def rolling_zscore(self, df, factor_cols, date_col='trade_date',
                       code_col='ts_code', window=None, min_periods=20):
        """
        滚动 Z-score 标准化：每个截面用过去 window 天的截面均值和标准差
        优化版：先计算每日截面统计量，再做 rolling，避免双重循环筛选数据
        """
        if window is None:
            window = self.zscore_window

        df = df.copy()
        df = df.sort_values([date_col, code_col]).reset_index(drop=True)

        # 1. 计算每日截面均值和标准差
        daily_mean = df.groupby(date_col)[factor_cols].mean()
        daily_std = df.groupby(date_col)[factor_cols].std().replace(0, np.nan)

        # 2. 对每日统计量做 rolling mean（过去 window 天）
        rolling_mean = daily_mean.rolling(window=window, min_periods=min_periods).mean()
        rolling_std = daily_std.rolling(window=window, min_periods=min_periods).mean()

        # 3. 把滚动统计量 merge 回原表
        rolling_mean = rolling_mean.add_suffix('_roll_mean').reset_index()
        rolling_std = rolling_std.add_suffix('_roll_std').reset_index()

        df = df.merge(rolling_mean, on=date_col, how='left')
        df = df.merge(rolling_std, on=date_col, how='left')

        # 4. 计算 Z-score
        for col in factor_cols:
            mean_col = f'{col}_roll_mean'
            std_col = f'{col}_roll_std'
            df[f'{col}_z'] = (df[col] - df[mean_col]) / (df[std_col] + 1e-10)
            df.drop(columns=[mean_col, std_col], inplace=True, errors='ignore')

        return df

    def simple_cross_sectional_zscore(self, df, factor_cols, date_col='trade_date'):
        """
        简单截面 Z-score（每天独立标准化）
        用于快速处理，不保证时序一致性，但计算快

        Args:
            df: DataFrame
            factor_cols: 因子列名
            date_col: 日期列名

        Returns:
            DataFrame: 新增 *_z 列
        """
        df = df.copy()
        for col in factor_cols:
            df[f'{col}_z'] = df.groupby(date_col)[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-10)
            )
        return df

    def neutralize(self, df, factor_cols, industry_col='sector',
                   market_cap_col='market_cap', date_col='trade_date'):
        """
        行业/市值中性化：每天截面上用 Ridge 回归，取残差
        优化版：每天对所有因子批量计算，只做一次矩阵求逆
        """
        df = df.copy()
        dates = sorted(df[date_col].unique())

        # 对市值取对数
        log_cap_col = 'log_market_cap'
        df[log_cap_col] = np.log(df[market_cap_col].replace(0, np.nan))

        for col in factor_cols:
            df[f'{col}_neu'] = np.nan

        for date in dates:
            day_df = df[df[date_col] == date].copy()
            if len(day_df) < 20:
                continue

            # 去掉缺失值
            valid = day_df[factor_cols + [industry_col, log_cap_col]].notna().all(axis=1)
            day_df = day_df[valid]
            if len(day_df) < 20:
                continue

            # 生成行业哑变量（去掉一个避免完全共线性）
            sector_dummies = pd.get_dummies(day_df[industry_col], prefix='sector', drop_first=True)
            X = pd.concat([sector_dummies, day_df[[log_cap_col]]], axis=1).values.astype(float)
            n, p = X.shape

            # 预计算 Ridge 解析解: (X'X + αI)^{-1} X'
            XtX = X.T @ X + self.neutralize_ridge_alpha * np.eye(p)
            try:
                XtX_inv = np.linalg.inv(XtX)
            except np.linalg.LinAlgError:
                XtX_inv = np.linalg.pinv(XtX)
            beta_transform = XtX_inv @ X.T  # shape (p, n)

            # 批量计算所有因子的残差
            Y = day_df[factor_cols].values.astype(float)  # (n, n_factors)
            beta = beta_transform @ Y       # (p, n_factors)
            residuals = Y - X @ beta        # (n, n_factors)

            for i, col in enumerate(factor_cols):
                df.loc[day_df.index, f'{col}_neu'] = residuals[:, i]

        df.drop(columns=[log_cap_col], inplace=True, errors='ignore')
        return df

    def preprocess_pipeline(self, df, factor_cols, industry_col='sector',
                            market_cap_col='market_cap', date_col='trade_date',
                            code_col='ts_code', use_rolling_zscore=True):
        """
        完整预处理流水线：去极值 → 标准化 → 中性化

        Args:
            df: DataFrame
            factor_cols: 原始因子列名
            industry_col: 行业列
            market_cap_col: 市值列
            use_rolling_zscore: True=滚动Z-score（推荐），False=简单截面Z-score

        Returns:
            DataFrame: 包含原始列 + *_z（标准化后）+ *_neu（中性化后）
        """
        print("因子预处理流水线...")
        print(f"  1. MAD 去极值 (n={self.mad_n})")
        df = self.mad_winsorize(df, factor_cols, date_col)

        print(f"  2. 截面标准化")
        if use_rolling_zscore:
            print(f"     使用滚动 Z-score (窗口={self.zscore_window}天)")
            df = self.rolling_zscore(df, factor_cols, date_col, code_col)
            zcols = [f'{c}_z' for c in factor_cols]
        else:
            print(f"     使用简单截面 Z-score")
            df = self.simple_cross_sectional_zscore(df, factor_cols, date_col)
            zcols = [f'{c}_z' for c in factor_cols]

        # 中性化判断：行业列和市值列必须存在，且市值有效值比例 >= 50%
        can_neutralize = (
            industry_col in df.columns and
            market_cap_col in df.columns and
            df[market_cap_col].notna().mean() >= 0.5
        )

        if can_neutralize:
            print(f"  3. 行业/市值中性化")
            print(f"     行业列: {industry_col}, 市值列: {market_cap_col}")
            df = self.neutralize(df, zcols, industry_col, market_cap_col, date_col)
            # neutralize 基于 zcols 创建 *_z_neu 列，统一重命名为 *_neu
            final_cols = []
            for col in factor_cols:
                z_neu_col = f'{col}_z_neu'
                neu_col = f'{col}_neu'
                if z_neu_col in df.columns:
                    df[neu_col] = df[z_neu_col]
                    df.drop(columns=[z_neu_col], inplace=True, errors='ignore')
                else:
                    # 若中性化未生成（极个别情况），回退到 z-score 列
                    df[neu_col] = df.get(f'{col}_z', np.nan)
                final_cols.append(neu_col)
        else:
            reason = "缺少行业或市值列" if not (industry_col in df.columns and market_cap_col in df.columns) else "市值缺失率过高"
            print(f"  3. 跳过中性化（{reason}），使用标准化后因子")
            final_cols = zcols

        print(f"  预处理完成，最终因子列: {final_cols[:5]}... 共 {len(final_cols)} 个")
        return df, final_cols

    def fit(self, train_df, factor_cols, **kwargs):
        """
        在训练集上 fit（记录统计量，用于后续 transform）

        对于本模块的预处理（MAD、Z-score、中性化），
        预测时的 transform 需要知道训练集的统计分布。
        但严格来说，MAD 和截面 Z-score 是每日截面操作，
        预测时可直接对当日截面应用相同逻辑。
        """
        # 当前实现是每日截面操作，无需跨日统计
        # 预留接口以备后续扩展（如全局因子均值）
        pass

    def transform(self, predict_df, factor_cols, **kwargs):
        """
        对预测日数据做 transform（与训练集保持一致）
        """
        # 预测日通常只有 1 天截面数据，直接做截面处理
        return self.preprocess_pipeline(predict_df, factor_cols, **kwargs)


def main():
    """测试因子预处理"""
    from data_loader import DataLoader
    from features import FeatureEngineer

    print("测试因子预处理模块...")
    loader = DataLoader()
    engineer = FeatureEngineer()
    preprocessor = FactorPreprocessor()

    trade_date = loader.get_latest_trade_date()
    stock_list = loader.get_stock_list(trade_date).head(30)

    all_dfs = []
    for _, row in stock_list.iterrows():
        ts_code = row['ts_code']
        df = loader.get_stock_daily(ts_code)
        if df is not None and len(df) >= config.MIN_HISTORY_DAYS:
            df = engineer.add_all_features(df)
            df['ts_code'] = ts_code
            df = df.dropna()
            if not df.empty:
                all_dfs.append(df)

    if not all_dfs:
        print("无有效数据")
        return

    full_df = pd.concat(all_dfs, ignore_index=True)
    feature_cols = engineer.get_feature_columns(full_df)

    # 伪造行业和市值数据用于测试
    full_df['sector'] = np.random.choice(['电子', '医药', '金融', '消费'], size=len(full_df))
    full_df['market_cap'] = np.random.lognormal(10, 1, size=len(full_df))

    print(f"原始因子样例: {full_df[feature_cols[0]].describe()}")

    df_processed, final_cols = preprocessor.preprocess_pipeline(
        full_df, feature_cols,
        industry_col='sector', market_cap_col='market_cap',
        use_rolling_zscore=True
    )

    print(f"\n预处理前后对比（因子 {feature_cols[0]}）:")
    print(f"  原始: mean={full_df[feature_cols[0]].mean():.4f}, std={full_df[feature_cols[0]].std():.4f}")
    print(f"  标准化后: mean={df_processed[final_cols[0]].mean():.4f}, std={df_processed[final_cols[0]].std():.4f}")


if __name__ == "__main__":
    main()
