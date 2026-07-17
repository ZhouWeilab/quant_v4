"""
因子分析模块
功能：Rank IC 检验、IC 衰减、分层回测、因子相关性分析
严格遵循时序隔离原则，所有检验均在截面维度进行
"""

import pandas as pd
import numpy as np
from scipy import stats
import config


class FactorAnalyzer:
    """因子分析器"""

    def __init__(self):
        pass

    @staticmethod
    def calculate_rank_ic(factor_series, return_series):
        """
        计算单期 Rank IC（Spearman 秩相关系数）

        Args:
            factor_series: 某截面日的因子值序列（已去 NaN）
            return_series: 对应的真实收益序列

        Returns:
            float: Rank IC 值，若有效样本不足返回 np.nan
        """
        valid = ~(np.isnan(factor_series) | np.isnan(return_series))
        n = valid.sum()
        if n < 10:
            return np.nan
        ic, _ = stats.spearmanr(factor_series[valid], return_series[valid])
        return ic

    def calculate_ic_series(self, df, factor_col, return_col='target_return',
                            date_col='trade_date', code_col='ts_code'):
        """
        计算某因子在全时段的 Rank IC 时间序列

        Args:
            df: DataFrame，必须含 trade_date, ts_code, factor_col, return_col
            factor_col: 因子列名
            return_col: 收益列名（默认 target_return）
            date_col: 日期列名
            code_col: 股票代码列名

        Returns:
            Series: index=trade_date, values=Rank IC
        """
        ic_list = []
        dates = sorted(df[date_col].unique())

        for date in dates:
            day_df = df[df[date_col] == date]
            if len(day_df) < 10:
                continue
            ic = self.calculate_rank_ic(
                day_df[factor_col].values,
                day_df[return_col].values
            )
            if not np.isnan(ic):
                ic_list.append({'trade_date': date, 'ic': ic})

        ic_df = pd.DataFrame(ic_list)
        if ic_df.empty:
            return pd.Series(dtype=float)
        ic_series = ic_df.set_index('trade_date')['ic']
        return ic_series

    def ic_statistics(self, ic_series):
        """
        从 IC 时间序列中提取统计量

        Args:
            ic_series: Series，index=date，values=IC

        Returns:
            dict: ic_mean, ic_std, ic_ir, ic_t, ic_pos_ratio, ic_significant_ratio
        """
        if len(ic_series) < 10:
            return {
                'ic_mean': np.nan, 'ic_std': np.nan, 'ic_ir': np.nan,
                'ic_t': np.nan, 'ic_pos_ratio': np.nan,
                'ic_significant_ratio': np.nan, 'n_periods': len(ic_series)
            }

        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        ic_ir = ic_mean / ic_std if ic_std > 0 else np.nan
        ic_t = ic_mean / (ic_std / np.sqrt(len(ic_series))) if ic_std > 0 else np.nan
        ic_pos_ratio = (ic_series > 0).mean()
        ic_significant_ratio = (np.abs(ic_series) > 0.02).mean()

        return {
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ic_ir': ic_ir,
            'ic_t': ic_t,
            'ic_pos_ratio': ic_pos_ratio,
            'ic_significant_ratio': ic_significant_ratio,
            'n_periods': len(ic_series)
        }

    def ic_decay(self, df, factor_col, return_col='target_return',
                 date_col='trade_date', code_col='ts_code', max_lag=5):
        """
        计算 IC 衰减曲线：当前因子值对未来 N 期收益的预测能力

        Args:
            df: 含因子和收益的 DataFrame
            factor_col: 因子列名
            return_col: 收益列名
            max_lag: 最大滞后期数

        Returns:
            DataFrame: columns=[lag, ic_mean, ic_std], lag=0..max_lag
        """
        df = df.sort_values([code_col, date_col]).copy()
        results = []

        for lag in range(0, max_lag + 1):
            # 构建滞后收益：当前因子值 对应 未来 lag 期的收益
            df[f'return_lag{lag}'] = df.groupby(code_col)[return_col].shift(-lag)
            ic_series = self.calculate_ic_series(
                df, factor_col, f'return_lag{lag}', date_col, code_col
            )
            if not ic_series.empty:
                results.append({
                    'lag': lag,
                    'ic_mean': ic_series.mean(),
                    'ic_std': ic_series.std(),
                    'n': len(ic_series)
                })
            df.drop(columns=[f'return_lag{lag}'], inplace=True, errors='ignore')

        return pd.DataFrame(results)

    def quantile_backtest(self, df, factor_col, return_col='target_return',
                          date_col='trade_date', code_col='ts_code',
                          n_quantiles=5, long_short=False):
        """
        分层回测：按因子值每日分 N 层，检验单调性

        Args:
            df: DataFrame
            factor_col: 因子列名
            return_col: 收益列名
            n_quantiles: 分层数（默认 5）
            long_short: 是否计算多空对冲收益（做多 Top，做空 Bottom）

        Returns:
            dict: 每层年化收益、多空对冲收益、单调性评分
        """
        df = df.copy()
        dates = sorted(df[date_col].unique())

        layer_returns = {i: [] for i in range(1, n_quantiles + 1)}
        long_short_returns = []

        for date in dates:
            day_df = df[df[date_col] == date].copy()
            if len(day_df) < n_quantiles * 5:
                continue

            # 去掉 NaN
            day_df = day_df.dropna(subset=[factor_col, return_col])
            if len(day_df) < n_quantiles * 5:
                continue

            # 按因子值分层
            day_df['layer'] = pd.qcut(
                day_df[factor_col],
                q=n_quantiles,
                labels=range(1, n_quantiles + 1),
                duplicates='drop'
            )

            # 如果分层失败（如所有值相同），跳过
            if day_df['layer'].isna().all():
                continue

            for i in range(1, n_quantiles + 1):
                layer_df = day_df[day_df['layer'] == i]
                if not layer_df.empty:
                    avg_ret = layer_df[return_col].mean()
                    layer_returns[i].append(avg_ret)

            if long_short:
                top_df = day_df[day_df['layer'] == n_quantiles]
                bottom_df = day_df[day_df['layer'] == 1]
                if not top_df.empty and not bottom_df.empty:
                    ls_ret = top_df[return_col].mean() - bottom_df[return_col].mean()
                    long_short_returns.append(ls_ret)

        # 汇总统计
        summary = {}
        annual_factor = 250  # 假设 250 个交易日/年

        for i in range(1, n_quantiles + 1):
            rets = layer_returns[i]
            if rets:
                summary[f'layer_{i}_mean'] = np.mean(rets)
                summary[f'layer_{i}_annual'] = np.mean(rets) * annual_factor
                summary[f'layer_{i}_sharpe'] = (
                    np.mean(rets) / np.std(rets) * np.sqrt(annual_factor)
                    if np.std(rets) > 0 else np.nan
                )

        # 单调性检验：层号与平均收益的相关性
        layer_means = [summary.get(f'layer_{i}_mean', np.nan) for i in range(1, n_quantiles + 1)]
        if all(~np.isnan(layer_means)):
            mono_corr, mono_p = stats.spearmanr(range(1, n_quantiles + 1), layer_means)
            summary['monotonicity_corr'] = mono_corr
            summary['monotonicity_pvalue'] = mono_p
        else:
            summary['monotonicity_corr'] = np.nan
            summary['monotonicity_pvalue'] = np.nan

        if long_short and long_short_returns:
            summary['long_short_mean'] = np.mean(long_short_returns)
            summary['long_short_annual'] = np.mean(long_short_returns) * annual_factor
            summary['long_short_sharpe'] = (
                np.mean(long_short_returns) / np.std(long_short_returns) * np.sqrt(annual_factor)
                if np.std(long_short_returns) > 0 else np.nan
            )

        summary['n_periods'] = len(dates)
        return summary, layer_returns

    def correlation_matrix(self, df, factor_cols, method='spearman'):
        """
        计算因子相关性矩阵

        Args:
            df: DataFrame
            factor_cols: 因子列名列表
            method: 'spearman' 或 'pearson'

        Returns:
            DataFrame: 相关性矩阵
        """
        corr = df[factor_cols].corr(method=method)
        return corr

    def find_high_correlation_pairs(self, corr_matrix, threshold=0.8):
        """
        找出相关性超过阈值的因子对

        Args:
            corr_matrix: 相关性矩阵 DataFrame
            threshold: 阈值（绝对值）

        Returns:
            DataFrame: columns=[factor_a, factor_b, correlation]
        """
        pairs = []
        cols = corr_matrix.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    pairs.append({
                        'factor_a': cols[i],
                        'factor_b': cols[j],
                        'correlation': corr_val
                    })
        return pd.DataFrame(pairs)

    def full_analysis(self, df, factor_cols, return_col='target_return',
                      date_col='trade_date', code_col='ts_code'):
        """
        对全部因子执行完整检验：IC + 分层 + 相关性

        Args:
            df: DataFrame
            factor_cols: 因子列名列表
            return_col: 收益列名

        Returns:
            dict: 含 ic_report, quantile_report, corr_matrix, high_corr_pairs
        """
        print("=" * 60)
        print("开始因子全量检验...")
        print("=" * 60)

        # 1. IC 检验
        print("\n[1/3] 计算 Rank IC...")
        ic_report = []
        for col in factor_cols:
            ic_series = self.calculate_ic_series(df, col, return_col, date_col, code_col)
            stats_dict = self.ic_statistics(ic_series)
            stats_dict['factor'] = col
            ic_report.append(stats_dict)
        ic_report_df = pd.DataFrame(ic_report)
        ic_report_df = ic_report_df.sort_values('ic_mean', key=abs, ascending=False)

        print(f"  检验因子数: {len(factor_cols)}")
        print(f"  IC 均值绝对值 > 0.02 的因子: {(ic_report_df['ic_mean'].abs() > 0.02).sum()} 个")

        # 2. 分层回测（仅对 IC 最高的前 10 个因子做，避免太慢）
        print("\n[2/3] 分层回测（Top 10 因子）...")
        top_factors = ic_report_df.head(10)['factor'].tolist()
        quantile_report = {}
        for col in top_factors:
            summary, _ = self.quantile_backtest(
                df, col, return_col, date_col, code_col,
                n_quantiles=5, long_short=True
            )
            quantile_report[col] = summary

        # 3. 相关性矩阵
        print("\n[3/3] 计算因子相关性矩阵...")
        corr_matrix = self.correlation_matrix(df, factor_cols)
        high_corr_pairs = self.find_high_correlation_pairs(corr_matrix, threshold=0.8)

        if not high_corr_pairs.empty:
            print(f"  警告: 发现 {len(high_corr_pairs)} 对高相关性因子（|r| >= 0.8）")
            print(high_corr_pairs.to_string(index=False))
        else:
            print("  未发现 |r| >= 0.8 的高相关性因子对")

        print("\n" + "=" * 60)
        print("因子检验完成")
        print("=" * 60)

        return {
            'ic_report': ic_report_df,
            'quantile_report': quantile_report,
            'corr_matrix': corr_matrix,
            'high_corr_pairs': high_corr_pairs
        }

    def print_ic_report(self, ic_report_df, top_n=20):
        """打印 IC 报告"""
        print("\n" + "-" * 80)
        print(f"Rank IC 检验报告（Top {top_n}）")
        print("-" * 80)
        print(f"{'因子':<25} {'IC均值':>10} {'IC标准差':>10} {'IC_IR':>10} {'IC_t':>10} {'正显著比':>10} {'样本期':>8}")
        print("-" * 80)
        for _, row in ic_report_df.head(top_n).iterrows():
            print(f"{row['factor']:<25} {row['ic_mean']:>10.4f} {row['ic_std']:>10.4f} "
                  f"{row['ic_ir']:>10.4f} {row['ic_t']:>10.4f} {row['ic_significant_ratio']:>10.2%} {int(row['n_periods']):>8d}")
        print("-" * 80)


def main():
    """测试因子分析功能"""
    from data_loader import DataLoader
    from features import FeatureEngineer
    from dataset import DatasetBuilder

    print("测试因子分析模块...")
    loader = DataLoader()
    engineer = FeatureEngineer()
    builder = DatasetBuilder()
    analyzer = FactorAnalyzer()

    trade_date = loader.get_latest_trade_date()
    stock_list = loader.get_stock_list(trade_date).head(20)

    stock_data = {}
    for _, row in stock_list.iterrows():
        ts_code = row['ts_code']
        df = loader.get_stock_daily(ts_code)
        if df is not None and len(df) >= config.MIN_HISTORY_DAYS:
            stock_data[ts_code] = df

    if not stock_data:
        print("无有效数据")
        return

    # 构建带特征的完整数据
    all_dfs = []
    for ts_code, df in stock_data.items():
        df = engineer.add_all_features(df)
        df = builder.create_target(df)
        df['ts_code'] = ts_code
        df = df.dropna()
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        print("无有效特征数据")
        return

    full_df = pd.concat(all_dfs, ignore_index=True)
    feature_cols = engineer.get_feature_columns(full_df)

    print(f"测试数据: {len(full_df)} 条，因子数: {len(feature_cols)}")

    # 全量检验
    results = analyzer.full_analysis(full_df, feature_cols)
    analyzer.print_ic_report(results['ic_report'])


if __name__ == "__main__":
    main()
