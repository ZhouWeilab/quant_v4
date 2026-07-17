"""
鏁版嵁闆嗘瀯寤烘ā鍧?鍔熻兘锛氭瀯寤鸿缁冮泦銆侀獙璇侀泦銆佹祴璇曢泦锛岃绠楁爣绛撅紙娆℃棩娑ㄨ穼骞咃級锛屾暟鎹爣鍑嗗寲
鏀寔鏃跺簭绐楀彛鏁版嵁鏋勫缓锛岄€傞厤 CNN+LSTM+Attention 妯″瀷
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import pickle
import config
from features import FeatureEngineer


class DatasetBuilder:
    """鏁版嵁闆嗘瀯寤哄櫒锛堟敮鎸佹椂搴忕獥鍙ｏ級"""

    def __init__(self, use_robust_scaler=True):
        """
        鍒濆鍖?
        Args:
            use_robust_scaler: 鏄惁浣跨敤 RobustScaler锛堝寮傚父鍊兼洿椴佹锛?        """
        self.feature_engineer = FeatureEngineer()
        self.scaler = None
        self.feature_cols = None
        self.use_robust_scaler = use_robust_scaler

    @staticmethod
    def finalize_feature_frame(df, require_target=False):
        """统一训练和预测的缺失值处理，避免两条流水线发生漂移。"""
        df = df.replace([np.inf, -np.inf], np.nan).copy()
        # 仅对每日连续口径的基本面字段使用历史值向前填充；龙虎榜、资金流等
        # 稀疏事件字段不能沿用到后续交易日，否则会制造虚假的持续信号。
        continuous_cols = [
            'turnover_rate', 'turnover_rate_f', 'volume_ratio',
            'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio', 'dv_ttm',
            'total_share', 'float_share', 'free_share', 'total_mv', 'circ_mv',
        ]
        continuous_cols = [col for col in continuous_cols if col in df.columns]
        if continuous_cols:
            df[continuous_cols] = df[continuous_cols].ffill()

        required = ['return_20d', 'ma_60', f'rsi_{config.RSI_PERIOD}']
        if require_target:
            required.append('target')
        required = [col for col in required if col in df.columns]
        df = df.dropna(subset=required).copy()
        numeric_cols = df.select_dtypes(include=[np.number, 'bool']).columns
        df[numeric_cols] = df[numeric_cols].fillna(0.0)
        return df

    def create_target(self, df, horizon=None):
        """创建与实盘一致的收益标签：T+1 开盘买入，T+6 收盘卖出。"""
        entry_shift = getattr(config, 'LABEL_ENTRY_SHIFT', 1)
        exit_shift = getattr(config, 'LABEL_EXIT_SHIFT', entry_shift + 1)

        df['target_return'] = (
            df['close'].shift(-exit_shift) / df['open'].shift(-entry_shift) - 1
        )

        # 鏍规嵁閰嶇疆閫夋嫨鐩爣绫诲瀷
        if getattr(config, 'TARGET_TYPE', 'classification') == 'regression':
            # 鍥炲綊鐩爣锛氱洿鎺ラ娴嬭繛缁敹鐩婄巼
            df['target'] = df['target_return']
        else:
            # 浜屽垎绫荤洰鏍囷細鎸佹湁鏈熶笂娑?1锛屼笅璺屾垨鎸佸钩=0
            df['target'] = (df['target_return'] > 0).astype(int)

        # 鍒犻櫎鏈€鍚?exit_shift 琛岋紙娌℃湁鏈潵鏁版嵁锛?        df = df[:-exit_shift].copy()

        return df

    def prepare_single_stock(self, df, ts_code):
        """
        鍑嗗鍗曞彧鑲＄エ鐨勬暟鎹?
        Args:
            df: 鍘熷鏃ョ嚎鏁版嵁
            ts_code: 鑲＄エ浠ｇ爜

        Returns:
            DataFrame: 鍖呭惈鐗瑰緛鍜岀洰鏍囩殑鏁版嵁
        """
        # 娣诲姞鑲＄エ浠ｇ爜
        df['ts_code'] = ts_code

        # 娣诲姞鐗瑰緛
        df = self.feature_engineer.add_all_features(df)

        # 鍒涘缓鐩爣
        df = self.create_target(df)

        df = self.finalize_feature_frame(df, require_target=True)

        return df

    def build_dataset(self, stock_data_dict):
        """
        鏋勫缓瀹屾暣鏁版嵁闆?
        Args:
            stock_data_dict: {ts_code: DataFrame} 鏍煎紡鐨勮偂绁ㄦ暟鎹瓧鍏?
        Returns:
            DataFrame: 鍚堝苟鍚庣殑瀹屾暣鏁版嵁闆?        """
        print("鏋勫缓鏁版嵁闆?..")

        all_dfs = []

        for ts_code, df in stock_data_dict.items():
            try:
                prepared_df = self.prepare_single_stock(df, ts_code)

                if len(prepared_df) > config.SEQUENCE_LENGTH:
                    all_dfs.append(prepared_df)

            except Exception as e:
                print(f"澶勭悊 {ts_code} 澶辫触: {e}")
                continue

        if not all_dfs:
            raise ValueError("没有有效的数据")

        dataset = pd.concat(all_dfs, ignore_index=True)

        # 鎸夎偂绁ㄤ唬鐮佸拰鏃ユ湡鎺掑簭锛堥噸瑕侊細鏃跺簭鏁版嵁蹇呴』鎸夎偂绁ㄥ垎缁勶級
        dataset = dataset.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
        dataset = self.add_cross_section_features(dataset)
        dataset = self.apply_universe_eligibility(dataset)
        dataset = self.apply_target_normalization(dataset)

        print(f"数据集构建完成: {len(dataset)} 条样本")

        return dataset

    @staticmethod
    def add_cross_section_features(dataset):
        """加入仅使用当日可见信息的市场、行业和横截面排序特征。"""
        dataset = dataset.copy()
        date_group = dataset.groupby('trade_date', sort=False)
        ret_1d = pd.to_numeric(dataset['return_1d'], errors='coerce')

        dataset['market_return_1d'] = date_group['return_1d'].transform('median')
        dataset['market_return_5d'] = date_group['return_5d'].transform('median')
        dataset['market_volatility_1d'] = date_group['return_1d'].transform('std')
        dataset['market_breadth'] = (
            ret_1d.gt(0).groupby(dataset['trade_date']).transform('mean')
        )
        dataset['relative_return_1d'] = dataset['return_1d'] - dataset['market_return_1d']
        dataset['relative_return_5d'] = dataset['return_5d'] - dataset['market_return_5d']

        if 'sector' in dataset.columns:
            sector_group = dataset.groupby(['trade_date', 'sector'], sort=False)
            dataset['sector_return_1d'] = sector_group['return_1d'].transform('median')
            dataset['sector_return_5d'] = sector_group['return_5d'].transform('median')
            dataset['sector_breadth'] = (
                ret_1d.gt(0)
                .groupby([dataset['trade_date'], dataset['sector']])
                .transform('mean')
            )
            dataset['sector_relative_return_1d'] = (
                dataset['return_1d'] - dataset['sector_return_1d']
            )
            dataset['sector_relative_return_5d'] = (
                dataset['return_5d'] - dataset['sector_return_5d']
            )

        # 只对核心因子生成当日截面排名，兼顾跨期可比性和内存占用。
        rank_candidates = [
            'return_1d', 'return_3d', 'return_5d', 'return_10d', 'return_20d',
            'volatility_5', 'volatility_10', 'volatility_20',
            'amplitude', 'close_position', 'price_volume_corr',
            'amount_change', 'amount_log_change', 'amount_zscore_20',
            'vol_change', 'vol_ratio_1_5', 'vol_ratio_5_20',
            'turnover_rate', 'turnover_ratio', 'turnover_zscore',
            'rsi_14', 'macd_norm', 'boll_width', 'boll_position',
            'atr_14_norm', 'momentum_5_norm', 'momentum_10_norm',
            'momentum_20_norm', 'main_net_ratio', 'net_mf_ratio',
            'elg_net_ratio', 'sm_net_ratio', 'top_net_ratio',
            'top_amount_ratio', 'breakout_20', 'breakout_60',
            'drawdown_10', 'drawdown_20', 'open_gap', 'ret_std_5',
            'ret_skew_10', 'market_cap'
        ]
        for col in rank_candidates:
            if col in dataset.columns:
                dataset[f'{col}_cs_rank'] = (
                    dataset.groupby('trade_date')[col]
                    .rank(pct=True, method='average')
                    .astype(np.float32)
                )
        return dataset

    @staticmethod
    def apply_universe_eligibility(dataset):
        """按每个交易日当时可获得的信息标记可交易训练样本。"""
        dataset = dataset.copy()
        eligible = pd.Series(True, index=dataset.index)

        price_col = 'raw_close' if 'raw_close' in dataset.columns else 'close'
        price = pd.to_numeric(dataset[price_col], errors='coerce')
        eligible &= price.between(config.MIN_PRICE, config.MAX_PRICE)

        if config.EXCLUDE_ST and 'is_st' in dataset.columns:
            eligible &= ~dataset['is_st'].fillna(False).astype(bool)

        if config.EXCLUDE_NEW_STOCK_DAYS > 0 and 'listing_age_days' in dataset.columns:
            min_calendar_days = int(np.ceil(config.EXCLUDE_NEW_STOCK_DAYS * 365 / 250))
            age = pd.to_numeric(dataset['listing_age_days'], errors='coerce')
            eligible &= age.isna() | age.ge(min_calendar_days)

        pool_limit = int(getattr(config, 'STOCK_POOL_LIMIT', 0) or 0)
        if (
            getattr(config, 'DYNAMIC_STOCK_POOL', False)
            and pool_limit > 0
            and 'market_cap' in dataset.columns
        ):
            market_cap = pd.to_numeric(dataset['market_cap'], errors='coerce')
            cap_rank = market_cap.groupby(dataset['trade_date']).rank(
                ascending=False, method='first'
            )
            eligible &= market_cap.notna() & cap_rank.le(pool_limit)

        dataset['universe_eligible'] = eligible
        print(
            f"动态股票池有效样本: {int(eligible.sum())}/{len(dataset)} "
            f"({eligible.mean():.2%})"
        )
        return dataset

    def apply_target_normalization(self, dataset):
        """
        灏嗗洖褰掓爣绛捐浆鎹负鏇撮€傚悎閫夎偂鎺掑簭鐨勬í鎴潰鐩爣銆?
        鍘熷 target_return 浠嶇劧淇濈暀锛岀敤浜庨獙璇佹椂璁＄畻鐪熷疄鏀剁泭銆丷ank IC 鍜屽垎灞傛敹鐩娿€?        """
        if getattr(config, 'TARGET_TYPE', 'classification') != 'regression':
            return dataset

        mode = getattr(config, 'TARGET_NORMALIZATION', 'raw')
        if mode == 'raw':
            dataset['target'] = dataset['target_return']
            return dataset
        if mode not in (
            'cs_rank', 'cs_zscore', 'excess_cs_zscore',
            'partial_excess_cs_zscore'
        ):
            raise ValueError(f"Unsupported TARGET_NORMALIZATION: {mode}")

        low_q = getattr(config, 'TARGET_WINSOR_Q_LOW', 0.01)
        high_q = getattr(config, 'TARGET_WINSOR_Q_HIGH', 0.99)
        clip_value = getattr(config, 'TARGET_ZSCORE_CLIP', 3.0)

        def _normalize_one_day(s, center=True):
            s = s.astype(float)
            if len(s) < 5:
                return pd.Series(np.nan, index=s.index)
            lower = s.quantile(low_q)
            upper = s.quantile(high_q)
            clipped = s.clip(lower=lower, upper=upper)
            std = clipped.std(ddof=0)
            if not np.isfinite(std) or std <= 1e-12:
                return pd.Series(0.0, index=s.index)
            center_value = clipped.mean() if center else 0.0
            z = (clipped - center_value) / std
            return z.clip(-clip_value, clip_value)

        def _rank_one_day(s):
            s = s.astype(float)
            if s.notna().sum() < 5:
                return pd.Series(np.nan, index=s.index)
            lower = s.quantile(low_q)
            upper = s.quantile(high_q)
            clipped = s.clip(lower=lower, upper=upper)
            rank = clipped.rank(method='average')
            count = int(clipped.notna().sum())
            return (rank - 1.0) / max(count - 1, 1) * 2.0 - 1.0

        dataset = dataset.copy()
        dataset['target_raw'] = dataset['target_return']
        target_col = 'target_return'
        eligible = dataset.get(
            'universe_eligible',
            pd.Series(True, index=dataset.index)
        ).fillna(False)

        if mode in ('excess_cs_zscore', 'partial_excess_cs_zscore'):
            method = getattr(config, 'TARGET_EXCESS_METHOD', 'median')
            if method == 'mean':
                benchmark_by_date = (
                    dataset.loc[eligible].groupby('trade_date')['target_return'].mean()
                )
            elif method == 'median':
                benchmark_by_date = (
                    dataset.loc[eligible].groupby('trade_date')['target_return'].median()
                )
            else:
                raise ValueError(f"Unsupported TARGET_EXCESS_METHOD: {method}")
            benchmark = dataset['trade_date'].map(benchmark_by_date)
            beta = 1.0 if mode == 'excess_cs_zscore' else getattr(config, 'TARGET_EXCESS_BETA', 0.5)
            dataset['target_excess_return'] = dataset['target_return'] - beta * benchmark
            target_col = 'target_excess_return'

        dataset['target'] = np.nan
        grouped_target = dataset.loc[eligible].groupby(
            'trade_date', group_keys=False
        )[target_col]
        if mode == 'cs_rank':
            normalized = grouped_target.apply(_rank_one_day)
        else:
            # 超额收益模式不能再次按日去均值，否则减去的基准会被抵消。
            centered = mode == 'cs_zscore'
            normalized = grouped_target.apply(
                lambda values: _normalize_one_day(values, center=centered)
            )
        dataset.loc[normalized.index, 'target'] = normalized

        print(
            "妯埅闈㈡爣绛炬爣鍑嗗寲瀹屾垚: "
            f"mean={dataset['target'].mean():.4f}, "
            f"std={dataset['target'].std():.4f}, "
            f"mode={mode}"
        )
        return dataset

    def split_dataset(self, dataset, val_split=None):
        """
        鍒掑垎璁粌闆嗗拰楠岃瘉闆嗭紙鎸夋椂闂撮『搴忥級

        Args:
            dataset: 瀹屾暣鏁版嵁闆?            val_split: 楠岃瘉闆嗘瘮渚?
        Returns:
            tuple: (train_df, val_df)
        """
        if val_split is None:
            val_split = config.VALIDATION_SPLIT

        unique_dates = sorted(dataset['trade_date'].unique())
        split_idx = int(len(unique_dates) * (1 - val_split))
        split_date = unique_dates[split_idx]
        purge_days = int(getattr(config, 'PURGE_DAYS', 0) or 0)
        train_end_idx = max(0, split_idx - purge_days)
        train_end_date = unique_dates[train_end_idx]

        train_df = dataset[dataset['trade_date'] < train_end_date].copy()
        val_df = dataset[dataset['trade_date'] >= split_date].copy()

        print(f"训练集: {len(train_df)} 条样本")
        print(f"验证集: {len(val_df)} 条样本")
        print(f"训练/验证隔离: {purge_days} 个交易日")

        return train_df, val_df

    def split_train_val_test(self, dataset, val_split=None, test_split=None):
        """按时间划分训练、验证和最终测试集，并在边界执行purge。"""
        val_split = config.VALIDATION_SPLIT if val_split is None else val_split
        test_split = getattr(config, 'TEST_SPLIT', 0.1) if test_split is None else test_split
        if val_split <= 0 or test_split <= 0 or val_split + test_split >= 0.5:
            raise ValueError("VALIDATION_SPLIT/TEST_SPLIT 配置不合理")

        dates = sorted(dataset['trade_date'].unique())
        val_start_idx = int(len(dates) * (1 - val_split - test_split))
        test_start_idx = int(len(dates) * (1 - test_split))
        purge_days = int(getattr(config, 'PURGE_DAYS', 0) or 0)

        train_end_idx = max(0, val_start_idx - purge_days)
        val_end_idx = max(val_start_idx, test_start_idx - purge_days)
        train_end = dates[train_end_idx]
        val_start = dates[val_start_idx]
        val_end = dates[val_end_idx]
        test_start = dates[test_start_idx]

        train_df = dataset[dataset['trade_date'] < train_end].copy()
        val_df = dataset[
            (dataset['trade_date'] >= val_start) & (dataset['trade_date'] < val_end)
        ].copy()
        test_df = dataset[dataset['trade_date'] >= test_start].copy()

        print(
            f"训练日期: {train_df['trade_date'].min()}~{train_df['trade_date'].max()}, "
            f"样本={len(train_df)}"
        )
        print(
            f"验证日期: {val_df['trade_date'].min()}~{val_df['trade_date'].max()}, "
            f"样本={len(val_df)}"
        )
        print(
            f"测试日期: {test_df['trade_date'].min()}~{test_df['trade_date'].max()}, "
            f"样本={len(test_df)}"
        )
        print(f"切分边界隔离: {purge_days} 个交易日")
        return train_df, val_df, test_df

    def get_feature_columns(self, dataset):
        """获取特征列名。"""
        return self.feature_engineer.get_feature_columns(dataset)

    def select_feature_columns(self, train_df, feature_cols):
        """仅用训练集选择跨时期稳定、覆盖充分且不过度相关的因子。"""
        if not getattr(config, 'FEATURE_SELECTION_ENABLED', False):
            return feature_cols

        max_features = int(getattr(config, 'FEATURE_SELECTION_MAX_FEATURES', 0) or 0)
        min_abs_ic = float(getattr(config, 'FEATURE_SELECTION_MIN_ABS_IC', 0.0) or 0.0)
        sample_size = int(getattr(config, 'FEATURE_SELECTION_SAMPLE_SIZE', 0) or 0)
        min_coverage = float(
            getattr(config, 'FEATURE_SELECTION_MIN_COVERAGE', 0.0) or 0.0
        )
        min_nonzero_ratio = float(
            getattr(config, 'FEATURE_SELECTION_MIN_NONZERO_RATIO', 0.0) or 0.0
        )
        subperiods = max(
            1, int(getattr(config, 'FEATURE_SELECTION_SUBPERIODS', 1) or 1)
        )
        min_sign_stability = float(
            getattr(config, 'FEATURE_SELECTION_MIN_SIGN_STABILITY', 0.0) or 0.0
        )
        if max_features <= 0:
            return feature_cols

        eligible_train = train_df.dropna(subset=['target'])
        if eligible_train.empty:
            raise ValueError("训练集没有可用于因子筛选的目标样本")
        sample_df = eligible_train
        if sample_size > 0 and len(eligible_train) > sample_size:
            # 抽取完整交易日，避免随机抽行破坏横截面排序结构。
            date_counts = eligible_train.groupby('trade_date').size()
            rng = np.random.default_rng(getattr(config, 'RANDOM_SEED', 42))
            sampled_dates = []
            sampled_rows = 0
            for trade_date in rng.permutation(date_counts.index.to_numpy()):
                count = int(date_counts.loc[trade_date])
                if sampled_dates and sampled_rows + count > sample_size:
                    continue
                sampled_dates.append(trade_date)
                sampled_rows += count
                if sampled_rows >= sample_size:
                    break
            sample_df = eligible_train[
                eligible_train['trade_date'].isin(sampled_dates)
            ].copy()

        date_groups = sample_df.groupby('trade_date', sort=False)
        target_rank = date_groups['target'].rank(pct=True, method='average')
        min_periods = int(getattr(config, 'IC_MIN_PERIODS', 10) or 10)
        sorted_dates = np.sort(sample_df['trade_date'].unique())
        period_by_date = {}
        for period_no, period_dates in enumerate(np.array_split(sorted_dates, subperiods)):
            for trade_date in period_dates:
                period_by_date[trade_date] = period_no

        scores = []
        for col in feature_cols:
            x = pd.to_numeric(sample_df[col], errors='coerce').replace(
                [np.inf, -np.inf], np.nan
            )
            coverage = float(x.notna().mean())
            if coverage < min_coverage:
                continue
            if x.nunique(dropna=True) < 3:
                continue
            nonzero_ratio = float(x.fillna(0.0).ne(0.0).mean())
            if nonzero_ratio < min_nonzero_ratio:
                continue
            feature_rank = date_groups[col].rank(pct=True, method='average')
            feature_std = feature_rank.groupby(
                sample_df['trade_date']
            ).transform('std')
            target_std = target_rank.groupby(
                sample_df['trade_date']
            ).transform('std')
            valid = (
                feature_rank.notna()
                & target_rank.notna()
                & feature_std.gt(1e-12)
                & target_std.gt(1e-12)
            )
            daily_ic = (
                feature_rank[valid].groupby(sample_df.loc[valid, 'trade_date'])
                .corr(target_rank[valid])
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if len(daily_ic) < min_periods:
                continue
            mean_ic = float(daily_ic.mean())
            std_ic = float(daily_ic.std(ddof=1))
            if abs(mean_ic) < min_abs_ic:
                continue

            period_ic = daily_ic.groupby(
                daily_ic.index.to_series().map(period_by_date).values
            ).mean()
            nonzero_period_ic = period_ic[np.abs(period_ic) > 1e-12]
            if len(nonzero_period_ic):
                sign_stability = float(
                    (np.sign(nonzero_period_ic) == np.sign(mean_ic)).mean()
                )
            else:
                sign_stability = 0.0
            if sign_stability < min_sign_stability:
                continue

            stability_score = (
                abs(mean_ic) * np.sqrt(len(daily_ic)) / (std_ic + 1e-12)
            )
            stability_score *= np.sqrt(max(coverage, 0.0))
            stability_score *= 0.5 + 0.5 * sign_stability
            scores.append((
                col, stability_score, mean_ic, std_ic,
                coverage, nonzero_ratio, sign_stability
            ))

        if not scores:
            raise ValueError("没有因子通过覆盖率、IC和跨时期同号筛选")

        scores.sort(key=lambda item: item[1], reverse=True)
        candidate_cols = [
            item[0] for item in scores[:max(max_features * 3, max_features)]
        ]
        corr_threshold = float(
            getattr(config, 'FEATURE_SELECTION_CORR_THRESHOLD', 0.0) or 0.0
        )
        selected = []
        if 0.0 < corr_threshold < 1.0 and len(candidate_cols) > 1:
            rank_matrix = (
                sample_df[candidate_cols]
                .groupby(sample_df['trade_date'])
                .rank(pct=True, method='average')
            )
            corr_sample_size = int(
                getattr(config, 'FEATURE_SELECTION_CORR_SAMPLE_SIZE', 0) or 0
            )
            if corr_sample_size > 0 and len(rank_matrix) > corr_sample_size:
                rng = np.random.default_rng(getattr(config, 'RANDOM_SEED', 42))
                indices = np.sort(
                    rng.choice(len(rank_matrix), corr_sample_size, replace=False)
                )
                rank_matrix = rank_matrix.iloc[indices]
            corr = rank_matrix.corr().abs()
            for col in candidate_cols:
                if not selected or corr.loc[col, selected].max() < corr_threshold:
                    selected.append(col)
                if len(selected) >= max_features:
                    break
        else:
            selected = candidate_cols[:max_features]

        if not selected:
            selected = [scores[0][0]]
        print(
            f"因子筛选完成: {len(feature_cols)} -> {len(selected)} "
            f"(sample={len(sample_df)}, dates={sample_df['trade_date'].nunique()}, "
            f"min_abs_daily_ic={min_abs_ic}, corr<{corr_threshold})"
        )
        print(f"Top因子: {[item[0] for item in scores[:10]]}")
        return selected

    def create_sequences(self, df, feature_cols):
        """创建时序窗口数据。"""
        X_list = []
        y_list = []
        meta_list = []

        grouped = df.groupby('ts_code')
        for ts_code, group_df in grouped:
            group_df = group_df.sort_values('trade_date').reset_index(drop=True)
            features = group_df[feature_cols].values
            targets = group_df['target'].values

            for i in range(len(group_df) - config.SEQUENCE_LENGTH + 1):
                X_seq = features[i:i + config.SEQUENCE_LENGTH]
                y_target = targets[i + config.SEQUENCE_LENGTH - 1]
                if not np.isfinite(y_target) or not np.isfinite(X_seq).all():
                    continue

                meta_cols = ['ts_code', 'trade_date', 'close']
                for col in [
                    'raw_close', 'target', 'target_return', 'target_raw',
                    'target_excess_return', 'sector', 'market_cap',
                    'amount', 'pct_chg', 'universe_eligible'
                ]:
                    if col in group_df.columns:
                        meta_cols.append(col)
                meta_row = group_df.iloc[i + config.SEQUENCE_LENGTH - 1][meta_cols]

                X_list.append(X_seq)
                y_list.append(y_target)
                meta_list.append(meta_row)

        X_sequences = np.array(X_list, dtype=np.float32)
        y_targets = np.array(y_list, dtype=np.float32)
        metadata = pd.DataFrame(meta_list).reset_index(drop=True)
        return X_sequences, y_targets, metadata

    def fit_scaler(self, train_df, feature_cols=None):
        """
        璁粌鏍囧噯鍖栧櫒

        Args:
            train_df: 璁粌鏁版嵁
            feature_cols: 鐗瑰緛鍒楀悕

        Returns:
            Scaler: 璁粌濂界殑鏍囧噯鍖栧櫒
        """
        if feature_cols is None:
            feature_cols = self.get_feature_columns(train_df)

        self.feature_cols = feature_cols

        if self.use_robust_scaler:
            # RobustScaler 瀵瑰紓甯稿€兼洿椴佹锛堜娇鐢ㄤ腑浣嶆暟鍜屽洓鍒嗕綅鏁帮級
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()

        # 璁粌
        scaler_df = train_df.dropna(subset=['target'])
        X_train = scaler_df[feature_cols]
        self.scaler.fit(X_train)

        print(f"鏍囧噯鍖栧櫒璁粌瀹屾垚锛岀壒寰佹暟閲? {len(feature_cols)}")
        print(f"浣跨敤鏍囧噯鍖栧櫒: {type(self.scaler).__name__}")

        return self.scaler

    def transform_features(self, df, feature_cols=None):
        """
        鏍囧噯鍖栫壒寰侊紙涓嶆敼鍙?DataFrame 缁撴瀯锛?
        Args:
            df: 鏁版嵁闆?            feature_cols: 鐗瑰緛鍒楀悕

        Returns:
            DataFrame: 鏍囧噯鍖栧悗鐨勬暟鎹泦
        """
        if feature_cols is None:
            feature_cols = self.feature_cols

        if self.scaler is None:
            raise ValueError("璇峰厛璋冪敤 fit_scaler() 璁粌鏍囧噯鍖栧櫒")

        df = df.copy()

        # 鏍囧噯鍖栫壒寰佸垪
        self._validate_scaler_features(feature_cols)
        df[feature_cols] = self.scaler.transform(df[feature_cols])

        return df

    def prepare_train_data(self, stock_data_dict, val_split=None, test_split=None):
        """
        鍑嗗璁粌鏁版嵁锛堝畬鏁存祦绋嬶紝鏀寔鏃跺簭绐楀彛锛?
        Args:
            stock_data_dict: 鑲＄エ鏁版嵁瀛楀吀
            val_split: 楠岃瘉闆嗘瘮渚?
        Returns:
            tuple: 训练/验证/测试张量、特征列和对应元数据
        """
        dataset = self.build_dataset(stock_data_dict)

        train_df, val_df, test_df = self.split_train_val_test(
            dataset, val_split, test_split
        )

        feature_cols = self.get_feature_columns(train_df)
        feature_cols = self.select_feature_columns(train_df, feature_cols)

        # 璁粌鏍囧噯鍖栧櫒
        self.fit_scaler(train_df, feature_cols)

        train_df = self.transform_features(train_df, feature_cols)
        val_df = self.transform_features(val_df, feature_cols)
        test_df = self.transform_features(test_df, feature_cols)

        # 鍒涘缓鏃跺簭绐楀彛鏁版嵁
        print("\n鍒涘缓璁粌闆嗘椂搴忕獥鍙?..")
        X_train, y_train, meta_train = self.create_sequences(train_df, feature_cols)

        print("鍒涘缓楠岃瘉闆嗘椂搴忕獥鍙?..")
        X_val, y_val, meta_val = self.create_sequences(val_df, feature_cols)

        print("创建最终测试集时序窗口...")
        X_test, y_test, meta_test = self.create_sequences(test_df, feature_cols)

        print(f"\n璁粌闆? X={X_train.shape}, y={y_train.shape}")
        print(f"楠岃瘉闆? X={X_val.shape}, y={y_val.shape}")
        print(f"测试集: X={X_test.shape}, y={y_test.shape}")

        return (
            X_train, y_train, X_val, y_val, X_test, y_test,
            feature_cols, meta_train, meta_val, meta_test
        )

    def prepare_tabular_data(self, stock_data_dict, stock_meta=None):
        """
        鍑嗗鎴潰琛ㄦ牸鏁版嵁锛堢敤浜?XGBoost / 绾挎€фā鍨嬶級
        涓嶆瀯寤烘椂搴忕獥鍙ｏ紝姣忚 = 鏌愯偂绁ㄦ煇鏃ョ殑鎴潰鏍锋湰

        Args:
            stock_data_dict: {ts_code: DataFrame}
            stock_meta: {ts_code: dict} 鍚?sector, market_cap 绛夊厓鏁版嵁

        Returns:
            DataFrame: 鍚堝苟鍚庣殑鎴潰鏁版嵁
        """
        print("鏋勫缓鎴潰琛ㄦ牸鏁版嵁闆嗭紙鐢ㄤ簬 XGBoost锛?..")
        all_dfs = []

        for ts_code, df in stock_data_dict.items():
            try:
                df = df.copy()
                df = self.feature_engineer.add_all_features(df)
                df = self.create_target(df)
                df['ts_code'] = ts_code

                if stock_meta and ts_code in stock_meta:
                    meta = stock_meta[ts_code]
                    if 'sector' in meta and 'sector' not in df.columns:
                        df['sector'] = meta['sector']
                    if 'market_cap' in meta and 'market_cap' not in df.columns:
                        df['market_cap'] = meta['market_cap']

                df = self.finalize_feature_frame(df, require_target=True)
                if len(df) > 0:
                    all_dfs.append(df)
            except Exception as e:
                print(f"澶勭悊 {ts_code} 澶辫触: {e}")
                continue

        if not all_dfs:
            raise ValueError("没有有效的数据")

        dataset = pd.concat(all_dfs, ignore_index=True)
        dataset = dataset.sort_values(['trade_date', 'ts_code']).reset_index(drop=True)
        dataset = self.add_cross_section_features(dataset)
        dataset = self.apply_universe_eligibility(dataset)
        dataset = dataset[dataset['universe_eligible']].copy()
        dataset = self.apply_target_normalization(dataset)
        print(f"截面数据集构建完成: {len(dataset)} 条样本, {len(dataset['trade_date'].unique())} 个交易日")
        return dataset

    def walkforward_split(self, dataset, train_months=None, test_months=None, step_months=None):
        """
        Walk-forward 婊氬姩浜ゅ弶楠岃瘉鍒掑垎

        鏃堕棿杞翠弗鏍奸殧绂伙細璁粌闆嗗彧鍚繃鍘绘暟鎹紝娴嬭瘯闆嗗彧鍚湭鏉ユ暟鎹?
        Args:
            dataset: DataFrame锛堝繀椤诲惈 trade_date 鍒楋級
            train_months: 璁粌绐楀彛鏈堟暟锛堥粯璁?config.ROLL_TRAIN_MONTHS锛?            test_months: 娴嬭瘯绐楀彛鏈堟暟锛堥粯璁?config.ROLL_TEST_MONTHS锛?            step_months: 婊氬姩姝ラ暱鏈堟暟锛堥粯璁?config.ROLL_STEP_MONTHS锛?
        Yields:
            tuple: (train_dates, test_dates, train_df, test_df)
        """
        if train_months is None:
            train_months = config.ROLL_TRAIN_MONTHS
        if test_months is None:
            test_months = config.ROLL_TEST_MONTHS
        if step_months is None:
            step_months = config.ROLL_STEP_MONTHS

        from datetime import datetime
        from dateutil.relativedelta import relativedelta

        dates = sorted(dataset['trade_date'].unique())
        if len(dates) < 20:
            raise ValueError(f"鏁版嵁閲忎笉瓒充互杩涜婊氬姩鍒掑垎锛屼粎 {len(dates)} 涓氦鏄撴棩")

        date_objs = [datetime.strptime(str(d), '%Y%m%d') for d in dates]
        start = date_objs[0]
        end = date_objs[-1]

        current_end = start + relativedelta(months=train_months)

        fold = 0
        while current_end + relativedelta(months=test_months) <= end + relativedelta(days=1):
            train_start_dt = current_end - relativedelta(months=train_months)
            train_end_dt = current_end
            test_start_dt = current_end
            test_end_dt = current_end + relativedelta(months=test_months)

            purge_days = int(getattr(config, 'PURGE_DAYS', 0) or 0)
            train_candidates = [
                i for i, d in enumerate(date_objs)
                if train_start_dt <= d <= train_end_dt
            ]
            if purge_days > 0:
                train_candidates = train_candidates[:-purge_days]
            train_index = set(train_candidates)
            train_mask = [i in train_index for i in range(len(date_objs))]
            test_mask = [(test_start_dt < d <= test_end_dt) for d in date_objs]

            train_dates = [dates[i] for i in range(len(dates)) if train_mask[i]]
            test_dates = [dates[i] for i in range(len(dates)) if test_mask[i]]

            if len(train_dates) < 20 or len(test_dates) < 5:
                current_end += relativedelta(months=step_months)
                continue

            train_df = dataset[dataset['trade_date'].isin(train_dates)].copy()
            test_df = dataset[dataset['trade_date'].isin(test_dates)].copy()

            fold += 1
            print(f"  Fold {fold}: 璁粌 {train_dates[0]}~{train_dates[-1]} ({len(train_dates)}澶?, "
                  f"娴嬭瘯 {test_dates[0]}~{test_dates[-1]} ({len(test_dates)}澶?")

            yield train_dates, test_dates, train_df, test_df
            current_end += relativedelta(months=step_months)

        if fold == 0:
            raise ValueError("滚动划分未生成有效fold，请检查数据时间范围")

    def prepare_predict_data(self, stock_data_dict, expected_trade_date=None):
        """
        鍑嗗棰勬祴鏁版嵁锛堜娇鐢ㄦ渶杩?N 澶╃殑鏁版嵁锛?
        Args:
            stock_data_dict: 鑲＄エ鏁版嵁瀛楀吀

        Returns:
            tuple: (X, df_info) - 鐗瑰緛鍜屽搴旂殑鑲＄エ淇℃伅
        """
        if self.scaler is None or self.feature_cols is None:
            raise ValueError("请先训练模型或加载标准化器")

        predict_dfs = []
        expected_trade_date = (
            str(expected_trade_date) if expected_trade_date is not None else None
        )

        for ts_code, df in stock_data_dict.items():
            try:
                prepared_df = self.feature_engineer.add_all_features(df)
                prepared_df['ts_code'] = ts_code
                prepared_df = self.finalize_feature_frame(
                    prepared_df, require_target=False
                )
                predict_dfs.append(prepared_df)

            except Exception as e:
                print(f"鍑嗗棰勬祴鏁版嵁 {ts_code} 澶辫触: {e}")
                continue

        if not predict_dfs:
            raise ValueError("没有有效的预测数据")

        df_predict = pd.concat(predict_dfs, ignore_index=True)
        df_predict = self.add_cross_section_features(df_predict)
        df_predict = df_predict.replace([np.inf, -np.inf], np.nan)

        missing = [col for col in self.feature_cols if col not in df_predict.columns]
        if missing:
            raise ValueError(f"预测数据缺少训练特征: {missing[:10]}")
        df_predict = df_predict.dropna(subset=self.feature_cols)

        # 预测特征必须与训练路径完全一致，只允许标准化一次。
        df_predict = self.transform_features(df_predict, self.feature_cols)

        # 鍒涘缓鏃跺簭绐楀彛锛堟瘡鍙偂绁ㄤ竴涓獥鍙ｏ級
        X_list = []
        info_list = []

        grouped = df_predict.groupby('ts_code')

        for ts_code, group_df in grouped:
            group_df = group_df.sort_values('trade_date').reset_index(drop=True)

            if len(group_df) < config.SEQUENCE_LENGTH:
                continue
            last_row = group_df.iloc[-1]
            if expected_trade_date and str(last_row['trade_date']) != expected_trade_date:
                continue

            recent = group_df.tail(config.SEQUENCE_LENGTH)
            X_seq = recent[self.feature_cols].values
            if not np.isfinite(X_seq).all():
                continue

            X_list.append(X_seq)
            info_list.append({
                'ts_code': last_row['ts_code'],
                'trade_date': last_row['trade_date'],
                'close': last_row.get('raw_close', last_row['close']),
                'sector': last_row.get('sector', config.DEFAULT_SECTOR),
                'market_cap': last_row.get('market_cap', np.nan)
            })

        # 杞崲涓?numpy 鏁扮粍
        X = np.array(X_list, dtype=np.float32)
        df_info = pd.DataFrame(info_list)

        print(f"预测数据准备完成: {len(X)} 只股票")

        return X, df_info

    def save_scaler(self, scaler_file=None, feature_cols_file=None):
        """
        淇濆瓨鏍囧噯鍖栧櫒鍜岀壒寰佸垪

        Args:
            scaler_file: 鏍囧噯鍖栧櫒鏂囦欢璺緞
            feature_cols_file: 鐗瑰緛鍒楁枃浠惰矾寰?        """
        if scaler_file is None:
            scaler_file = config.SCALER_FILE
        if feature_cols_file is None:
            feature_cols_file = config.FEATURE_COLS_FILE

        self._validate_scaler_features(self.feature_cols)

        # 淇濆瓨鏍囧噯鍖栧櫒
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)

        # 淇濆瓨鐗瑰緛鍒?        with open(feature_cols_file, 'wb') as f:
        with open(feature_cols_file, 'wb') as f:
            pickle.dump(self.feature_cols, f)

        print(f"鏍囧噯鍖栧櫒宸蹭繚瀛? {scaler_file}")
        print(f"鐗瑰緛鍒楀凡淇濆瓨: {feature_cols_file}")

    def load_scaler(self, scaler_file=None, feature_cols_file=None):
        """
        鍔犺浇鏍囧噯鍖栧櫒鍜岀壒寰佸垪

        Args:
            scaler_file: 鏍囧噯鍖栧櫒鏂囦欢璺緞
            feature_cols_file: 鐗瑰緛鍒楁枃浠惰矾寰?        """
        if scaler_file is None:
            scaler_file = config.SCALER_FILE
        if feature_cols_file is None:
            feature_cols_file = config.FEATURE_COLS_FILE

        # 鍔犺浇鏍囧噯鍖栧櫒
        with open(scaler_file, 'rb') as f:
            self.scaler = pickle.load(f)

        # 鍔犺浇鐗瑰緛鍒?        with open(feature_cols_file, 'rb') as f:
        with open(feature_cols_file, 'rb') as f:
            self.feature_cols = pickle.load(f)

        self._validate_scaler_features(self.feature_cols)

        print(f"鏍囧噯鍖栧櫒宸插姞杞? {scaler_file}")
        print(f"鐗瑰緛鍒楀凡鍔犺浇锛屾暟閲? {len(self.feature_cols)}")
    def _validate_scaler_features(self, feature_cols):
        """校验标准化器与特征清单的维度和顺序。"""
        if self.scaler is None or feature_cols is None:
            raise ValueError("标准化器或特征清单未加载")

        expected = getattr(self.scaler, 'n_features_in_', None)
        if expected is not None and expected != len(feature_cols):
            raise ValueError(
                f"特征维度不一致: scaler={expected}, feature_cols={len(feature_cols)}"
            )

        scaler_cols = getattr(self.scaler, 'feature_names_in_', None)
        if scaler_cols is not None and list(scaler_cols) != list(feature_cols):
            raise ValueError("标准化器与特征清单的顺序不一致")


def main():
    """Test dataset builder."""
    from data_loader import DataLoader

    loader = DataLoader()
    builder = DatasetBuilder()

    # 鑾峰彇娴嬭瘯鏁版嵁
    trade_date = loader.get_latest_trade_date()
    stock_list = loader.get_stock_list(trade_date)

    # 鍙彇鍓?鍙偂绁ㄦ祴璇?    test_stocks = stock_list.head(5)

    stock_data = {}
    for idx, row in test_stocks.iterrows():
        ts_code = row['ts_code']
        df = loader.get_stock_daily(ts_code)
        if df is not None and len(df) >= config.MIN_HISTORY_DAYS:
            stock_data[ts_code] = df

    if stock_data:
        print(f"测试数据: {len(stock_data)} 只股票")

        # 鍑嗗璁粌鏁版嵁
        (
            X_train, y_train, X_val, y_val, X_test, y_test,
            feature_cols, _, _, _
        ) = builder.prepare_train_data(stock_data)

        print(f"\n鐗瑰緛鍒? {feature_cols[:5]}...")
        print(f"鏃跺簭绐楀彛闀垮害: {config.SEQUENCE_LENGTH}")
        print(f"鐩爣缁熻: mean={y_train.mean():.4f}, std={y_train.std():.4f}")


if __name__ == "__main__":
    main()
