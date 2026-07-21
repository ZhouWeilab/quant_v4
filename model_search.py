"""以滚动样本外扣费后收益为目标的模型与交易口径搜索。"""

import gc
import hashlib
import json
import os
import pickle
from contextlib import contextmanager
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

import config
from baseline_models import XGBoostModel, _daily_regression_metrics
from data_loader import DataLoader
from dataset import DatasetBuilder


@contextmanager
def temporary_config(**updates):
    """临时覆盖全局配置，退出时恢复，避免不同实验互相污染。"""
    original = {name: getattr(config, name) for name in updates}
    try:
        for name, value in updates.items():
            setattr(config, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(config, name, value)


class NetReturnModelSearch:
    """按严格时间切分比较模型、训练窗、股票池和持有期。"""

    def __init__(self, end_date=None):
        self.loader = DataLoader()
        self.end_date = end_date
        self.stock_data = None
        self.stock_meta = None
        self.output_dir = getattr(
            config, 'MODEL_SEARCH_OUTPUT_DIR',
            os.path.join(config.MODEL_DIR, 'model_search')
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def load_market_data(self):
        """一次加载搜索所需的最大历史股票池，后续实验复用。"""
        self.end_date = self.end_date or self.loader.get_latest_trade_date()
        stocks = self.loader.get_stock_list(self.end_date, historical=True)
        sample_count = getattr(config, 'RUNTIME_SAMPLE_STOCK_COUNT', None)
        if sample_count:
            stocks = stocks.head(sample_count).reset_index(drop=True)
        self.stock_meta = {
            row['ts_code']: {
                'sector': row.get('sector', config.DEFAULT_SECTOR),
                'market_cap': row.get('market_cap', np.nan),
            }
            for _, row in stocks.iterrows()
        }
        self.stock_data = self.loader.get_all_stocks_daily(
            stocks, start_date=config.START_DATE, end_date=self.end_date
        )
        if not self.stock_data:
            raise ValueError("模型搜索未获取到有效历史数据")

    def build_horizon_dataset(self, holding_days):
        """构建指定持有期标签；买入始终为T+1开盘。"""
        entry_shift = int(config.LABEL_ENTRY_SHIFT)
        exit_shift = entry_shift + int(holding_days)
        with temporary_config(
            FORECAST_HORIZON=int(holding_days),
            LABEL_EXIT_SHIFT=exit_shift,
            PURGE_DAYS=exit_shift,
            BACKTEST_REBALANCE_DAYS=int(holding_days),
        ):
            builder = DatasetBuilder()
            return builder.prepare_tabular_data(self.stock_data, self.stock_meta)

    @staticmethod
    def filter_stock_pool(dataset, pool_limit):
        """按每个交易日当时可见的市值选择流动性更好的股票池。"""
        frame = dataset
        if pool_limit and 'market_cap' in frame.columns:
            cap = pd.to_numeric(frame['market_cap'], errors='coerce')
            cap_rank = cap.groupby(frame['trade_date']).rank(
                ascending=False, method='first'
            )
            frame = frame[cap.notna() & cap_rank.le(int(pool_limit))].copy()
        else:
            frame = frame.copy()
        frame = DatasetBuilder.add_cross_section_features(frame)
        frame = DatasetBuilder().apply_target_normalization(frame)
        return frame.sort_values(['trade_date', 'ts_code']).reset_index(drop=True)

    @staticmethod
    def _recent_sample(frame, max_rows):
        """模型限样时保留最近的完整交易日，不随机打散时间结构。"""
        max_rows = int(max_rows or 0)
        if max_rows <= 0 or len(frame) <= max_rows:
            return frame
        counts = frame.groupby('trade_date').size().sort_index(ascending=False)
        selected_dates = []
        rows = 0
        for trade_date, count in counts.items():
            if selected_dates and rows + int(count) > max_rows:
                continue
            selected_dates.append(trade_date)
            rows += int(count)
            if rows >= max_rows:
                break
        return frame[frame['trade_date'].isin(selected_dates)].copy()

    @staticmethod
    def _fit_model(model_name, train_df, val_df, feature_cols):
        X_train = train_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
        y_train = train_df['target'].to_numpy(dtype=np.float32, copy=False)
        if model_name == 'ridge':
            model = Pipeline([
                ('impute', SimpleImputer(strategy='median')),
                ('scale', RobustScaler()),
                ('model', Ridge(alpha=10.0)),
            ])
            model.fit(X_train, y_train)
            return model

        if model_name == 'mlp':
            sampled = NetReturnModelSearch._recent_sample(
                train_df, getattr(config, 'MODEL_SEARCH_MLP_MAX_SAMPLES', 0)
            )
            model = Pipeline([
                ('impute', SimpleImputer(strategy='median')),
                ('scale', RobustScaler()),
                ('model', MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation='relu',
                    alpha=0.001,
                    batch_size=1024,
                    learning_rate_init=0.001,
                    max_iter=40,
                    early_stopping=True,
                    validation_fraction=0.10,
                    n_iter_no_change=6,
                    random_state=config.RANDOM_SEED,
                )),
            ])
            model.fit(
                sampled[feature_cols].to_numpy(dtype=np.float32, copy=False),
                sampled['target'].to_numpy(dtype=np.float32, copy=False)
            )
            return model

        if model_name == 'xgb':
            model = XGBoostModel()
            model.fit(
                X_train,
                y_train,
                val_df[feature_cols].to_numpy(dtype=np.float32, copy=False),
                val_df['target'].to_numpy(dtype=np.float32, copy=False),
                verbose=False,
            )
            return model

        raise ValueError(f"不支持的搜索模型: {model_name}")

    @staticmethod
    def _predict_model(model_name, model, test_df, feature_cols):
        X_test = test_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
        if model_name == 'xgb':
            return model.predict(X_test)
        return model.predict(X_test)

    def evaluate_config(self, dataset, holding_days, pool_limit,
                        train_months, model_names):
        """在同一组fold和特征上比较多个模型。"""
        builder = DatasetBuilder()
        folds = list(builder.walkforward_split(
            dataset,
            train_months=int(train_months),
            test_months=config.ROLL_TEST_MONTHS,
            step_months=config.ROLL_STEP_MONTHS,
        ))
        max_folds = int(getattr(config, 'MODEL_SEARCH_MAX_FOLDS', 0) or 0)
        if max_folds > 0:
            folds = folds[-max_folds:]
        rows = []

        for fold_no, (train_dates, test_dates, train_df, test_df) in enumerate(folds, 1):
            inner_train, inner_val = builder.split_dataset(
                train_df, val_split=config.ROLL_VAL_RATIO
            )
            feature_cols = builder.get_feature_columns(inner_train)
            feature_cols = builder.select_feature_columns(inner_train, feature_cols)
            raw_col = 'target_raw' if 'target_raw' in test_df.columns else 'target_return'

            for model_name in model_names:
                print(
                    f"搜索 fold {fold_no}/{len(folds)}: model={model_name}, "
                    f"holding={holding_days}, pool={pool_limit}, train={train_months}m"
                )
                model = self._fit_model(
                    model_name, inner_train, inner_val, feature_cols
                )
                predictions = self._predict_model(
                    model_name, model, test_df, feature_cols
                )
                metrics = _daily_regression_metrics(
                    predictions,
                    test_df[raw_col].to_numpy(dtype=float),
                    test_df['trade_date'].to_numpy(),
                )
                row = {
                    'model': model_name,
                    'holding_days': int(holding_days),
                    'pool_limit': int(pool_limit),
                    'train_months': int(train_months),
                    'fold': int(fold_no),
                    'train_start': train_dates[0],
                    'train_end': train_dates[-1],
                    'test_start': test_dates[0],
                    'test_end': test_dates[-1],
                    'feature_count': len(feature_cols),
                    'feature_hash': hashlib.sha256(
                        '\n'.join(feature_cols).encode('utf-8')
                    ).hexdigest(),
                }
                for key, value in metrics.items():
                    if not isinstance(value, (list, tuple, dict)):
                        row[key] = value
                rows.append(row)
                del model, predictions
                gc.collect()
        return rows

    @staticmethod
    def summarize(rows):
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        top_n = int(getattr(config, 'MODEL_SEARCH_TOP_N', 10))
        net_col = f'top_{top_n}_net_return'
        summary = (
            frame.groupby(
                ['model', 'holding_days', 'pool_limit', 'train_months'],
                as_index=False
            )
            .agg(
                fold_count=('fold', 'count'),
                rank_ic_mean=('rank_ic', 'mean'),
                rank_ic_std=('rank_ic', 'std'),
                net_return_mean=(net_col, 'mean'),
                net_return_std=(net_col, 'std'),
                selection_score_mean=('selection_score', 'mean'),
                positive_net_fold_ratio=(net_col, lambda x: float((x > 0).mean())),
            )
        )
        summary['objective_score'] = (
            summary['net_return_mean']
            - float(getattr(config, 'VALIDATION_RETURN_SE_PENALTY', 0.0))
            * summary['net_return_std'].fillna(0.0)
            / np.sqrt(summary['fold_count'].clip(lower=1))
        )
        summary['accepted'] = (
            summary['rank_ic_mean'].gt(0)
            & summary['net_return_mean'].gt(0)
            & summary['positive_net_fold_ratio'].ge(0.5)
        )
        return summary.sort_values(
            ['accepted', 'objective_score'], ascending=[False, False]
        ).reset_index(drop=True)

    @staticmethod
    def _select_stage2_configs(ridge_summary):
        limit = int(getattr(config, 'MODEL_SEARCH_STAGE2_CONFIGS', 4) or 4)
        selected = []
        # 先保证5日和10日口径都有机会进入非线性模型复核。
        for _, group in ridge_summary.groupby('holding_days', sort=False):
            selected.append(group.iloc[0])
        for _, row in ridge_summary.iterrows():
            key = (row['holding_days'], row['pool_limit'], row['train_months'])
            existing = {
                (item['holding_days'], item['pool_limit'], item['train_months'])
                for item in selected
            }
            if key not in existing:
                selected.append(row)
            if len(selected) >= limit:
                break
        return selected[:limit]

    def fit_production_model(self, best, dataset):
        """仅将通过样本外门槛的最佳表格模型训练为生产候选。"""
        builder = DatasetBuilder()
        train_df, val_df = builder.split_dataset(
            dataset, val_split=config.ROLL_VAL_RATIO
        )
        feature_cols = builder.get_feature_columns(train_df)
        feature_cols = builder.select_feature_columns(train_df, feature_cols)
        model_name = str(best['model'])
        model = self._fit_model(model_name, train_df, val_df, feature_cols)
        stored_model = model.model if model_name == 'xgb' else model
        artifact = {
            'model': stored_model,
            'model_name': model_name,
            'feature_cols': feature_cols,
        }
        staging_file = config.TABULAR_MODEL_FILE + '.tmp'
        with open(staging_file, 'wb') as file:
            pickle.dump(artifact, file)
        os.replace(staging_file, config.TABULAR_MODEL_FILE)

        meta = {
            'model_type': 'tabular',
            'model_name': model_name,
            'trained_at': datetime.now().isoformat(timespec='seconds'),
            'data_start_date': str(dataset['trade_date'].min()),
            'data_end_date': str(dataset['trade_date'].max()),
            'label_entry_shift': int(config.LABEL_ENTRY_SHIFT),
            'label_exit_shift': int(config.LABEL_ENTRY_SHIFT + int(best['holding_days'])),
            'label_entry_price': 'open',
            'label_exit_price': 'close',
            'target_normalization': config.TARGET_NORMALIZATION,
            'holding_days': int(best['holding_days']),
            'pool_limit': int(best['pool_limit']),
            'train_months': int(best['train_months']),
            'feature_count': len(feature_cols),
            'feature_hash': hashlib.sha256(
                '\n'.join(feature_cols).encode('utf-8')
            ).hexdigest(),
            'walkforward_metrics': {
                key: (bool(value) if isinstance(value, (bool, np.bool_)) else float(value))
                for key, value in best.items()
                if key not in {'model'} and isinstance(value, (int, float, np.number, bool))
            },
        }
        staging_meta = config.TABULAR_MODEL_META_FILE + '.tmp'
        with open(staging_meta, 'w', encoding='utf-8') as file:
            json.dump(meta, file, ensure_ascii=False, indent=2)
        os.replace(staging_meta, config.TABULAR_MODEL_META_FILE)
        return meta

    def run(self):
        self.load_market_data()
        stage1_rows = []
        horizons = list(getattr(config, 'MODEL_SEARCH_HOLDING_DAYS', [5, 10]))
        pool_limits = list(getattr(config, 'MODEL_SEARCH_POOL_LIMITS', [1000, 2000]))
        train_windows = list(getattr(config, 'MODEL_SEARCH_TRAIN_MONTHS', [24, 36]))

        for holding_days in horizons:
            base_dataset = self.build_horizon_dataset(holding_days)
            for pool_limit, train_months in product(pool_limits, train_windows):
                pool_dataset = self.filter_stock_pool(base_dataset, pool_limit)
                with temporary_config(
                    PURGE_DAYS=int(config.LABEL_ENTRY_SHIFT + int(holding_days))
                ):
                    stage1_rows.extend(self.evaluate_config(
                        pool_dataset, holding_days, pool_limit,
                        train_months, ['ridge']
                    ))
                del pool_dataset
                gc.collect()
            del base_dataset
            gc.collect()

        ridge_summary = self.summarize(stage1_rows)
        stage2_configs = self._select_stage2_configs(ridge_summary)
        stage2_rows = []
        for holding_days, config_group in pd.DataFrame(stage2_configs).groupby('holding_days'):
            base_dataset = self.build_horizon_dataset(int(holding_days))
            for _, candidate in config_group.iterrows():
                pool_limit = int(candidate['pool_limit'])
                train_months = int(candidate['train_months'])
                pool_dataset = self.filter_stock_pool(base_dataset, pool_limit)
                with temporary_config(
                    PURGE_DAYS=int(config.LABEL_ENTRY_SHIFT + int(holding_days))
                ):
                    stage2_rows.extend(self.evaluate_config(
                        pool_dataset, int(holding_days), pool_limit,
                        train_months, ['xgb', 'mlp']
                    ))
                del pool_dataset
                gc.collect()
            del base_dataset
            gc.collect()

        all_rows = stage1_rows + stage2_rows
        fold_report = pd.DataFrame(all_rows)
        summary = self.summarize(all_rows)
        fold_file = os.path.join(
            self.output_dir, f'model_search_folds_{self.end_date}.csv'
        )
        summary_file = os.path.join(
            self.output_dir, f'model_search_summary_{self.end_date}.csv'
        )
        fold_report.to_csv(fold_file, index=False, encoding='utf-8-sig')
        summary.to_csv(summary_file, index=False, encoding='utf-8-sig')

        best = summary.iloc[0].to_dict()
        best['end_date'] = self.end_date
        best['production_promoted'] = False
        if (
            bool(best['accepted'])
            and bool(getattr(config, 'MODEL_SEARCH_PROMOTE_PRODUCTION', True))
        ):
            dataset = self.filter_stock_pool(
                self.build_horizon_dataset(int(best['holding_days'])),
                int(best['pool_limit'])
            )
            with temporary_config(
                LABEL_EXIT_SHIFT=int(config.LABEL_ENTRY_SHIFT + int(best['holding_days'])),
                FORECAST_HORIZON=int(best['holding_days']),
                PURGE_DAYS=int(config.LABEL_ENTRY_SHIFT + int(best['holding_days'])),
            ):
                best['production_meta'] = self.fit_production_model(best, dataset)
            best['production_promoted'] = True

        best_file = os.path.join(
            self.output_dir, f'best_config_{self.end_date}.json'
        )
        with open(best_file, 'w', encoding='utf-8') as file:
            json.dump(best, file, ensure_ascii=False, indent=2, default=str)
        print(f"模型搜索fold报告: {fold_file}")
        print(f"模型搜索汇总: {summary_file}")
        print(f"最佳配置: {best_file}")
        return summary


def main():
    summary = NetReturnModelSearch().run()
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
