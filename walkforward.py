"""深度学习滚动样本外评估。"""

import gc
import json
import os

import numpy as np
import pandas as pd
from tensorflow import keras

import config
from data_loader import DataLoader
from dataset import DatasetBuilder
from model import ModelTrainer


class WalkForwardEvaluator:
    """每个fold仅使用测试期之前的数据训练并评估。"""

    def __init__(self):
        self.loader = DataLoader()

    @staticmethod
    def _prepare_fold(train_df, test_df):
        builder = DatasetBuilder()
        inner_train, inner_val = builder.split_dataset(
            train_df, val_split=config.ROLL_VAL_RATIO
        )
        feature_cols = builder.get_feature_columns(inner_train)
        feature_cols = builder.select_feature_columns(inner_train, feature_cols)
        builder.fit_scaler(inner_train, feature_cols)

        inner_train = builder.transform_features(inner_train, feature_cols)
        inner_val = builder.transform_features(inner_val, feature_cols)
        test_df = builder.transform_features(test_df, feature_cols)

        X_train, y_train, meta_train = builder.create_sequences(
            inner_train, feature_cols
        )
        X_val, y_val, meta_val = builder.create_sequences(
            inner_val, feature_cols
        )
        X_test, y_test, meta_test = builder.create_sequences(
            test_df, feature_cols
        )
        return (
            builder, X_train, y_train, X_val, y_val, X_test, y_test,
            feature_cols, meta_train, meta_val, meta_test
        )

    @staticmethod
    def _round_trip_cost():
        return (
            2 * config.BACKTEST_COMMISSION
            + 2 * config.BACKTEST_TRANSFER_FEE
            + 2 * config.BACKTEST_SLIPPAGE
            + config.BACKTEST_STAMP_DUTY
        )

    def run(self, end_date=None):
        end_date = end_date or self.loader.get_latest_trade_date()
        stocks = self.loader.get_stock_list(end_date, historical=True)
        sample_count = getattr(config, 'RUNTIME_SAMPLE_STOCK_COUNT', None)
        if sample_count:
            stocks = stocks.head(sample_count).reset_index(drop=True)

        stock_data = self.loader.get_all_stocks_daily(
            stocks, start_date=config.START_DATE, end_date=end_date
        )
        base_builder = DatasetBuilder()
        dataset = base_builder.build_dataset(stock_data)
        folds = list(base_builder.walkforward_split(dataset))
        max_folds = int(getattr(config, 'WALKFORWARD_MAX_FOLDS', 0) or 0)
        if max_folds > 0:
            folds = folds[-max_folds:]

        output_dir = os.path.join(config.MODEL_DIR, 'walkforward')
        os.makedirs(output_dir, exist_ok=True)
        rows = []
        original_epochs = config.EPOCHS
        original_refit = config.PRODUCTION_REFIT_EPOCHS
        config.EPOCHS = int(getattr(config, 'WALKFORWARD_EPOCHS', original_epochs))
        config.PRODUCTION_REFIT_EPOCHS = 0

        try:
            for fold_no, (train_dates, test_dates, train_df, test_df) in enumerate(folds, 1):
                print(
                    f"\nWalk-forward fold {fold_no}/{len(folds)}: "
                    f"训练至 {train_dates[-1]}, 测试 {test_dates[0]}~{test_dates[-1]}"
                )
                parts = self._prepare_fold(train_df, test_df)
                (
                    builder, X_train, y_train, X_val, y_val, X_test, y_test,
                    feature_cols, meta_train, meta_val, meta_test
                ) = parts
                model_file = os.path.join(output_dir, f'fold_{fold_no}.h5')
                trainer = ModelTrainer()
                model = trainer.train_model(
                    X_train, y_train, X_val, y_val,
                    input_shape=(config.SEQUENCE_LENGTH, len(feature_cols)),
                    meta_train=meta_train,
                    meta_val=meta_val,
                    X_test=X_test,
                    y_test=y_test,
                    meta_test=meta_test,
                    model_file=model_file
                )
                metrics = dict(getattr(model, 'test_metrics', {}))
                metrics['fold'] = fold_no
                metrics['train_start'] = train_dates[0]
                metrics['train_end'] = train_dates[-1]
                metrics['test_start'] = test_dates[0]
                metrics['test_end'] = test_dates[-1]
                for top_n in getattr(config, 'EVAL_TOP_N_LIST', [5, 10, 20]):
                    key = f'top_{top_n}_return'
                    if key in metrics:
                        metrics[f'top_{top_n}_net_return'] = (
                            metrics[key] - self._round_trip_cost()
                        )
                if 'quantile_returns' in metrics:
                    metrics['quantile_returns'] = json.dumps(
                        metrics['quantile_returns'], ensure_ascii=False
                    )
                rows.append(metrics)

                del parts, builder, trainer, model
                del X_train, y_train, X_val, y_val, X_test, y_test
                keras.backend.clear_session()
                gc.collect()
        finally:
            config.EPOCHS = original_epochs
            config.PRODUCTION_REFIT_EPOCHS = original_refit

        report = pd.DataFrame(rows)
        if report.empty:
            raise ValueError("Walk-forward 未生成有效评估结果")
        report_file = os.path.join(
            output_dir, f'walkforward_report_{end_date}.csv'
        )
        report.to_csv(report_file, index=False, encoding='utf-8-sig')
        rank_ic = pd.to_numeric(report.get('rank_ic'), errors='coerce').dropna()
        summary = {
            'fold_count': int(len(report)),
            'rank_ic_mean': float(rank_ic.mean()) if len(rank_ic) else None,
            'rank_ic_std': float(rank_ic.std(ddof=1)) if len(rank_ic) > 1 else None,
            'rank_ic_ir': (
                float(rank_ic.mean() / rank_ic.std(ddof=1))
                if len(rank_ic) > 1 and rank_ic.std(ddof=1) > 0 else None
            ),
        }
        for top_n in getattr(config, 'EVAL_TOP_N_LIST', [5, 10, 20]):
            key = f'top_{top_n}_net_return'
            if key in report.columns:
                values = pd.to_numeric(report[key], errors='coerce')
                summary[f'{key}_mean'] = float(values.mean())
                summary[f'{key}_positive_fold_ratio'] = float((values > 0).mean())
        summary_file = os.path.join(
            output_dir, f'walkforward_summary_{end_date}.json'
        )
        with open(summary_file, 'w', encoding='utf-8') as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
        print(f"Walk-forward报告已保存: {report_file}")
        print(f"Walk-forward汇总已保存: {summary_file}")
        return report


def main():
    report = WalkForwardEvaluator().run()
    print(report.to_string(index=False))


if __name__ == '__main__':
    main()
