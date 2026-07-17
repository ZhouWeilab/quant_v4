"""关键数据、模型元数据与成交模拟测试，不依赖Tushare网络。"""

import hashlib
import json
import tempfile
import numpy as np
import pandas as pd

import config
from data_loader import DataLoader
from dataset import DatasetBuilder
from features import FeatureEngineer
from model import (
    CrossSectionalRankLoss,
    DateGroupedBatchSequence,
    calculate_daily_ranking_metrics,
)
from model_search import NetReturnModelSearch
from predictor import BacktestEngine, StockPredictor


class IdentityFeatureEngineer:
    def add_all_features(self, df):
        return df.copy()


class CountingScaler:
    def __init__(self):
        self.calls = 0
        self.n_features_in_ = 1
        self.feature_names_in_ = np.array(['feature_a'])

    def transform(self, values):
        self.calls += 1
        return np.asarray(values, dtype=float) + 10.0


class DelayedDailyPro:
    def __init__(self):
        self.daily_calls = []

    def trade_cal(self, **kwargs):
        return pd.DataFrame({'cal_date': ['20260714', '20260715']})

    def daily(self, trade_date, fields):
        self.daily_calls.append(trade_date)
        if trade_date == '20260715':
            return pd.DataFrame(columns=['ts_code', 'trade_date', 'close'])
        return pd.DataFrame({
            'ts_code': [f'{idx:06d}.SZ' for idx in range(120)],
            'trade_date': [trade_date] * 120,
            'close': [10.0] * 120,
        })


def test_latest_trade_date_waits_for_daily_publication():
    loader = object.__new__(DataLoader)
    loader.pro = DelayedDailyPro()
    original_cache_dir = config.DAILY_CACHE_DIR
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            config.DAILY_CACHE_DIR = temp_dir
            result = loader.get_latest_trade_date(as_of_date='20260715')
    finally:
        config.DAILY_CACHE_DIR = original_cache_dir
    assert result == '20260714'
    assert loader.pro.daily_calls == ['20260715', '20260714']


def test_prediction_scaler_runs_once():
    builder = DatasetBuilder()
    builder.feature_engineer = IdentityFeatureEngineer()
    builder.feature_cols = ['feature_a']
    builder.scaler = CountingScaler()

    dates = pd.date_range('2026-01-01', periods=25, freq='B').strftime('%Y%m%d')
    stock_data = {}
    for code, offset in [('000001.SZ', 0.0), ('600000.SH', 1.0)]:
        values = np.arange(len(dates), dtype=float) + offset
        stock_data[code] = pd.DataFrame({
            'ts_code': code,
            'trade_date': dates,
            'close': 10.0 + values / 100,
            'raw_close': 10.0 + values / 100,
            'feature_a': values,
            'return_1d': values / 1000,
            'return_5d': values / 500,
            'return_20d': values / 200,
            'ma_60': 10.0,
            f'rsi_{config.RSI_PERIOD}': 50.0,
            'amount': 100000.0,
            'sector': '测试行业',
            'market_cap': 1000000.0,
        })

    X, info = builder.prepare_predict_data(
        stock_data, expected_trade_date=dates[-1]
    )
    assert builder.scaler.calls == 1
    assert X.shape == (2, config.SEQUENCE_LENGTH, 1)
    assert len(info) == 2
    assert np.all(X[:, -1, 0] >= 10.0)


def test_purged_train_val_test_split():
    builder = DatasetBuilder()
    dates = pd.date_range('2020-01-01', periods=200, freq='B').strftime('%Y%m%d')
    dataset = pd.DataFrame({
        'trade_date': np.repeat(dates, 2),
        'ts_code': ['000001.SZ', '600000.SH'] * len(dates),
        'target': 0.0,
    })
    train, val, test = builder.split_train_val_test(dataset)
    date_pos = {date: idx for idx, date in enumerate(dates)}
    train_gap = date_pos[val['trade_date'].min()] - date_pos[train['trade_date'].max()] - 1
    test_gap = date_pos[test['trade_date'].min()] - date_pos[val['trade_date'].max()] - 1
    assert train_gap >= config.PURGE_DAYS
    assert test_gap >= config.PURGE_DAYS


def test_walkforward_split_keeps_purge_gap():
    dates = pd.date_range('2020-01-01', periods=320, freq='B').strftime('%Y%m%d')
    frame = pd.DataFrame({
        'trade_date': dates,
        'ts_code': ['000001.SZ'] * len(dates),
        'target': np.arange(len(dates), dtype=float),
    })
    folds = list(DatasetBuilder().walkforward_split(
        frame, train_months=6, test_months=2, step_months=2
    ))
    assert folds
    date_positions = {date: idx for idx, date in enumerate(dates)}
    for train_dates, test_dates, _, _ in folds:
        assert max(train_dates) < min(test_dates)
        gap = date_positions[min(test_dates)] - date_positions[max(train_dates)] - 1
        assert gap >= config.PURGE_DAYS


def test_date_grouped_batches_and_rank_loss():
    X = np.zeros((6, 2, 1), dtype=np.float32)
    y = np.array([-2, -1, 1, -1, 0, 2], dtype=np.float32)
    dates = np.array(['20260101'] * 3 + ['20260102'] * 3)
    sequence = DateGroupedBatchSequence(X, y, dates, shuffle=False)
    assert len(sequence) == 2
    assert all(len(sequence[i][1]) == 3 for i in range(len(sequence)))

    loss = CrossSectionalRankLoss(
        huber_weight=0.0, direction_weight=0.0, ranking_weight=1.0
    )
    aligned = float(loss(y[:3], y[:3, None]).numpy())
    reversed_loss = float(loss(y[:3], -y[:3, None]).numpy())
    assert aligned < reversed_loss


def test_target_uses_next_open():
    builder = DatasetBuilder()
    dates = pd.date_range('2026-01-01', periods=10, freq='B').strftime('%Y%m%d')
    frame = pd.DataFrame({
        'trade_date': dates,
        'open': np.arange(10, dtype=float) + 10.0,
        'close': np.arange(10, dtype=float) + 20.0,
    })
    result = builder.create_target(frame.copy())
    expected = frame.loc[config.LABEL_EXIT_SHIFT, 'close'] / frame.loc[
        config.LABEL_ENTRY_SHIFT, 'open'
    ] - 1
    assert np.isclose(result.iloc[0]['target_return'], expected)


def test_cs_rank_target_is_centered_and_ordered():
    builder = DatasetBuilder()
    dates = np.repeat(['20260101', '20260102'], 10)
    values = np.tile(np.arange(10, dtype=float), 2)
    frame = pd.DataFrame({
        'trade_date': dates,
        'target_return': values,
        'universe_eligible': True,
    })
    old_mode = config.TARGET_NORMALIZATION
    try:
        config.TARGET_NORMALIZATION = 'cs_rank'
        result = builder.apply_target_normalization(frame)
    finally:
        config.TARGET_NORMALIZATION = old_mode
    assert result.groupby('trade_date')['target'].mean().abs().max() < 1e-7
    assert result['target'].between(-1.0, 1.0).all()
    assert result.groupby('trade_date')['target'].apply(
        lambda values: values.is_monotonic_increasing
    ).all()


def test_ranking_metrics_use_relative_direction_and_net_return():
    dates = np.repeat(['20260101', '20260102'], 20)
    raw_returns = np.tile(np.linspace(-0.02, 0.04, 20), 2)
    predictions = raw_returns.copy()
    metrics = calculate_daily_ranking_metrics(
        predictions, raw_returns, dates, top_n_list=[10]
    )
    assert np.isclose(metrics['relative_direction_accuracy'], 1.0)
    assert np.isclose(
        metrics['top_10_net_return'],
        metrics['top_10_return'] - metrics['round_trip_cost']
    )
    assert metrics['selection_score'] <= metrics['top_10_net_return']


def test_nonstationary_absolute_features_are_excluded():
    frame = pd.DataFrame({
        'ma_5': [10.0, 11.0],
        'boll_upper': [12.0, 13.0],
        'buy_elg_vol': [100.0, 200.0],
        'macd_norm': [0.01, 0.02],
        'return_5d': [0.03, 0.04],
    })
    cols = FeatureEngineer().get_feature_columns(frame)
    assert 'ma_5' not in cols
    assert 'boll_upper' not in cols
    assert 'buy_elg_vol' not in cols
    assert 'macd_norm' in cols
    assert 'return_5d' in cols


def test_sparse_event_values_are_not_forward_filled():
    frame = pd.DataFrame({'net_amount': [5.0, np.nan]})
    result = DatasetBuilder.finalize_feature_frame(frame)
    assert result['net_amount'].tolist() == [5.0, 0.0]


def test_model_search_prefers_stable_net_return():
    rows = []
    for fold in [1, 2]:
        rows.extend([
            {
                'model': 'ridge', 'holding_days': 5, 'pool_limit': 1000,
                'train_months': 24, 'fold': fold, 'rank_ic': 0.01,
                'top_10_net_return': 0.002, 'selection_score': 0.0018,
            },
            {
                'model': 'xgb', 'holding_days': 5, 'pool_limit': 1000,
                'train_months': 24, 'fold': fold, 'rank_ic': 0.02,
                'top_10_net_return': -0.001, 'selection_score': -0.0012,
            },
        ])
    summary = NetReturnModelSearch.summarize(rows)
    assert summary.iloc[0]['model'] == 'ridge'
    assert bool(summary.iloc[0]['accepted'])


def test_daily_ic_feature_selection():
    rng = np.random.default_rng(42)
    dates = pd.date_range('2025-01-01', periods=20, freq='B').strftime('%Y%m%d')
    rows = []
    for trade_date in dates:
        target = np.linspace(-1, 1, 30)
        noise = rng.normal(size=30)
        for idx in range(30):
            rows.append({
                'trade_date': trade_date,
                'target': target[idx],
                'good': target[idx] + rng.normal(scale=0.01),
                'noise': noise[idx],
            })
    frame = pd.DataFrame(rows)
    old_values = (
        config.FEATURE_SELECTION_MAX_FEATURES,
        config.FEATURE_SELECTION_MIN_ABS_IC,
        config.FEATURE_SELECTION_SAMPLE_SIZE,
    )
    try:
        config.FEATURE_SELECTION_MAX_FEATURES = 1
        config.FEATURE_SELECTION_MIN_ABS_IC = 0.0
        config.FEATURE_SELECTION_SAMPLE_SIZE = 0
        selected = DatasetBuilder().select_feature_columns(
            frame, ['good', 'noise']
        )
    finally:
        (
            config.FEATURE_SELECTION_MAX_FEATURES,
            config.FEATURE_SELECTION_MIN_ABS_IC,
            config.FEATURE_SELECTION_SAMPLE_SIZE,
        ) = old_values
    assert selected == ['good']


def test_top_list_indicator_ignores_sparse_zero_fill():
    engineer = FeatureEngineer()
    frame = pd.DataFrame({
        'net_amount': [np.nan, 0.0, 12.0, 0.0],
        'l_amount': [np.nan, 0.0, 0.0, 30.0],
        'amount': [100.0] * 4,
    })
    result = engineer.add_top_list_features(frame)
    assert result['is_top_list'].tolist() == [0, 0, 1, 1]


def test_sector_selection_always_fills_top_n():
    predictor = object.__new__(StockPredictor)
    frame = pd.DataFrame({
        'ts_code': [f'{idx:06d}.SZ' for idx in range(12)],
        'model_score': np.arange(12, 0, -1, dtype=float),
        'sector': ['同一行业'] * 12,
        'market_cap': np.arange(12, 0, -1, dtype=float),
    })
    selected = predictor.rank_and_select(frame, top_n=10, sector_neutral=True)
    assert len(selected) == 10
    assert np.isclose(selected['weight'].sum(), 1.0)


def test_model_metadata_is_required_and_hashed():
    feature_cols = ['feature_a', 'feature_b']
    meta = {
        'model_type': 'dl',
        'data_start_date': '20200101',
        'data_end_date': '20251231',
        'label_entry_shift': config.LABEL_ENTRY_SHIFT,
        'label_exit_shift': config.LABEL_EXIT_SHIFT,
        'label_entry_price': 'open',
        'label_exit_price': 'close',
        'feature_count': len(feature_cols),
        'feature_hash': hashlib.sha256(
            '\n'.join(feature_cols).encode('utf-8')
        ).hexdigest(),
    }
    predictor = object.__new__(StockPredictor)
    predictor.model_meta = {}
    with tempfile.TemporaryDirectory() as temp_dir:
        meta_file = f'{temp_dir}/model_meta.json'
        with open(meta_file, 'w', encoding='utf-8') as file:
            json.dump(meta, file)
        predictor._load_and_validate_meta(meta_file, feature_cols, 'dl')
        assert predictor.model_meta['feature_hash'] == meta['feature_hash']

        meta['label_entry_price'] = 'close'
        with open(meta_file, 'w', encoding='utf-8') as file:
            json.dump(meta, file)
        try:
            predictor._load_and_validate_meta(meta_file, feature_cols, 'dl')
        except ValueError:
            pass
        else:
            raise AssertionError('旧收盘买入模型未被拒绝')

        meta['model_type'] = 'tabular'
        meta['label_entry_price'] = 'open'
        meta['label_exit_shift'] = config.LABEL_ENTRY_SHIFT + 10
        with open(meta_file, 'w', encoding='utf-8') as file:
            json.dump(meta, file)
        predictor._load_and_validate_meta(
            meta_file, feature_cols, expected_model_type='tabular'
        )
        assert predictor.model_meta['label_exit_shift'] == config.LABEL_ENTRY_SHIFT + 10


def _make_trade_bars(locked_exit=False):
    dates = ['20260102', '20260105', '20260106', '20260107']
    frame = pd.DataFrame({
        'trade_date': dates,
        'open': [5.0, 5.1, 5.2, 5.3],
        'close': [5.1, 5.2, 5.4, 5.5],
        'high': [5.2, 5.3, 5.5, 5.6],
        'low': [4.9, 5.0, 5.1, 5.2],
        'pct_chg': [1.0, 1.0, 1.0, 1.0],
        'raw_open': [10.0, 10.2, 10.4, 10.6],
        'raw_close': [10.2, 10.4, 10.8, 11.0],
        'raw_high': [10.4, 10.6, 11.0, 11.2],
        'raw_low': [9.8, 10.0, 10.2, 10.4],
        'raw_pct_chg': [1.0, 1.0, 1.0, 1.0],
        'amount': [100000.0] * 4,
    })
    if locked_exit:
        mask = frame['trade_date'] >= '20260106'
        frame.loc[mask, 'raw_high'] = 8.0
        frame.loc[mask, 'raw_low'] = 8.0
        frame.loc[mask, 'raw_pct_chg'] = -10.0
    return frame


def test_trade_simulation_keeps_unclosed_capital():
    engine = object.__new__(BacktestEngine)
    closed = engine._simulate_stock_trade(
        _make_trade_bars(), '20260102', '20260106', 100000.0
    )
    assert closed is not None and closed['closed']
    assert closed['cash_in'] > closed['cash_out']
    assert closed['exit_date'] == '20260106'

    unclosed = engine._simulate_stock_trade(
        _make_trade_bars(locked_exit=True),
        '20260102', '20260106', 100000.0
    )
    assert unclosed is not None and not unclosed['closed']
    assert unclosed['cash_in'] > 0


def main():
    tests = [
        test_latest_trade_date_waits_for_daily_publication,
        test_prediction_scaler_runs_once,
        test_purged_train_val_test_split,
        test_walkforward_split_keeps_purge_gap,
        test_date_grouped_batches_and_rank_loss,
        test_target_uses_next_open,
        test_cs_rank_target_is_centered_and_ordered,
        test_ranking_metrics_use_relative_direction_and_net_return,
        test_nonstationary_absolute_features_are_excluded,
        test_sparse_event_values_are_not_forward_filled,
        test_model_search_prefers_stable_net_return,
        test_daily_ic_feature_selection,
        test_top_list_indicator_ignores_sparse_zero_fill,
        test_sector_selection_always_fills_top_n,
        test_model_metadata_is_required_and_hashed,
        test_trade_simulation_keeps_unclosed_capital,
    ]
    for test in tests:
        test()
        print(f"[OK] {test.__name__}")


if __name__ == '__main__':
    main()
