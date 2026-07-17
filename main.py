"""
主程序入口
功能：整合所有模块，提供训练和预测功能
"""

import argparse
import sys
import os
import json
import hashlib
import numpy as np
from datetime import datetime
import config
from data_loader import DataLoader
from dataset import DatasetBuilder
from model import ModelTrainer
from predictor import StockPredictor, BacktestEngine


def apply_runtime_options(args):
    """应用命令行运行时配置，主要用于小样本验证。"""
    sample_count = args.sample_size if args.sample_size is not None else None

    if args.quick:
        sample_count = sample_count or config.SAMPLE_STOCK_COUNT
        config.START_DATE = args.start_date or config.QUICK_START_DATE
        config.EPOCHS = min(config.EPOCHS, config.QUICK_EPOCHS)
        config.INCLUDE_MONEYFLOW = False
        config.INCLUDE_TOP_LIST = False
        config.INCLUDE_LIMIT_LIST = False
        config.XGB_PARAMS_BASE['n_estimators'] = min(
            config.XGB_PARAMS_BASE.get('n_estimators', config.QUICK_XGB_N_ESTIMATORS),
            config.QUICK_XGB_N_ESTIMATORS
        )
        if args.mode == 'optimize':
            config.MODEL_SEARCH_TRAIN_MONTHS = [6]
            config.MODEL_SEARCH_POOL_LIMITS = [sample_count]
            config.MODEL_SEARCH_HOLDING_DAYS = [5]
            config.MODEL_SEARCH_MAX_FOLDS = 1
            config.MODEL_SEARCH_STAGE2_CONFIGS = 1
            config.MODEL_SEARCH_MLP_MAX_SAMPLES = 50000
            config.MODEL_SEARCH_PROMOTE_PRODUCTION = False
            config.ROLL_TEST_MONTHS = 2
            config.ROLL_STEP_MONTHS = 2
        print(f"小样本模式: 股票数={sample_count}, 起始日期={config.START_DATE}, XGB树数={config.XGB_PARAMS_BASE['n_estimators']}")

    if args.start_date and not args.quick:
        config.START_DATE = args.start_date
    if args.end_date:
        config.RUNTIME_END_DATE = args.end_date
    if args.no_advanced_data:
        config.INCLUDE_MONEYFLOW = False
        config.INCLUDE_TOP_LIST = False
        config.INCLUDE_LIMIT_LIST = False

    config.RUNTIME_SAMPLE_STOCK_COUNT = sample_count


def get_effective_trade_date(loader):
    """获取本次运行的数据截止日。"""
    return getattr(config, 'RUNTIME_END_DATE', None) or loader.get_latest_trade_date()


def get_training_artifact_path(path):
    """生成与正式产物同目录的训练临时路径。"""
    root, ext = os.path.splitext(path)
    return f"{root}.training{ext}"


def validate_training_artifacts(model, builder, feature_cols):
    """发布前校验模型、标准化器和特征清单。"""
    builder._validate_scaler_features(feature_cols)
    model_features = model.model.input_shape[-1]
    if model_features != len(feature_cols):
        raise ValueError(
            f"训练产物维度不一致: "
            f"model={model_features}, feature_cols={len(feature_cols)}"
        )


def limit_stock_sample(stock_list):
    """小样本模式下限制股票数量，降低本地验证成本。"""
    sample_count = getattr(config, 'RUNTIME_SAMPLE_STOCK_COUNT', None)
    if sample_count:
        print(f"小样本模式：仅使用前 {sample_count} 只候选股票")
        return stock_list.head(sample_count).reset_index(drop=True)
    return stock_list


def train_model():
    """训练模型流程"""
    print("\n" + "=" * 70)
    print(" " * 25 + "模型训练模式")
    print("=" * 70)

    try:
        # 初始化
        loader = DataLoader()
        builder = DatasetBuilder()
        trainer = ModelTrainer()

        # 获取最近交易日
        trade_date = get_effective_trade_date(loader)
        print(f"\n数据截止日期: {trade_date}")

        # 获取股票列表
        stock_list = loader.get_stock_list(
            trade_date,
            historical=getattr(config, 'USE_HISTORICAL_UNIVERSE', False)
        )
        stock_list = limit_stock_sample(stock_list)
        
        print(f"符合条件的股票数量: {len(stock_list)} 只")

        # 获取历史数据
        print(f"\n开始获取历史数据 (起始日期: {config.START_DATE})...")
        stock_data = loader.get_all_stocks_daily(
            stock_list,
            start_date=config.START_DATE,
            end_date=trade_date
        )

        if not stock_data:
            print("错误: 未获取到有效数据")
            return

        print(f"成功获取 {len(stock_data)} 只股票的历史数据")

        # 准备训练数据（包含时序窗口构建）
        print("\n" + "-" * 70)
        (
            X_train, y_train, X_val, y_val, X_test, y_test,
            feature_cols, meta_train, meta_val, meta_test
        ) = builder.prepare_train_data(stock_data)

        staging_model_file = get_training_artifact_path(config.MODEL_FILE)
        staging_scaler_file = get_training_artifact_path(config.SCALER_FILE)
        staging_feature_file = get_training_artifact_path(config.FEATURE_COLS_FILE)
        staging_meta_file = get_training_artifact_path(config.MODEL_META_FILE)
        for path in (
            staging_model_file, staging_scaler_file,
            staging_feature_file, staging_meta_file
        ):
            if os.path.exists(path):
                os.remove(path)

        # 训练模型
        print("\n" + "-" * 70)
        # input_shape = (sequence_length, n_features)
        input_shape = (config.SEQUENCE_LENGTH, len(feature_cols))
        model = trainer.train_model(
            X_train, y_train,
            X_val, y_val,
            input_shape=input_shape,
            meta_train=meta_train,
            meta_val=meta_val,
            X_test=X_test,
            y_test=y_test,
            meta_test=meta_test,
            model_file=staging_model_file
        )

        builder.save_scaler(staging_scaler_file, staging_feature_file)
        staged_builder = DatasetBuilder()
        staged_builder.load_scaler(staging_scaler_file, staging_feature_file)
        model.load(staging_model_file)
        validate_training_artifacts(model, staged_builder, staged_builder.feature_cols)

        def _json_metrics(metrics):
            result = {}
            for key, value in (metrics or {}).items():
                if isinstance(value, (list, tuple)):
                    result[key] = [float(item) for item in value]
                else:
                    result[key] = float(value)
            return result

        model_meta = {
            'model_type': 'dl',
            'trained_at': datetime.now().isoformat(timespec='seconds'),
            'data_start_date': str(config.START_DATE),
            'data_end_date': str(trade_date),
            'label_entry_shift': int(config.LABEL_ENTRY_SHIFT),
            'label_exit_shift': int(config.LABEL_EXIT_SHIFT),
            'label_entry_price': 'open',
            'label_exit_price': 'close',
            'target_normalization': config.TARGET_NORMALIZATION,
            'feature_count': len(feature_cols),
            'feature_hash': hashlib.sha256(
                '\n'.join(feature_cols).encode('utf-8')
            ).hexdigest(),
            'use_adjusted_prices': bool(config.USE_ADJUSTED_PRICES),
            'historical_universe': bool(config.USE_HISTORICAL_UNIVERSE),
            'production_refit_epochs': int(config.PRODUCTION_REFIT_EPOCHS),
            'validation_metrics': _json_metrics(
                getattr(model, 'validation_metrics', None)
                or getattr(trainer.model, 'validation_metrics', {})
            ),
            'test_metrics': _json_metrics(
                getattr(model, 'test_metrics', None)
                or getattr(trainer.model, 'test_metrics', {})
            ),
        }
        with open(staging_meta_file, 'w', encoding='utf-8') as file:
            json.dump(model_meta, file, ensure_ascii=False, indent=2)

        os.replace(staging_scaler_file, config.SCALER_FILE)
        os.replace(staging_feature_file, config.FEATURE_COLS_FILE)
        os.replace(staging_model_file, config.MODEL_FILE)
        os.replace(staging_meta_file, config.MODEL_META_FILE)

        print("\n" + "=" * 70)
        print(" " * 25 + "训练完成!")
        print("=" * 70)
        print(f"模型已保存至: {config.MODEL_FILE}")
        print(f"标准化器已保存至: {config.SCALER_FILE}")
        print(f"特征列已保存至: {config.FEATURE_COLS_FILE}")
        print(f"模型元数据已保存至: {config.MODEL_META_FILE}")
        return True

    except Exception as e:
        print(f"\n错误: 训练失败 - {e}")
        import traceback
        traceback.print_exc()
        return False


def predict_stocks(model_type='dl', trade_date=None):
    """预测选股流程"""
    print("\n" + "=" * 70)
    print(" " * 25 + "股票预测模式")
    print("=" * 70)

    try:
        # 检查模型是否存在
        if model_type == 'dl':
            model_file = config.MODEL_FILE
        elif model_type == 'xgb':
            model_file = config.MODEL_FILE.replace('.h5', '_xgb.pkl')
        else:
            model_file = config.TABULAR_MODEL_FILE
        if not os.path.exists(model_file):
            print(f"\n错误: {model_type.upper()} 模型文件不存在: {model_file}")
            print("请先运行对应训练模式:")
            if model_type == 'dl':
                print("  python main.py --mode train")
            else:
                print("  python main.py --mode baseline")
            return

        # 创建预测器
        predictor = StockPredictor(model_type=model_type)

        # 预测并选股
        recommendations = predictor.predict_and_select(
            trade_date=trade_date or getattr(config, 'RUNTIME_END_DATE', None),
            top_n=config.TOP_N_STOCKS
        )

        # 输出结果
        if not recommendations.empty:
            print("\n" + "=" * 70)
            print(f"{' ' * 15}T日买入持有期推荐 (Top {len(recommendations)})")
            print("=" * 70)
            print("\n" + recommendations.to_string(index=False))
            print("\n" + "=" * 70)

            # 保存到文件
            output_file = f"./recommendations_{datetime.now().strftime('%Y%m%d')}.csv"
            recommendations.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\n推荐结果已保存至: {output_file}")

        else:
            print("\n今日没有符合条件的推荐股票")

    except Exception as e:
        print(f"\n错误: 预测失败 - {e}")
        import traceback
        traceback.print_exc()
        raise


def run_backtest(model_type='dl'):
    """运行回测流程"""
    print("\n" + "=" * 70)
    print(" " * 25 + "回测模式")
    print("=" * 70)

    try:
        # 检查模型是否存在
        if model_type == 'dl':
            model_file = config.MODEL_FILE
        elif model_type == 'xgb':
            model_file = config.MODEL_FILE.replace('.h5', '_xgb.pkl')
        else:
            model_file = config.TABULAR_MODEL_FILE
        if not os.path.exists(model_file):
            print(f"\n错误: {model_type.upper()} 模型文件不存在: {model_file}")
            print("请先运行对应训练模式")
            return

        # 创建回测引擎
        engine = BacktestEngine(model_type=model_type)

        # 运行回测
        results = engine.run_simple_backtest(
            start_date=config.BACKTEST_START_DATE
        )

        # 输出回测结果
        print("\n" + "=" * 70)
        print(" " * 25 + "回测结果统计")
        print("=" * 70)

        for key, value in results.items():
            if isinstance(value, bool):
                print(f"{key:.<20} {'是' if value else '否'}")
            elif any(word in key for word in ['次数', '股票数', '延迟卖出数']):
                print(f"{key:.<20} {int(value)}")
            elif key == '夏普比率':
                print(f"{key:.<20} {value:.4f}")
            elif '收益率' in key or key in ['胜率', '最大回撤']:
                print(f"{key:.<20} {value:.2%}")
            else:
                print(f"{key:.<20} {value}")

        print("=" * 70)

    except Exception as e:
        print(f"\n错误: 回测失败 - {e}")
        import traceback
        traceback.print_exc()


def run_factor_test():
    """因子检验模式"""
    print("\n" + "=" * 70)
    print(" " * 25 + "因子检验模式")
    print("=" * 70)

    try:
        from factor_analysis import FactorAnalyzer
        from factor_preprocessing import FactorPreprocessor

        loader = DataLoader()
        builder = DatasetBuilder()
        analyzer = FactorAnalyzer()
        preprocessor = FactorPreprocessor()

        trade_date = get_effective_trade_date(loader)
        stock_list = loader.get_stock_list(trade_date, historical=True)
        stock_list = limit_stock_sample(stock_list)

        print(f"\n获取历史数据用于因子检验 ({config.START_DATE} ~ {trade_date})...")
        stock_data = loader.get_all_stocks_daily(
            stock_list,
            start_date=config.START_DATE,
            end_date=trade_date
        )

        if not stock_data:
            print("错误: 未获取到有效数据")
            return

        # 构建截面数据（用于 XGBoost 的格式）
        dataset = builder.prepare_tabular_data(stock_data)
        feature_cols = builder.get_feature_columns(dataset)

        print(f"\n原始因子数: {len(feature_cols)}")
        print(f"总样本数: {len(dataset)}")

        # 因子预处理（用于检验中性化前后的效果）
        if 'sector' in dataset.columns and 'market_cap' in dataset.columns:
            print("\n执行因子预处理（去极值 + 标准化 + 中性化）...")
            dataset_processed, final_cols = preprocessor.preprocess_pipeline(
                dataset, feature_cols,
                industry_col='sector', market_cap_col='market_cap'
            )
            # 同时检验原始因子和中性化后因子
            print("\n--- 原始因子检验 ---")
            results_raw = analyzer.full_analysis(dataset, feature_cols)
            analyzer.print_ic_report(results_raw['ic_report'], top_n=20)

            print("\n--- 中性化后因子检验 ---")
            results_neu = analyzer.full_analysis(dataset_processed, final_cols)
            analyzer.print_ic_report(results_neu['ic_report'], top_n=20)
        else:
            print("\n缺少行业/市值数据，仅检验原始因子...")
            results = analyzer.full_analysis(dataset, feature_cols)
            analyzer.print_ic_report(results['ic_report'], top_n=20)

        # 保存报告
        report_file = f"./factor_report_{trade_date}.csv"
        if 'results_raw' in dir():
            results_raw['ic_report'].to_csv(report_file, index=False, encoding='utf-8-sig')
        elif 'results' in dir():
            results['ic_report'].to_csv(report_file, index=False, encoding='utf-8-sig')
        print(f"\n因子检验报告已保存: {report_file}")

    except Exception as e:
        print(f"\n错误: 因子检验失败 - {e}")
        import traceback
        traceback.print_exc()


def run_baseline():
    """训练基准模型（XGBoost + 线性模型）"""
    print("\n" + "=" * 70)
    print(" " * 25 + "基准模型训练模式")
    print("=" * 70)

    try:
        from baseline_models import train_and_compare

        loader = DataLoader()
        builder = DatasetBuilder()

        trade_date = get_effective_trade_date(loader)
        stock_list = loader.get_stock_list(trade_date)
        stock_list = limit_stock_sample(stock_list)

        print(f"\n获取历史数据 ({config.START_DATE} ~ {trade_date})...")
        stock_data = loader.get_all_stocks_daily(
            stock_list,
            start_date=config.START_DATE,
            end_date=trade_date
        )

        if not stock_data:
            print("错误: 未获取到有效数据")
            return

        # 构建截面数据
        stock_meta = {
            row['ts_code']: {'sector': row.get('sector', config.DEFAULT_SECTOR),
                             'market_cap': row.get('market_cap', np.nan)}
            for _, row in stock_list.iterrows()
        }
        dataset = builder.prepare_tabular_data(stock_data, stock_meta)
        feature_cols = builder.get_feature_columns(dataset)

        # 生产预测链路需要训练/预测特征完全一致。
        # 中性化因子暂放在 factor_test 做研究，待预处理器可持久化后再接入预测。

        dataset[feature_cols] = (
            dataset[feature_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

        # 数据划分（一次性时序划分，非滚动）
        train_df, test_df = builder.split_dataset(dataset)
        # 训练集内再划分验证集，并在边界清除重叠标签。
        train_df, val_df = builder.split_dataset(
            train_df, val_split=config.ROLL_VAL_RATIO
        )
        feature_cols = builder.select_feature_columns(train_df, feature_cols)

        X_train = train_df[feature_cols].values
        y_train = train_df['target'].values
        X_val = val_df[feature_cols].values
        y_val = val_df['target'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['target'].values

        print(f"\n训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
        print(f"特征数: {len(feature_cols)}")

        # 训练并对比
        trained_models, comparator = train_and_compare(
            X_train, y_train, X_val, y_val, X_test, y_test,
            feature_cols, models_to_train=['xgb', 'ridge', 'logistic'],
            test_trade_dates=test_df['trade_date'].values,
            test_raw_returns=test_df[
                'target_raw' if 'target_raw' in test_df.columns else 'target_return'
            ].values
        )

        # 保存最佳模型（回归优先按扣费后 Top N 收益，分类按 AUC）
        comp_df = comparator.compare()
        target_type = getattr(config, 'TARGET_TYPE', 'classification')
        selection_col = f"top_{int(getattr(config, 'SELECTION_TOP_N', 10))}_net_return"
        if (
            target_type == 'regression'
            and selection_col in comp_df.columns
            and comp_df[selection_col].notna().any()
        ):
            best_model_name = comp_df[selection_col].idxmax()
            print(
                f"\n最佳模型: {best_model_name}，"
                f"{selection_col} = {comp_df.loc[best_model_name, selection_col]:.4f}，"
                "正在保存..."
            )
        elif target_type == 'regression' and 'rank_ic' in comp_df.columns and comp_df['rank_ic'].notna().any():
            best_model_name = comp_df['rank_ic'].idxmax()
            print(f"\n最佳模型: {best_model_name}，Rank IC = {comp_df.loc[best_model_name, 'rank_ic']:.4f}，正在保存...")
        elif target_type == 'regression' and 'long_short_return' in comp_df.columns and comp_df['long_short_return'].notna().any():
            best_model_name = comp_df['long_short_return'].idxmax()
            print(f"\n最佳模型: {best_model_name}，多空收益 = {comp_df.loc[best_model_name, 'long_short_return']:.4f}，正在保存...")
        else:
            best_model_name = comp_df['auc'].idxmax()
            print(f"\n最佳模型: {best_model_name}，AUC = {comp_df.loc[best_model_name, 'auc']:.4f}，正在保存...")

        if 'xgb' in trained_models:
            trained_models['xgb'].save()
            print("生产预测模型使用 XGBoost")
        elif best_model_name == 'Ridge' and 'ridge' in trained_models:
            trained_models['ridge'].save()
        elif best_model_name == 'Logistic' and 'logistic' in trained_models:
            trained_models['logistic'].save()

        # 同时保存特征列（供预测时对齐）
        import pickle
        xgb_feature_file = config.FEATURE_COLS_FILE.replace('.pkl', '_xgb.pkl')
        with open(xgb_feature_file, 'wb') as f:
            pickle.dump(feature_cols, f)
        print(f"特征列已保存: {xgb_feature_file}")

        xgb_meta = {
            'model_type': 'xgb',
            'trained_at': datetime.now().isoformat(timespec='seconds'),
            'data_start_date': str(config.START_DATE),
            'data_end_date': str(trade_date),
            'label_entry_shift': int(config.LABEL_ENTRY_SHIFT),
            'label_exit_shift': int(config.LABEL_EXIT_SHIFT),
            'label_entry_price': 'open',
            'label_exit_price': 'close',
            'target_normalization': config.TARGET_NORMALIZATION,
            'feature_count': len(feature_cols),
            'feature_hash': hashlib.sha256(
                '\n'.join(feature_cols).encode('utf-8')
            ).hexdigest(),
            'use_adjusted_prices': bool(config.USE_ADJUSTED_PRICES),
            'historical_universe': bool(config.USE_HISTORICAL_UNIVERSE),
        }
        with open(config.XGB_MODEL_META_FILE, 'w', encoding='utf-8') as file:
            json.dump(xgb_meta, file, ensure_ascii=False, indent=2)
        print(f"XGBoost 模型元数据已保存: {config.XGB_MODEL_META_FILE}")

        print("\n" + "=" * 70)
        print(" " * 25 + "基准模型训练完成")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"\n错误: 基准模型训练失败 - {e}")
        import traceback
        traceback.print_exc()
        return False


def run_weekly_rebalance(model_type='dl', force=False, skip_train=False):
    """周末调仓：训练最新模型并输出下一周持仓建议。"""
    today = datetime.now()
    if today.weekday() not in config.WEEKLY_REBALANCE_WEEKDAYS and not force:
        print("当前不是周末，已跳过调仓。需要手动运行可加 --force。")
        return

    print("\n" + "=" * 70)
    print(" " * 25 + "周末调仓模式")
    print("=" * 70)

    if not skip_train:
        if model_type == 'dl':
            if not train_model():
                print("训练失败，停止本次调仓。")
                return
        else:
            if not run_baseline():
                print("训练失败，停止本次调仓。")
                return

    predict_stocks(model_type=model_type)


def run_walkforward():
    """运行严格滚动样本外评估。"""
    from walkforward import WalkForwardEvaluator

    report = WalkForwardEvaluator().run(
        end_date=getattr(config, 'RUNTIME_END_DATE', None)
    )
    print(report.to_string(index=False))


def run_model_search():
    """运行收益导向的模型、训练窗、股票池和持有期搜索。"""
    from model_search import NetReturnModelSearch

    summary = NetReturnModelSearch(
        end_date=getattr(config, 'RUNTIME_END_DATE', None)
    ).run()
    print(summary.to_string(index=False))


def print_banner():
    """打印程序横幅"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║          A 股超短线量化交易系统 v4.0                          ║
    ║          Quantitative Trading System                          ║
    ║                                                               ║
    ║          T日收盘信号、T+1开盘买入、T+6收盘卖出选股       ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """主函数"""
    # 打印横幅
    print_banner()

    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='A 股超短线量化交易系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py --mode train      # 训练模型
  python main.py --mode predict    # 预测选股（默认）
  python main.py --mode backtest   # 运行回测
  python main.py --mode weekly_rebalance --quick --force  # 小样本周末调仓验证

注意事项:
  1. 首次使用请配置环境变量 TUSHARE_TOKEN 或用户私有 Token 文件
  2. 首次运行请先执行训练模式: python main.py --mode train
  3. 训练完成后即可使用预测模式获取每日推荐股票
  4. 本策略为"T日收盘生成信号，T+1开盘买入，T+6收盘卖出"
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=[
            'train', 'predict', 'backtest', 'factor_test', 'baseline',
            'walkforward', 'optimize', 'weekly_rebalance'
        ],
        default='predict',
        help='运行模式: train=训练DL模型, predict=预测选股, backtest=成交模拟, factor_test=因子检验, baseline=训练基准模型, walkforward=滚动样本外评估, weekly_rebalance=周末调仓 (默认: predict)'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['dl', 'xgb', 'tabular'],
        default='dl',
        help='预测/回测使用的模型: dl=深度学习, xgb=XGBoost (默认: dl)'
    )
    parser.add_argument('--quick', action='store_true', help='启用小样本快速验证配置')
    parser.add_argument('--sample-size', type=int, default=None, help='限制候选股票数量，用于小样本验证')
    parser.add_argument('--start-date', type=str, default=None, help='覆盖训练数据起始日期，格式 YYYYMMDD')
    parser.add_argument('--end-date', type=str, default=None, help='覆盖数据截止日期，格式 YYYYMMDD')
    parser.add_argument('--no-advanced-data', action='store_true', help='跳过资金流、龙虎榜、涨跌停等高阶数据拉取')
    parser.add_argument('--force', action='store_true', help='非周末也强制执行 weekly_rebalance')
    parser.add_argument('--skip-train', action='store_true', help='weekly_rebalance 中跳过训练，直接用已有模型预测')

    args = parser.parse_args()
    apply_runtime_options(args)

    # 检查 Token
    if not config.TUSHARE_TOKEN:
        print("\n" + "!" * 70)
        print("错误: 未设置 Tushare Token")
        print("请设置环境变量 TUSHARE_TOKEN，或写入 ~/.config/quant_v4/tushare_token")
        print("获取 Token: https://tushare.pro/register")
        print("!" * 70)
        sys.exit(1)

    # 根据模式执行
    if args.mode == 'train':
        train_model()
    elif args.mode == 'predict':
        predict_stocks(model_type=args.model)
    elif args.mode == 'backtest':
        run_backtest(model_type=args.model)
    elif args.mode == 'factor_test':
        run_factor_test()
    elif args.mode == 'baseline':
        run_baseline()
    elif args.mode == 'walkforward':
        run_walkforward()
    elif args.mode == 'optimize':
        run_model_search()
    elif args.mode == 'weekly_rebalance':
        run_weekly_rebalance(model_type=args.model, force=args.force, skip_train=args.skip_train)


if __name__ == "__main__":
    main()
