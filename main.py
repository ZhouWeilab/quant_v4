"""
主程序入口
功能：整合所有模块，提供训练和预测功能
"""

import argparse
import sys
import os
from datetime import datetime
import config
from data_loader import DataLoader
from dataset import DatasetBuilder
from model import ModelTrainer
from predictor import StockPredictor, BacktestEngine


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
        trade_date = loader.get_latest_trade_date()
        print(f"\n数据截止日期: {trade_date}")

        # 获取股票列表
        stock_list = loader.get_stock_list(trade_date)
        
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
        X_train, y_train, X_val, y_val, feature_cols, _, _ = builder.prepare_train_data(stock_data)

        # 保存标准化器和特征列
        builder.save_scaler()

        # 训练模型
        print("\n" + "-" * 70)
        # input_shape = (sequence_length, n_features)
        input_shape = (config.SEQUENCE_LENGTH, len(feature_cols))
        model = trainer.train_model(
            X_train, y_train,
            X_val, y_val,
            input_shape=input_shape
        )

        print("\n" + "=" * 70)
        print(" " * 25 + "训练完成!")
        print("=" * 70)
        print(f"模型已保存至: {config.MODEL_FILE}")
        print(f"标准化器已保存至: {config.SCALER_FILE}")
        print(f"特征列已保存至: {config.FEATURE_COLS_FILE}")

    except Exception as e:
        print(f"\n错误: 训练失败 - {e}")
        import traceback
        traceback.print_exc()


def predict_stocks():
    """预测选股流程"""
    print("\n" + "=" * 70)
    print(" " * 25 + "股票预测模式")
    print("=" * 70)

    try:
        # 检查模型是否存在
        if not os.path.exists(config.MODEL_FILE):
            print("\n错误: 模型文件不存在，请先运行训练模式")
            print("命令: python main.py --mode train")
            return

        # 创建预测器
        predictor = StockPredictor()

        # 预测并选股
        recommendations = predictor.predict_and_select(top_n=config.TOP_N_STOCKS)

        # 输出结果
        if not recommendations.empty:
            print("\n" + "=" * 70)
            print(f"{' ' * 20}今日推荐买入股票 (Top {len(recommendations)})")
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


def run_backtest():
    """运行回测流程"""
    print("\n" + "=" * 70)
    print(" " * 25 + "回测模式")
    print("=" * 70)

    try:
        # 检查模型是否存在
        if not os.path.exists(config.MODEL_FILE):
            print("\n错误: 模型文件不存在，请先运行训练模式")
            print("命令: python main.py --mode train")
            return

        # 创建回测引擎
        engine = BacktestEngine()

        # 运行回测
        results = engine.run_simple_backtest(
            start_date=config.BACKTEST_START_DATE
        )

        # 输出回测结果
        print("\n" + "=" * 70)
        print(" " * 25 + "回测结果统计")
        print("=" * 70)

        for key, value in results.items():
            if '率' in key or '比率' in key:
                if '胜率' in key:
                    print(f"{key:.<20} {value:.2%}")
                else:
                    print(f"{key:.<20} {value:.4f}")
            elif '天数' in key:
                print(f"{key:.<20} {int(value)}")
            else:
                print(f"{key:.<20} {value:.2%}")

        print("=" * 70)

    except Exception as e:
        print(f"\n错误: 回测失败 - {e}")
        import traceback
        traceback.print_exc()


def print_banner():
    """打印程序横幅"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║          A 股超短线量化交易系统 v4.0                          ║
    ║          Quantitative Trading System                          ║
    ║                                                               ║
    ║          基于深度学习的次日涨幅预测与智能选股                 ║
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

注意事项:
  1. 首次使用请先在 config.py 中配置 Tushare Token
  2. 首次运行请先执行训练模式: python main.py --mode train
  3. 训练完成后即可使用预测模式获取每日推荐股票
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'predict', 'backtest'],
        default='predict',
        help='运行模式: train=训练模型, predict=预测选股, backtest=回测 (默认: predict)'
    )

    args = parser.parse_args()

    # 检查 Token
    if not config.TUSHARE_TOKEN:
        print("\n" + "!" * 70)
        print("错误: 未设置 Tushare Token")
        print("请在 config.py 文件中设置 TUSHARE_TOKEN")
        print("获取 Token: https://tushare.pro/register")
        print("!" * 70)
        sys.exit(1)

    # 根据模式执行
    if args.mode == 'train':
        train_model()
    elif args.mode == 'predict':
        predict_stocks()
    elif args.mode == 'backtest':
        run_backtest()


if __name__ == "__main__":
    main()
