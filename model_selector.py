"""
模型配置切换脚本
可以在不同模型架构之间快速切换
"""

import config


def set_simple_dnn():
    """配置为简单 DNN 模型（快速训练）"""
    print("切换到简单 DNN 模型")
    print("=" * 50)

    config.MODEL_TYPE = "dnn"
    config.SEQUENCE_LENGTH = 1  # 不使用时序窗口

    # 关闭 CNN、LSTM、Attention
    config.USE_CNN = False
    config.USE_LSTM = False
    config.USE_ATTENTION = False

    # 简单的全连接层
    config.DENSE_LAYERS = [256, 128, 64, 32]

    # 更快的训练
    config.BATCH_SIZE = 128
    config.EPOCHS = 50
    config.LEARNING_RATE = 0.001

    print("✓ 简单快速")
    print("✓ 适合快速验证")
    print("✓ 计算量小")
    print("✗ 无时序信息")
    print("=" * 50)


def set_cnn_only():
    """配置为纯 CNN 模型（提取局部模式）"""
    print("切换到 CNN 模型")
    print("=" * 50)

    config.MODEL_TYPE = "cnn"
    config.SEQUENCE_LENGTH = 20

    # 只使用 CNN
    config.USE_CNN = True
    config.USE_LSTM = False
    config.USE_ATTENTION = False

    config.CNN_FILTERS = [64, 128, 256]
    config.CNN_KERNEL_SIZES = [3, 5, 5]

    config.BATCH_SIZE = 64
    config.EPOCHS = 80

    print("✓ 提取局部模式")
    print("✓ 训练较快")
    print("✗ 无长期记忆")
    print("=" * 50)


def set_lstm_only():
    """配置为纯 LSTM 模型（捕捉时序依赖）"""
    print("切换到 LSTM 模型")
    print("=" * 50)

    config.MODEL_TYPE = "lstm"
    config.SEQUENCE_LENGTH = 20

    # 只使用 LSTM
    config.USE_CNN = False
    config.USE_LSTM = True
    config.USE_ATTENTION = False

    config.LSTM_UNITS = [128, 64]
    config.BIDIRECTIONAL = True

    config.BATCH_SIZE = 64
    config.EPOCHS = 80

    print("✓ 捕捉时序依赖")
    print("✓ 记忆历史信息")
    print("✗ 训练较慢")
    print("=" * 50)


def set_cnn_lstm():
    """配置为 CNN + LSTM 模型（局部特征 + 时序依赖）"""
    print("切换到 CNN + LSTM 模型")
    print("=" * 50)

    config.MODEL_TYPE = "cnn_lstm"
    config.SEQUENCE_LENGTH = 20

    # CNN + LSTM
    config.USE_CNN = True
    config.USE_LSTM = True
    config.USE_ATTENTION = False

    config.CNN_FILTERS = [64, 128, 128]
    config.LSTM_UNITS = [128, 64]
    config.BIDIRECTIONAL = True

    config.BATCH_SIZE = 64
    config.EPOCHS = 100

    print("✓ 局部模式 + 时序依赖")
    print("✓ 性能优秀")
    print("✗ 训练较慢")
    print("=" * 50)


def set_cnn_lstm_attention():
    """配置为 CNN + LSTM + Attention 完整模型（推荐）"""
    print("切换到 CNN + LSTM + Attention 模型（完整版）")
    print("=" * 50)

    config.MODEL_TYPE = "cnn_lstm_attention"
    config.SEQUENCE_LENGTH = 20

    # 全部启用
    config.USE_CNN = True
    config.USE_LSTM = True
    config.USE_ATTENTION = True

    config.CNN_FILTERS = [64, 128, 128]
    config.CNN_KERNEL_SIZES = [3, 3, 3]

    config.LSTM_UNITS = [128, 64]
    config.BIDIRECTIONAL = True
    config.RETURN_SEQUENCES = True  # 必须返回序列才能用 Attention

    config.NUM_ATTENTION_HEADS = 8
    config.ATTENTION_KEY_DIM = 32

    config.DENSE_LAYERS = [128, 64, 32]

    config.BATCH_SIZE = 64
    config.EPOCHS = 100
    config.LEARNING_RATE = 0.0005

    print("✓ 完整架构")
    print("✓ 性能最佳")
    print("✓ 自动学习特征权重")
    print("✗ 训练最慢")
    print("✗ 需要 GPU")
    print("=" * 50)


def set_lightweight():
    """配置为轻量级模型（快速 + 较好性能）"""
    print("切换到轻量级模型（平衡性能和速度）")
    print("=" * 50)

    config.MODEL_TYPE = "cnn_gru_attention"
    config.SEQUENCE_LENGTH = 15  # 更短的窗口

    config.USE_CNN = True
    config.USE_LSTM = False  # 使用 GRU（更快）
    config.USE_ATTENTION = True

    config.CNN_FILTERS = [64, 64]  # 更少的卷积层
    config.CNN_KERNEL_SIZES = [3, 3]

    config.LSTM_UNITS = [64]  # 更少的 RNN 层
    config.BIDIRECTIONAL = False  # 单向（更快）
    config.RETURN_SEQUENCES = True

    config.NUM_ATTENTION_HEADS = 4  # 更少的头
    config.ATTENTION_KEY_DIM = 16

    config.DENSE_LAYERS = [64, 32]

    config.BATCH_SIZE = 128
    config.EPOCHS = 80

    print("✓ 速度快")
    print("✓ 性能较好")
    print("✓ 内存占用小")
    print("✓ 适合无 GPU 环境")
    print("=" * 50)


def compare_models():
    """打印模型对比"""
    print("\n" + "=" * 80)
    print(" " * 30 + "模型架构对比")
    print("=" * 80)

    models = [
        {
            'name': '1. 简单 DNN',
            'speed': '⭐⭐⭐⭐⭐',
            'performance': '⭐⭐',
            'memory': '⭐⭐⭐⭐⭐',
            'gpu': '不需要',
            'time': '5 分钟',
            'best_for': '快速验证、小数据集'
        },
        {
            'name': '2. 纯 CNN',
            'speed': '⭐⭐⭐⭐',
            'performance': '⭐⭐⭐',
            'memory': '⭐⭐⭐⭐',
            'gpu': '推荐',
            'time': '15 分钟',
            'best_for': '局部模式识别'
        },
        {
            'name': '3. 纯 LSTM',
            'speed': '⭐⭐⭐',
            'performance': '⭐⭐⭐',
            'memory': '⭐⭐⭐',
            'gpu': '推荐',
            'time': '20 分钟',
            'best_for': '时序依赖捕捉'
        },
        {
            'name': '4. CNN + LSTM',
            'speed': '⭐⭐',
            'performance': '⭐⭐⭐⭐',
            'memory': '⭐⭐',
            'gpu': '强烈推荐',
            'time': '30 分钟',
            'best_for': '综合性能'
        },
        {
            'name': '5. CNN + LSTM + Attention（完整版）',
            'speed': '⭐',
            'performance': '⭐⭐⭐⭐⭐',
            'memory': '⭐',
            'gpu': '必须',
            'time': '45 分钟',
            'best_for': '追求最佳性能'
        },
        {
            'name': '6. 轻量级（CNN + GRU + Attention）',
            'speed': '⭐⭐⭐',
            'performance': '⭐⭐⭐⭐',
            'memory': '⭐⭐⭐',
            'gpu': '推荐',
            'time': '20 分钟',
            'best_for': '平衡性能和速度'
        }
    ]

    for model in models:
        print(f"\n{model['name']}")
        print("-" * 80)
        print(f"训练速度:    {model['speed']}")
        print(f"预测性能:    {model['performance']}")
        print(f"内存占用:    {model['memory']}")
        print(f"GPU 需求:    {model['gpu']}")
        print(f"训练时间:    {model['time']} (假设 3000 只股票 × 2 年数据)")
        print(f"最适合:      {model['best_for']}")

    print("\n" + "=" * 80)
    print("推荐选择:")
    print("  - 有 GPU + 追求性能 → 选择 5（完整版）")
    print("  - 有 GPU + 平衡考虑 → 选择 6（轻量级）")
    print("  - 无 GPU + 快速验证 → 选择 1（DNN）")
    print("  - 无 GPU + 较好性能 → 选择 6（轻量级，CPU 勉强可用）")
    print("=" * 80 + "\n")


def main():
    """主函数"""
    compare_models()

    print("请选择模型架构:")
    print("1 - 简单 DNN")
    print("2 - 纯 CNN")
    print("3 - 纯 LSTM")
    print("4 - CNN + LSTM")
    print("5 - CNN + LSTM + Attention（完整版，推荐）")
    print("6 - 轻量级（CNN + GRU + Attention）")

    choice = input("\n请输入选择 (1-6): ").strip()

    if choice == '1':
        set_simple_dnn()
    elif choice == '2':
        set_cnn_only()
    elif choice == '3':
        set_lstm_only()
    elif choice == '4':
        set_cnn_lstm()
    elif choice == '5':
        set_cnn_lstm_attention()
    elif choice == '6':
        set_lightweight()
    else:
        print("无效选择")
        return

    print("\n配置已更新！")
    print("现在可以运行: python main.py --mode train")


if __name__ == "__main__":
    main()
