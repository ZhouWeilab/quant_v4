"""
模型架构测试脚本
快速验证模型是否正确构建（不需要真实数据）
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import config
from model import QuantModel, MultiHeadSelfAttention


def test_model_build():
    """测试模型构建"""
    print("=" * 70)
    print(" " * 20 + "测试模型构建")
    print("=" * 70)

    # 模拟输入形状
    sequence_length = config.SEQUENCE_LENGTH
    n_features = 50  # 假设有50个特征

    print(f"\n输入形状: ({sequence_length}, {n_features})")
    print(f"批次大小: {config.BATCH_SIZE}")

    # 创建模型
    try:
        model = QuantModel(input_shape=(sequence_length, n_features))
        model.build_model()
        model.compile_model()

        print("\n✓ 模型构建成功")

        # 打印模型结构
        print("\n模型结构:")
        print("-" * 70)
        model.summary()

        return model

    except Exception as e:
        print(f"\n✗ 模型构建失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_forward_pass(model):
    """测试前向传播"""
    print("\n" + "=" * 70)
    print(" " * 20 + "测试前向传播")
    print("=" * 70)

    # 创建随机输入
    batch_size = config.BATCH_SIZE
    sequence_length = config.SEQUENCE_LENGTH
    n_features = 50

    X_dummy = np.random.randn(batch_size, sequence_length, n_features).astype(np.float32)
    y_dummy = np.random.randn(batch_size).astype(np.float32)

    print(f"\n输入数据: X={X_dummy.shape}, y={y_dummy.shape}")

    try:
        # 前向传播
        predictions = model.predict(X_dummy)

        print(f"输出数据: predictions={predictions.shape}")
        print(f"预测样例: {predictions[:5].flatten()}")

        print("\n✓ 前向传播成功")
        return True

    except Exception as e:
        print(f"\n✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step(model):
    """测试训练步骤"""
    print("\n" + "=" * 70)
    print(" " * 20 + "测试训练步骤")
    print("=" * 70)

    # 创建小规模训练数据
    n_samples = 200
    sequence_length = config.SEQUENCE_LENGTH
    n_features = 50

    X_train = np.random.randn(n_samples, sequence_length, n_features).astype(np.float32)
    y_train = np.random.randn(n_samples).astype(np.float32)

    X_val = np.random.randn(50, sequence_length, n_features).astype(np.float32)
    y_val = np.random.randn(50).astype(np.float32)

    print(f"\n训练数据: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"验证数据: X_val={X_val.shape}, y_val={y_val.shape}")

    try:
        # 训练几个 epoch
        print("\n开始训练（只训练 3 个 epoch 测试）...")

        history = model.model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=3,
            validation_data=(X_val, y_val),
            verbose=1
        )

        print("\n✓ 训练步骤成功")
        print(f"训练历史: {list(history.history.keys())}")

        return True

    except Exception as e:
        print(f"\n✗ 训练步骤失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_save_load(model):
    """测试模型保存和加载"""
    print("\n" + "=" * 70)
    print(" " * 20 + "测试模型保存和加载")
    print("=" * 70)

    test_model_file = "./test_model.h5"

    try:
        # 保存模型
        print("\n保存模型...")
        model.save(test_model_file)
        print(f"✓ 模型已保存到: {test_model_file}")

        # 加载模型
        print("\n加载模型...")
        new_model = QuantModel()
        new_model.load(test_model_file)
        print("✓ 模型加载成功")

        # 测试加载的模型
        X_test = np.random.randn(10, config.SEQUENCE_LENGTH, 50).astype(np.float32)
        predictions = new_model.predict(X_test)
        print(f"✓ 加载的模型可以正常预测: {predictions.shape}")

        # 清理测试文件
        import os
        if os.path.exists(test_model_file):
            os.remove(test_model_file)
            print(f"✓ 清理测试文件")

        return True

    except Exception as e:
        print(f"\n✗ 保存/加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_custom_layers():
    """测试自定义层"""
    print("\n" + "=" * 70)
    print(" " * 20 + "测试自定义层")
    print("=" * 70)

    try:
        # 测试 MultiHeadSelfAttention
        print("\n测试 MultiHeadSelfAttention 层...")

        batch_size = 8
        seq_len = 20
        dim = 64

        # 创建输入
        inputs = tf.random.normal((batch_size, seq_len, dim))

        # 创建层
        attention = MultiHeadSelfAttention(
            num_heads=config.NUM_ATTENTION_HEADS,
            key_dim=config.ATTENTION_KEY_DIM
        )

        # 前向传播
        outputs = attention(inputs, training=False)

        print(f"输入形状: {inputs.shape}")
        print(f"输出形状: {outputs.shape}")
        print("✓ MultiHeadSelfAttention 层工作正常")

        return True

    except Exception as e:
        print(f"\n✗ 自定义层测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_model_config():
    """打印当前模型配置"""
    print("\n" + "=" * 70)
    print(" " * 20 + "当前模型配置")
    print("=" * 70)

    print(f"\n模型类型: {config.MODEL_TYPE}")
    print(f"时序窗口长度: {config.SEQUENCE_LENGTH}")

    print(f"\nCNN 配置:")
    print(f"  - 使用 CNN: {config.USE_CNN}")
    if config.USE_CNN:
        print(f"  - 卷积核数量: {config.CNN_FILTERS}")
        print(f"  - 卷积核大小: {config.CNN_KERNEL_SIZES}")

    print(f"\nRNN 配置:")
    print(f"  - 类型: {'LSTM' if config.USE_LSTM else 'GRU'}")
    print(f"  - 单元数: {config.LSTM_UNITS}")
    print(f"  - 双向: {config.BIDIRECTIONAL}")
    print(f"  - 返回序列: {config.RETURN_SEQUENCES}")

    print(f"\nAttention 配置:")
    print(f"  - 使用 Attention: {config.USE_ATTENTION}")
    if config.USE_ATTENTION:
        print(f"  - 注意力头数: {config.NUM_ATTENTION_HEADS}")
        print(f"  - 键维度: {config.ATTENTION_KEY_DIM}")

    print(f"\n全连接层: {config.DENSE_LAYERS}")
    print(f"Dropout 率: {config.DROPOUT_RATE}")

    print(f"\n训练配置:")
    print(f"  - Batch Size: {config.BATCH_SIZE}")
    print(f"  - Epochs: {config.EPOCHS}")
    print(f"  - Learning Rate: {config.LEARNING_RATE}")

    print(f"\n损失函数权重:")
    print(f"  - MSE: {config.LOSS_MSE_WEIGHT}")
    print(f"  - Direction: {config.LOSS_DIRECTION_WEIGHT}")
    print(f"  - Ranking: {config.LOSS_RANKING_WEIGHT}")


def main():
    """主测试流程"""
    print("\n" + "🔧" * 35)
    print(" " * 20 + "模型架构测试")
    print("🔧" * 35)

    # 打印配置
    print_model_config()

    # 测试自定义层
    test1 = test_custom_layers()

    # 测试模型构建
    model = test_model_build()

    if model is None:
        print("\n❌ 模型构建失败，跳过后续测试")
        return

    # 测试前向传播
    test2 = test_forward_pass(model)

    # 测试训练步骤
    test3 = test_training_step(model)

    # 测试保存和加载
    test4 = test_save_load(model)

    # 总结
    print("\n" + "=" * 70)
    print(" " * 25 + "测试总结")
    print("=" * 70)

    tests = [
        ("自定义层", test1),
        ("模型构建", model is not None),
        ("前向传播", test2),
        ("训练步骤", test3),
        ("保存加载", test4)
    ]

    all_passed = True
    for name, result in tests:
        status = "✓" if result else "✗"
        print(f"{status} {name}: {'通过' if result else '失败'}")
        if not result:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n🎉 所有测试通过！模型架构正确。")
        print("\n下一步:")
        print("  1. 在 config.py 中配置 Tushare Token")
        print("  2. 运行: python main.py --mode train")
    else:
        print("\n⚠️  部分测试失败，请检查模型配置。")

    print()


if __name__ == "__main__":
    main()
