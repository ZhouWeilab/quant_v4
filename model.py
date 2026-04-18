"""
模型定义与训练模块
功能：定义 CNN + LSTM/GRU + Multi-Head Attention 混合架构，训练、评估、保存模型
专为 A 股超短线量化交易设计（高噪声、微弱信号、非平稳、T+1约束）
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import config
import os


class MultiHeadSelfAttention(layers.Layer):
    """多头自注意力层"""

    def __init__(self, num_heads, key_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=config.DROPOUT_RATE
        )
        self.layernorm = layers.LayerNormalization()

    def call(self, inputs, training=False):
        # 自注意力
        attn_output = self.attention(
            query=inputs,
            key=inputs,
            value=inputs,
            training=training
        )
        # 残差连接 + 层归一化
        return self.layernorm(inputs + attn_output)

    def get_config(self):
        config_dict = super().get_config()
        config_dict.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim
        })
        return config_dict


class QuantModel:
    """量化交易深度学习模型（CNN + LSTM/GRU + Attention）"""

    def __init__(self, input_shape=None):
        """
        初始化模型

        Args:
            input_shape: 输入形状 (sequence_length, n_features)
        """
        self.input_shape = input_shape
        self.model = None
        self.history = None

        # 设置随机种子
        np.random.seed(config.RANDOM_SEED)
        tf.random.set_seed(config.RANDOM_SEED)

    def build_cnn_block(self, inputs):
        """
        构建 CNN 模块（提取局部特征模式）

        Args:
            inputs: 输入张量

        Returns:
            CNN 输出张量
        """
        x = inputs

        # 多层 1D 卷积
        for i, (filters, kernel_size) in enumerate(zip(config.CNN_FILTERS, config.CNN_KERNEL_SIZES)):
            x = layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding='same',
                activation=None,
                name=f'conv1d_{i+1}'
            )(x)

            # Batch Normalization
            x = layers.BatchNormalization(name=f'bn_cnn_{i+1}')(x)

            # 激活函数
            x = layers.Activation(config.ACTIVATION, name=f'act_cnn_{i+1}')(x)

            # Spatial Dropout（对卷积层更有效）
            x = layers.SpatialDropout1D(
                config.SPATIAL_DROPOUT_RATE,
                name=f'spatial_dropout_{i+1}'
            )(x)

            # 最后一层不使用池化（保留时序信息）
            if i < len(config.CNN_FILTERS) - 1 and config.CNN_POOL_SIZE > 1:
                x = layers.MaxPooling1D(
                    pool_size=config.CNN_POOL_SIZE,
                    name=f'maxpool_{i+1}'
                )(x)

        return x

    def build_rnn_block(self, inputs):
        x = inputs
        RNN = layers.LSTM if config.USE_LSTM else layers.GRU
        rnn_name = 'LSTM' if config.USE_LSTM else 'GRU'
        num_layers = len(config.LSTM_UNITS)
        for i, units in enumerate(config.LSTM_UNITS):
            # 判断是否返回序列
            if i == num_layers - 1:
                return_sequences = config.RETURN_SEQUENCES   # 最后一层根据配置
            else:
                return_sequences = True                       # 中间层必须返回序列
            
            if config.BIDIRECTIONAL:
                x = layers.Bidirectional(
                    RNN(units, return_sequences=return_sequences, 
                        dropout=config.DROPOUT_RATE, recurrent_dropout=0.1),
                    name=f'bi_{rnn_name}_{i+1}'
                )(x)
            else:
                x = RNN(units, return_sequences=return_sequences,
                        dropout=config.DROPOUT_RATE, recurrent_dropout=0.1,
                        name=f'{rnn_name}_{i+1}')(x)
            
            if return_sequences:
                x = layers.BatchNormalization(name=f'bn_rnn_{i+1}')(x)
        return x

    def build_attention_block(self, inputs):
        """
        构建多头自注意力模块

        Args:
            inputs: 输入张量

        Returns:
            Attention 输出张量
        """
        x = MultiHeadSelfAttention(
            num_heads=config.NUM_ATTENTION_HEADS,
            key_dim=config.ATTENTION_KEY_DIM,
            name='multi_head_attention'
        )(inputs)

        return x

    def build_model(self):
        """
        构建完整的 CNN + LSTM/GRU + Attention 模型

        Returns:
            Model: Keras 模型
        """
        if self.input_shape is None:
            raise ValueError("请先设置 input_shape")

        # 输入层：(sequence_length, n_features)
        inputs = layers.Input(shape=self.input_shape, name='input')

        # ========== CNN 模块 ==========
        if config.USE_CNN:
            x = self.build_cnn_block(inputs)
        else:
            x = inputs

        # ========== RNN 模块 ==========
        # 如果 RNN 不返回序列，需要先通过 RNN 再使用 Attention
        if config.RETURN_SEQUENCES and config.USE_ATTENTION:
            x = self.build_rnn_block(x)
            # ========== Attention 模块 ==========
            x = self.build_attention_block(x)
            # 全局平均池化
            x = layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
        else:
            x = self.build_rnn_block(x)

        # ========== 全连接层 ==========
        for i, units in enumerate(config.DENSE_LAYERS):
            x = layers.Dense(
                units,
                activation=None,
                kernel_initializer='he_normal',
                name=f'dense_{i+1}'
            )(x)

            # Batch Normalization
            x = layers.BatchNormalization(name=f'bn_dense_{i+1}')(x)

            # 激活函数
            x = layers.Activation(config.ACTIVATION, name=f'act_dense_{i+1}')(x)

            # Dropout
            x = layers.Dropout(config.DROPOUT_RATE, name=f'dropout_{i+1}')(x)

        # ========== 输出层 ==========
        outputs = layers.Dense(1, activation='linear', name='output')(x)

        # 构建模型
        model = models.Model(inputs=inputs, outputs=outputs, name='cnn_lstm_attention_quant_model')

        # 保存已构建模型实例，供后续编译/训练使用
        self.model = model

        return model

    def compile_model(self, model=None):
        """
        编译模型

        Args:
            model: Keras 模型，如果为 None 则使用 self.model
        """
        if model is None:
            model = self.model

        # 自定义损失函数：MSE + 方向性损失 + 排序损失
        def custom_loss(y_true, y_pred):
            # 1. MSE 损失（预测准确性）
            mse = tf.reduce_mean(tf.square(y_true - y_pred))

            # 2. 方向性损失（预测方向一致性）
            direction_loss = -tf.reduce_mean(
                tf.sign(y_true) * tf.sign(y_pred)
            )

            # 3. 排序损失（Ranking Loss，确保高收益股票排在前面）
            # 使用 Pairwise Ranking Loss
            y_true_expanded = tf.expand_dims(y_true, 1)
            y_pred_expanded = tf.expand_dims(y_pred, 1)

            # 计算真实值的差值
            true_diff = y_true_expanded - tf.transpose(y_true_expanded)

            # 计算预测值的差值
            pred_diff = y_pred_expanded - tf.transpose(y_pred_expanded)

            # 只考虑真实差值显著的样本对（避免噪声）
            significant_mask = tf.abs(true_diff) > 0.01

            # Ranking Loss：希望预测差值的符号与真实差值一致
            ranking_loss = tf.reduce_mean(
                tf.where(
                    significant_mask,
                    tf.nn.relu(1.0 - true_diff * pred_diff),  # Hinge loss
                    0.0
                )
            )

            # 组合损失
            total_loss = (
                config.LOSS_MSE_WEIGHT * mse +
                config.LOSS_DIRECTION_WEIGHT * direction_loss +
                config.LOSS_RANKING_WEIGHT * ranking_loss
            )

            return total_loss

        # 自定义指标：方向准确率
        def direction_accuracy(y_true, y_pred):
            return tf.reduce_mean(
                tf.cast(
                    tf.equal(tf.sign(y_true), tf.sign(y_pred)),
                    tf.float32
                )
            )

        # 编译
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            loss=custom_loss,
            metrics=[
                'mae',
                'mse',
                keras.metrics.RootMeanSquaredError(name='rmse'),
                direction_accuracy
            ]
        )

        self.model = model

        return model

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        训练模型

        Args:
            X_train: 训练特征 (n_samples, sequence_length, n_features)
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标

        Returns:
            History: 训练历史
        """
        if self.model is None:
            self.build_model()
            self.compile_model()

        # 回调函数
        callback_list = [
            # 早停
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),

            # 学习率衰减
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=config.REDUCE_LR_PATIENCE,
                min_lr=1e-7,
                verbose=1
            ),

            # 模型检查点
            callbacks.ModelCheckpoint(
                filepath=config.MODEL_FILE,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),

            # TensorBoard（可选）
            # callbacks.TensorBoard(
            #     log_dir='./logs',
            #     histogram_freq=1
            # )
        ]

        # 准备验证数据
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        # 训练
        print("开始训练模型...")
        print(f"模型架构: CNN + {'LSTM' if config.USE_LSTM else 'GRU'} + Attention")
        print(f"输入形状: {X_train.shape}")

        self.history = self.model.fit(
            X_train, y_train,
            batch_size=config.BATCH_SIZE,
            epochs=config.EPOCHS,
            validation_data=validation_data,
            callbacks=callback_list,
            verbose=1
        )

        print("模型训练完成")

        return self.history

    def evaluate(self, X_test, y_test):
        """
        评估模型

        Args:
            X_test: 测试特征
            y_test: 测试目标

        Returns:
            dict: 评估指标
        """
        if self.model is None:
            raise ValueError("请先训练或加载模型")

        # 评估
        results = self.model.evaluate(X_test, y_test, verbose=0)

        # 预测
        y_pred = self.model.predict(X_test, verbose=0).flatten()

        # 计算额外指标
        # 方向准确率
        direction_accuracy = np.mean(
            (y_test > 0) == (y_pred > 0)
        )

        # 相关系数
        correlation = np.corrcoef(y_test, y_pred)[0, 1]

        # IC (Information Coefficient)
        ic = correlation

        # 分位数分析（检查模型是否能区分高低收益）
        n_quantiles = 5
        quantiles = np.array_split(np.argsort(y_pred), n_quantiles)
        quantile_returns = [y_test[q].mean() for q in quantiles]

        # 组织结果
        metrics = {
            'loss': results[0],
            'mae': results[1],
            'mse': results[2],
            'rmse': results[3],
            'direction_accuracy': direction_accuracy,
            'correlation': correlation,
            'ic': ic,
            'quantile_returns': quantile_returns,
            'top_quantile_return': quantile_returns[-1],  # 预测最高的20%平均收益
            'bottom_quantile_return': quantile_returns[0]  # 预测最低的20%平均收益
        }

        return metrics

    def predict(self, X):
        """
        预测

        Args:
            X: 特征数据 (n_samples, sequence_length, n_features)

        Returns:
            array: 预测结果
        """
        if self.model is None:
            raise ValueError("请先训练或加载模型")

        predictions = self.model.predict(X, verbose=0).flatten()

        return predictions

    def save(self, model_file=None):
        """
        保存模型

        Args:
            model_file: 模型文件路径
        """
        if model_file is None:
            model_file = config.MODEL_FILE

        if self.model is None:
            raise ValueError("没有可保存的模型")

        self.model.save(model_file)
        print(f"模型已保存: {model_file}")

    def load(self, model_file=None):
        """
        加载模型

        Args:
            model_file: 模型文件路径
        """
        if model_file is None:
            model_file = config.MODEL_FILE

        if not os.path.exists(model_file):
            raise FileNotFoundError(f"模型文件不存在: {model_file}")

        # 自定义对象
        custom_objects = {
            'MultiHeadSelfAttention': MultiHeadSelfAttention
        }

        # 自定义损失函数
        def custom_loss(y_true, y_pred):
            mse = tf.reduce_mean(tf.square(y_true - y_pred))
            direction_loss = -tf.reduce_mean(tf.sign(y_true) * tf.sign(y_pred))

            y_true_expanded = tf.expand_dims(y_true, 1)
            y_pred_expanded = tf.expand_dims(y_pred, 1)
            true_diff = y_true_expanded - tf.transpose(y_true_expanded)
            pred_diff = y_pred_expanded - tf.transpose(y_pred_expanded)
            significant_mask = tf.abs(true_diff) > 0.01
            ranking_loss = tf.reduce_mean(
                tf.where(significant_mask, tf.nn.relu(1.0 - true_diff * pred_diff), 0.0)
            )

            return (
                config.LOSS_MSE_WEIGHT * mse +
                config.LOSS_DIRECTION_WEIGHT * direction_loss +
                config.LOSS_RANKING_WEIGHT * ranking_loss
            )

        def direction_accuracy(y_true, y_pred):
            return tf.reduce_mean(
                tf.cast(tf.equal(tf.sign(y_true), tf.sign(y_pred)), tf.float32)
            )

        custom_objects['custom_loss'] = custom_loss
        custom_objects['direction_accuracy'] = direction_accuracy

        self.model = keras.models.load_model(model_file, custom_objects=custom_objects)

        print(f"模型已加载: {model_file}")

    def summary(self):
        """打印模型结构"""
        if self.model is None:
            raise ValueError("请先构建模型")

        self.model.summary()


class ModelTrainer:
    """模型训练器（高级接口）"""

    def __init__(self):
        self.model = None

    def train_model(self, X_train, y_train, X_val, y_val, input_shape):
        """
        训练模型（完整流程）

        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            input_shape: 输入形状 (sequence_length, n_features)

        Returns:
            QuantModel: 训练好的模型
        """
        print("=" * 70)
        print(" " * 20 + "开始训练量化模型")
        print("=" * 70)

        # 创建模型
        self.model = QuantModel(input_shape=input_shape)

        # 构建和编译
        self.model.build_model()
        self.model.compile_model()

        # 打印模型结构
        print("\n模型结构:")
        self.model.summary()

        # 训练
        print(f"\n训练数据: {X_train.shape}")
        print(f"验证数据: {X_val.shape}")
        print(f"时序窗口长度: {config.SEQUENCE_LENGTH}")

        history = self.model.train(X_train, y_train, X_val, y_val)

        # 评估
        print("\n验证集评估:")
        val_metrics = self.model.evaluate(X_val, y_val)
        for key, value in val_metrics.items():
            if key == 'quantile_returns':
                print(f"{key}: {[f'{v:.4f}' for v in value]}")
            else:
                print(f"{key}: {value:.4f}")

        # 保存模型
        self.model.save()

        print("\n" + "=" * 70)
        print(" " * 20 + "模型训练完成")
        print("=" * 70)

        return self.model


def main():
    """测试模型训练功能"""
    from data_loader import DataLoader
    from dataset import DatasetBuilder

    loader = DataLoader()
    builder = DatasetBuilder()
    trainer = ModelTrainer()

    # 获取测试数据（只用少量数据测试）
    trade_date = loader.get_latest_trade_date()
    stock_list = loader.get_stock_list(trade_date)

    # 只取前10只股票测试
    test_stocks = stock_list.head(10)

    stock_data = {}
    for idx, row in test_stocks.iterrows():
        ts_code = row['ts_code']
        df = loader.get_stock_daily(ts_code)
        if df is not None and len(df) >= config.MIN_HISTORY_DAYS:
            stock_data[ts_code] = df

    if stock_data:
        print(f"测试数据: {len(stock_data)} 只股票")

        # 准备训练数据
        X_train, y_train, X_val, y_val, feature_cols, _, _ = builder.prepare_train_data(stock_data)

        # 训练模型
        model = trainer.train_model(
            X_train, y_train,
            X_val, y_val,
            input_shape=(config.SEQUENCE_LENGTH, len(feature_cols))
        )


if __name__ == "__main__":
    main()
