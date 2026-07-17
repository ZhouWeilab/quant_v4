"""
模型定义与训练模块
功能：定义 CNN + LSTM/GRU + Multi-Head Attention 混合架构，训练、评估、保存模型
专为 A 股超短线量化交易设计（高噪声、微弱信号、非平稳、T+1约束）
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from scipy import stats
import config
import os


def configure_tensorflow_runtime():
    """配置 TensorFlow 运行环境：优先使用指定 GPU，必要时回退 CPU。"""
    if not config.USE_GPU:
        print("TensorFlow 配置为使用 CPU")
        return

    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("未检测到 GPU，TensorFlow 使用 CPU")
        return

    gpu_id = getattr(config, 'GPU_ID', 0)
    if gpu_id < 0 or gpu_id >= len(gpus):
        print(f"GPU_ID={gpu_id} 超出范围，自动使用 GPU 0")
        gpu_id = 0

    selected_gpu = gpus[gpu_id]
    try:
        tf.config.set_visible_devices(selected_gpu, 'GPU')
        if getattr(config, 'TF_GPU_MEMORY_GROWTH', True):
            tf.config.experimental.set_memory_growth(selected_gpu, True)

        if getattr(config, 'ENABLE_MIXED_PRECISION', False):
            keras.mixed_precision.set_global_policy('mixed_float16')
            print("TensorFlow 已启用 mixed_float16 混合精度")

        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"TensorFlow 使用 GPU: {selected_gpu.name}，逻辑 GPU 数量: {len(logical_gpus)}")
    except RuntimeError as e:
        print(f"GPU 设置失败，TensorFlow 可能已初始化: {e}")


configure_tensorflow_runtime()


class NumpyBatchSequence(keras.utils.Sequence):
    """按批次从 numpy 数组取数据，避免一次性把完整训练集拷入 GPU。"""

    def __init__(self, X, y, batch_size, shuffle=False):
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.float32, copy=False)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.y))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.y) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self.X[batch_indices], self.y[batch_indices]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


class DateGroupedBatchSequence(keras.utils.Sequence):
    """每个批次只包含同一交易日，保证横截面损失定义正确。"""

    def __init__(self, X, y, trade_dates, shuffle=False):
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.float32, copy=False)
        dates = np.asarray(trade_dates)
        if len(dates) != len(self.y):
            raise ValueError("trade_dates 与训练样本数量不一致")
        self.groups = [np.flatnonzero(dates == date) for date in np.unique(dates)]
        self.groups = [indices for indices in self.groups if len(indices) >= 2]
        self.shuffle = shuffle
        self.order = np.arange(len(self.groups))
        self.on_epoch_end()

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        indices = self.groups[self.order[idx]]
        return self.X[indices], self.y[indices]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.order)


class MultiArrayDateGroupedBatchSequence(keras.utils.Sequence):
    """在不复制大型数组的前提下合并多个日期分组数据集。"""

    def __init__(self, parts, shuffle=False):
        self.parts = []
        self.groups = []
        for part_idx, (X, y, trade_dates) in enumerate(parts):
            X = X.astype(np.float32, copy=False)
            y = y.astype(np.float32, copy=False)
            dates = np.asarray(trade_dates)
            self.parts.append((X, y))
            for date in np.unique(dates):
                indices = np.flatnonzero(dates == date)
                if len(indices) >= 2:
                    self.groups.append((part_idx, indices))
        self.shuffle = shuffle
        self.order = np.arange(len(self.groups))
        self.on_epoch_end()

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        part_idx, indices = self.groups[self.order[idx]]
        X, y = self.parts[part_idx]
        return X[indices], y[indices]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.order)


@keras.utils.register_keras_serializable(package='quant_v4')
class CrossSectionalRankLoss(keras.losses.Loss):
    """Huber 与日内成对排序的联合损失。"""

    def __init__(
        self, huber_weight=1.0, direction_weight=0.05,
        ranking_weight=0.25, delta=1.0, pair_fractions=None,
        name='cross_sectional_rank_loss'
    ):
        super().__init__(name=name)
        self.huber_weight = float(huber_weight)
        self.direction_weight = float(direction_weight)
        self.ranking_weight = float(ranking_weight)
        self.delta = float(delta)
        self.pair_fractions = list(
            pair_fractions
            or getattr(config, 'RANK_PAIR_FRACTIONS', (0.01, 0.05, 0.10, 0.25, 0.50))
        )
        self.huber = keras.losses.Huber(delta=self.delta)

    def call(self, y_true, y_pred):
        y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1])

        huber_loss = self.huber(y_true, y_pred)
        direction_loss = tf.reduce_mean(
            tf.nn.softplus(-tf.sign(y_true) * y_pred)
        )

        # 每个批次是同一交易日。按多个距离构造成对样本，直接优化排序，
        # 避免用 Pearson 相关替代最终监控的 Spearman 排序目标。
        sample_count = tf.size(y_true)
        pair_losses = []
        for fraction in self.pair_fractions:
            shift = tf.maximum(
                1,
                tf.cast(
                    tf.round(tf.cast(sample_count, tf.float32) * float(fraction)),
                    tf.int32
                )
            )
            true_diff = y_true - tf.roll(y_true, shift=shift, axis=0)
            pred_diff = y_pred - tf.roll(y_pred, shift=shift, axis=0)
            valid_pair = tf.cast(tf.abs(true_diff) > 1e-6, tf.float32)
            pair_loss = tf.nn.softplus(-tf.sign(true_diff) * pred_diff)
            pair_losses.append(
                tf.reduce_sum(pair_loss * valid_pair)
                / (tf.reduce_sum(valid_pair) + 1e-8)
            )
        ranking_loss = tf.add_n(pair_losses) / float(len(pair_losses))

        return (
            self.huber_weight * huber_loss
            + self.direction_weight * direction_loss
            + self.ranking_weight * ranking_loss
        )

    def get_config(self):
        config_dict = super().get_config()
        config_dict.update({
            'huber_weight': self.huber_weight,
            'direction_weight': self.direction_weight,
            'ranking_weight': self.ranking_weight,
            'delta': self.delta,
            'pair_fractions': self.pair_fractions,
        })
        return config_dict


def calculate_daily_ranking_metrics(preds, raw_returns, trade_dates, top_n_list=None):
    """按交易日计算排序和扣费后收益，避免跨市场环境混合评估。"""
    preds = np.asarray(preds, dtype=np.float32)
    raw_returns = np.asarray(raw_returns, dtype=np.float32)
    trade_dates = np.asarray(trade_dates)
    top_n_list = top_n_list or getattr(config, 'EVAL_TOP_N_LIST', [5, 10, 20])

    ic_values = []
    quantile_returns = []
    top_n_returns = {int(n): [] for n in top_n_list}
    long_short_returns = []
    relative_direction_values = []
    raw_direction_values = []

    for trade_date in np.unique(trade_dates):
        mask = trade_dates == trade_date
        if mask.sum() < 10:
            continue

        day_pred = preds[mask]
        day_ret = raw_returns[mask]
        valid = ~(np.isnan(day_pred) | np.isnan(day_ret))
        if valid.sum() < 10:
            continue

        day_pred = day_pred[valid]
        day_ret = day_ret[valid]
        if np.std(day_pred) > 1e-12 and np.std(day_ret) > 1e-12:
            ic, _ = stats.spearmanr(day_pred, day_ret)
            if not np.isnan(ic):
                ic_values.append(ic)

        # 模型目标是横截面强弱，方向应解释为“高于当日中位数”，
        # 原始涨跌方向另行报告，避免把二者混为一谈。
        relative_direction_values.append(float(np.mean(
            (day_pred > np.median(day_pred)) == (day_ret > np.median(day_ret))
        )))
        raw_direction_values.append(float(np.mean(
            (day_pred > 0) == (day_ret > 0)
        )))

        order = np.argsort(day_pred)
        quantile_idx = np.array_split(order, getattr(config, 'QUANTILE_N', 5))
        if quantile_idx and all(len(idx) > 0 for idx in quantile_idx):
            q_returns = [float(day_ret[idx].mean()) for idx in quantile_idx]
            quantile_returns.append(q_returns)
            long_short_returns.append(q_returns[-1] - q_returns[0])

        for top_n in top_n_returns:
            n = min(top_n, len(order))
            if n > 0:
                top_n_returns[top_n].append(float(day_ret[order[-n:]].mean()))

    round_trip_cost = (
        2 * float(getattr(config, 'BACKTEST_COMMISSION', 0.0))
        + 2 * float(getattr(config, 'BACKTEST_TRANSFER_FEE', 0.0))
        + 2 * float(getattr(config, 'BACKTEST_SLIPPAGE', 0.0))
        + float(getattr(config, 'BACKTEST_STAMP_DUTY', 0.0))
    )
    rank_ic = float(np.nanmean(ic_values)) if ic_values else np.nan
    rank_ic_std = float(np.nanstd(ic_values)) if ic_values else np.nan
    metrics = {
        'rank_ic': rank_ic,
        'rank_ic_std': rank_ic_std,
        'rank_ic_ir': (
            rank_ic / rank_ic_std
            if np.isfinite(rank_ic) and np.isfinite(rank_ic_std) and rank_ic_std > 0
            else np.nan
        ),
        'rank_ic_positive_ratio': (
            float(np.mean(np.asarray(ic_values) > 0)) if ic_values else np.nan
        ),
        'relative_direction_accuracy': (
            float(np.nanmean(relative_direction_values))
            if relative_direction_values else np.nan
        ),
        # 兼容旧报告字段，但含义已统一为横截面相对方向。
        'direction_accuracy': (
            float(np.nanmean(relative_direction_values))
            if relative_direction_values else np.nan
        ),
        'raw_direction_accuracy': (
            float(np.nanmean(raw_direction_values)) if raw_direction_values else np.nan
        ),
        'long_short_return': float(np.nanmean(long_short_returns)) if long_short_returns else np.nan,
        'evaluation_days': int(len(ic_values)),
        'round_trip_cost': round_trip_cost,
    }

    if quantile_returns:
        q_mean = np.nanmean(np.asarray(quantile_returns), axis=0)
        metrics['quantile_returns'] = [float(v) for v in q_mean]
        metrics['bottom_quantile_return'] = float(q_mean[0])
        metrics['top_quantile_return'] = float(q_mean[-1])
    else:
        metrics['quantile_returns'] = []
        metrics['bottom_quantile_return'] = np.nan
        metrics['top_quantile_return'] = np.nan

    for top_n, values in top_n_returns.items():
        mean_return = float(np.nanmean(values)) if values else np.nan
        return_std = float(np.nanstd(values)) if values else np.nan
        return_se = (
            return_std / np.sqrt(len(values)) if values and len(values) > 1 else np.nan
        )
        metrics[f'top_{top_n}_return'] = mean_return
        metrics[f'top_{top_n}_return_std'] = return_std
        metrics[f'top_{top_n}_return_se'] = return_se
        metrics[f'top_{top_n}_positive_ratio'] = (
            float(np.mean(np.asarray(values) > 0)) if values else np.nan
        )
        metrics[f'top_{top_n}_net_return'] = (
            mean_return - round_trip_cost if np.isfinite(mean_return) else np.nan
        )
        metrics[f'top_{top_n}_net_positive_ratio'] = (
            float(np.mean(np.asarray(values) > round_trip_cost)) if values else np.nan
        )

    selection_top_n = int(getattr(config, 'SELECTION_TOP_N', 10))
    selection_return = metrics.get(f'top_{selection_top_n}_net_return', np.nan)
    selection_se = metrics.get(f'top_{selection_top_n}_return_se', np.nan)
    se_penalty = float(
        getattr(config, 'VALIDATION_RETURN_SE_PENALTY', 0.0) or 0.0
    )
    metrics['selection_score'] = (
        selection_return - se_penalty * selection_se
        if np.isfinite(selection_return) and np.isfinite(selection_se)
        else selection_return
    )

    return metrics


class ValidationRankingCallback(callbacks.Callback):
    """每个 epoch 后用真实收益评估横截面排序能力，并把指标写入 Keras logs。"""

    def __init__(self, X_val, metadata, batch_size, max_samples=0):
        super().__init__()
        self.X_val = X_val.astype(np.float32, copy=False)
        self.batch_size = batch_size
        self.metadata = metadata.reset_index(drop=True)

        raw_col = 'target_raw' if 'target_raw' in self.metadata.columns else 'target_return'
        if raw_col in self.metadata.columns:
            self.raw_returns = self.metadata[raw_col].astype(float).values
        else:
            self.raw_returns = None
        self.trade_dates = self.metadata['trade_date'].values

        n = len(self.metadata)
        max_samples = int(max_samples or 0)
        if max_samples > 0 and max_samples < n:
            rng = np.random.default_rng(config.RANDOM_SEED)
            self.indices = np.sort(rng.choice(n, size=max_samples, replace=False))
        else:
            self.indices = np.arange(n)

    def _predict_sample(self):
        preds = []
        for start in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[start:start + self.batch_size]
            batch_pred = self.model.predict_on_batch(self.X_val[batch_indices]).reshape(-1)
            preds.append(batch_pred.astype(np.float32, copy=False))
        return np.concatenate(preds) if preds else np.array([], dtype=np.float32)

    @staticmethod
    def _safe_spearman(x, y):
        if len(x) < 10 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
            return np.nan
        corr, _ = stats.spearmanr(x, y)
        return corr

    def _daily_metrics(self, preds, raw_returns, trade_dates):
        metrics = calculate_daily_ranking_metrics(preds, raw_returns, trade_dates)
        return metrics
        ic_values = []
        top_returns = []
        bottom_returns = []

        for trade_date in np.unique(trade_dates):
            mask = trade_dates == trade_date
            if mask.sum() < 10:
                continue

            day_pred = preds[mask]
            day_ret = raw_returns[mask]
            valid = ~(np.isnan(day_pred) | np.isnan(day_ret))
            if valid.sum() < 10:
                continue

            day_pred = day_pred[valid]
            day_ret = day_ret[valid]
            ic = self._safe_spearman(day_pred, day_ret)
            if not np.isnan(ic):
                ic_values.append(ic)

            order = np.argsort(day_pred)
            q = max(1, int(np.ceil(len(order) / 5)))
            bottom_returns.append(float(np.mean(day_ret[order[:q]])))
            top_returns.append(float(np.mean(day_ret[order[-q:]])))

        rank_ic = float(np.nanmean(ic_values)) if ic_values else np.nan
        top_return = float(np.nanmean(top_returns)) if top_returns else np.nan
        bottom_return = float(np.nanmean(bottom_returns)) if bottom_returns else np.nan
        long_short = top_return - bottom_return if not np.isnan(top_return + bottom_return) else np.nan
        return rank_ic, top_return, bottom_return, long_short

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.raw_returns is None:
            return

        preds = self._predict_sample()
        raw_returns = self.raw_returns[self.indices]
        trade_dates = self.trade_dates[self.indices]
        metrics = self._daily_metrics(
            preds, raw_returns, trade_dates
        )

        rank_ic = metrics['rank_ic']
        top_return = metrics['top_quantile_return']
        bottom_return = metrics['bottom_quantile_return']
        long_short = metrics['long_short_return']
        top_10_return = metrics.get('top_10_return', np.nan)
        top_10_net_return = metrics.get('top_10_net_return', np.nan)
        selection_score = metrics.get('selection_score', np.nan)

        logs['val_rank_ic'] = rank_ic
        logs['val_top_quantile_return'] = top_return
        logs['val_bottom_quantile_return'] = bottom_return
        logs['val_long_short_return'] = long_short
        logs['val_top_10_return'] = top_10_return
        logs['val_top_10_net_return'] = top_10_net_return
        logs['val_selection_score'] = selection_score
        print(
            f"\nval_rank_ic: {rank_ic:.4f} - "
            f"val_top_quantile_return: {top_return:.4f} - "
            f"val_top_10_return: {top_10_return:.4f} - "
            f"val_top_10_net_return: {top_10_net_return:.4f} - "
            f"val_selection_score: {selection_score:.4f} - "
            f"val_long_short_return: {long_short:.4f}"
        )


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
                        dropout=config.DROPOUT_RATE,
                        recurrent_dropout=getattr(config, 'RNN_RECURRENT_DROPOUT', 0.0)),
                    name=f'bi_{rnn_name}_{i+1}'
                )(x)
            else:
                x = RNN(units, return_sequences=return_sequences,
                        dropout=config.DROPOUT_RATE,
                        recurrent_dropout=getattr(config, 'RNN_RECURRENT_DROPOUT', 0.0),
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
        target_type = getattr(config, 'TARGET_TYPE', 'classification')
        if target_type == 'regression':
            # 回归：预测连续收益率（线性输出）
            outputs = layers.Dense(1, activation='linear', dtype='float32', name='output')(x)
        else:
            # 二分类：预测上涨概率
            outputs = layers.Dense(1, activation='sigmoid', dtype='float32', name='output')(x)

        # 构建模型
        model = models.Model(inputs=inputs, outputs=outputs, name='cnn_lstm_attention_quant_model')

        # 保存已构建模型实例，供后续编译/训练使用
        self.model = model

        return model

    def compile_model(self, model=None):
        """
        编译模型（回归/分类自适应）

        Args:
            model: Keras 模型，如果为 None 则使用 self.model
        """
        if model is None:
            model = self.model

        target_type = getattr(config, 'TARGET_TYPE', 'classification')
        if target_type == 'regression':
            regression_loss = getattr(config, 'REGRESSION_LOSS', 'mse')
            if regression_loss == 'cross_sectional_rank':
                loss_fn = CrossSectionalRankLoss(
                    huber_weight=getattr(config, 'LOSS_MSE_WEIGHT', 1.0),
                    direction_weight=getattr(config, 'LOSS_DIRECTION_WEIGHT', 0.05),
                    ranking_weight=getattr(config, 'LOSS_RANKING_WEIGHT', 0.25),
                    delta=getattr(config, 'HUBER_DELTA', 1.0)
                )
            elif regression_loss == 'huber':
                loss_fn = keras.losses.Huber(delta=getattr(config, 'HUBER_DELTA', 1.0))
            elif regression_loss == 'mse':
                loss_fn = 'mse'
            else:
                raise ValueError(f"Unsupported REGRESSION_LOSS: {regression_loss}")
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
                loss=loss_fn,
                metrics=['mae', 'mse']
            )
        else:
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
                loss='binary_crossentropy',
                metrics=[
                    'accuracy',
                    keras.metrics.AUC(name='auc'),
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')
                ]
            )

        self.model = model
        return model

    def train(
        self, X_train, y_train, X_val=None, y_val=None,
        train_metadata=None, val_metadata=None, model_file=None
    ):
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
        monitor = 'val_loss'
        monitor_mode = 'min'
        ranking_callback = None
        if X_val is not None and val_metadata is not None:
            monitor = getattr(config, 'VALIDATION_MONITOR', 'val_rank_ic')
            monitor_mode = getattr(config, 'VALIDATION_MONITOR_MODE', 'max')
            ranking_callback = ValidationRankingCallback(
                X_val,
                val_metadata,
                batch_size=config.BATCH_SIZE,
                max_samples=getattr(config, 'VALIDATION_METRIC_MAX_SAMPLES', 0)
            )

        checkpoint_file = model_file or config.MODEL_FILE
        callback_list = [
            *([ranking_callback] if ranking_callback is not None else []),
            # 早停
            callbacks.EarlyStopping(
                monitor=monitor,
                mode=monitor_mode,
                patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),

            # 学习率衰减
            callbacks.ReduceLROnPlateau(
                monitor=monitor,
                mode=monitor_mode,
                factor=0.5,
                patience=config.REDUCE_LR_PATIENCE,
                min_lr=1e-7,
                verbose=1
            ),

            # 模型检查点
            callbacks.ModelCheckpoint(
                filepath=checkpoint_file,
                monitor=monitor,
                mode=monitor_mode,
                save_best_only=True,
                verbose=1
            ),

            # TensorBoard（可选）
            # callbacks.TensorBoard(
            #     log_dir='./logs',
            #     histogram_freq=1
            # )
        ]

        # 使用按批次取数的 Sequence，避免超大 numpy 数组被一次性转成 GPU 常量
        use_date_batches = (
            getattr(config, 'DATE_GROUPED_BATCHES', False)
            and train_metadata is not None
            and val_metadata is not None
            and 'trade_date' in train_metadata.columns
            and 'trade_date' in val_metadata.columns
        )
        validation_data = None
        if X_val is not None and y_val is not None:
            if use_date_batches:
                validation_data = DateGroupedBatchSequence(
                    X_val, y_val, val_metadata['trade_date'].values,
                    shuffle=False
                )
            else:
                validation_data = NumpyBatchSequence(
                    X_val, y_val,
                    batch_size=config.BATCH_SIZE,
                    shuffle=False
                )
        if use_date_batches:
            train_data = DateGroupedBatchSequence(
                X_train, y_train, train_metadata['trade_date'].values,
                shuffle=True
            )
            print("训练批次按交易日分组，启用横截面排序损失")
        else:
            train_data = NumpyBatchSequence(
                X_train, y_train,
                batch_size=config.BATCH_SIZE,
                shuffle=True
            )

        # 训练
        print("开始训练模型...")
        print(f"模型架构: CNN + {'LSTM' if config.USE_LSTM else 'GRU'} + Attention")
        print(f"输入形状: {X_train.shape}")
        print(f"训练批次数: {len(train_data)}, 验证批次数: {len(validation_data) if validation_data else 0}")

        self.history = self.model.fit(
            train_data,
            epochs=config.EPOCHS,
            validation_data=validation_data,
            callbacks=callback_list,
            verbose=1
        )

        print("模型训练完成")

        return self.history

    def evaluate(self, X_test, y_test, metadata=None):
        """
        评估模型（二分类指标）

        Args:
            X_test: 测试特征
            y_test: 测试目标

        Returns:
            dict: 评估指标
        """
        if self.model is None:
            raise ValueError("请先训练或加载模型")

        if (
            getattr(config, 'DATE_GROUPED_BATCHES', False)
            and metadata is not None
            and 'trade_date' in metadata.columns
        ):
            test_data = DateGroupedBatchSequence(
                X_test, y_test, metadata['trade_date'].values,
                shuffle=False
            )
        else:
            test_data = NumpyBatchSequence(
                X_test, y_test,
                batch_size=config.BATCH_SIZE,
                shuffle=False
            )
        results = self.model.evaluate(test_data, verbose=0)
        y_pred_value = self.model.predict(test_data, verbose=0).flatten()
        y_test = y_test.astype(np.float32, copy=False)
        eval_target = y_test
        if metadata is not None:
            raw_col = 'target_raw' if 'target_raw' in metadata.columns else 'target_return'
            if raw_col in metadata.columns:
                eval_target = metadata[raw_col].astype(float).values

        target_type = getattr(config, 'TARGET_TYPE', 'classification')
        if target_type == 'regression':
            mse = float(np.mean(np.square(y_test - y_pred_value)))
            mae = float(np.mean(np.abs(y_test - y_pred_value)))
            if metadata is not None and 'trade_date' in metadata.columns:
                ranking_metrics = calculate_daily_ranking_metrics(
                    y_pred_value,
                    eval_target,
                    metadata['trade_date'].values
                )
            else:
                valid = ~(np.isnan(y_pred_value) | np.isnan(eval_target))
                rank_ic = np.nan
                if valid.sum() > 10:
                    rank_ic, _ = stats.spearmanr(y_pred_value[valid], eval_target[valid])
                n_quantiles = getattr(config, 'QUANTILE_N', 5)
                quantiles = np.array_split(np.argsort(y_pred_value), n_quantiles)
                quantile_returns = [float(eval_target[q].mean()) for q in quantiles]
                ranking_metrics = {
                    'relative_direction_accuracy': float(np.mean(
                        (y_pred_value > np.median(y_pred_value))
                        == (eval_target > np.median(eval_target))
                    )),
                    'direction_accuracy': float(np.mean(
                        (y_pred_value > np.median(y_pred_value))
                        == (eval_target > np.median(eval_target))
                    )),
                    'raw_direction_accuracy': float(np.mean(
                        (y_pred_value > 0) == (eval_target > 0)
                    )),
                    'rank_ic': float(rank_ic) if not np.isnan(rank_ic) else np.nan,
                    'quantile_returns': quantile_returns,
                    'top_quantile_return': quantile_returns[-1],
                    'bottom_quantile_return': quantile_returns[0],
                    'long_short_return': quantile_returns[-1] - quantile_returns[0]
                }

            metrics = {
                'loss': float(results[0]),
                'mae': float(results[1]) if len(results) > 1 else mae,
                'mse': mse,
                'rmse': float(np.sqrt(mse)),
            }
            metrics.update(ranking_metrics)
            return metrics

        y_pred = (y_pred_value > 0.5).astype(int)
        accuracy = np.mean(y_test == y_pred)
        up_mask = y_test == 1
        up_accuracy = np.mean(y_pred[up_mask] == 1) if np.any(up_mask) else 0.0
        n_quantiles = 5
        quantiles = np.array_split(np.argsort(y_pred_value), n_quantiles)
        quantile_hit_rates = [float(y_test[q].mean()) for q in quantiles]

        metrics = {
            'loss': results[0],
            'accuracy': results[1],
            'auc': results[2],
            'precision': results[3],
            'recall': results[4],
            'direction_accuracy': accuracy,
            'up_accuracy': up_accuracy,
            'quantile_hit_rates': quantile_hit_rates,
            'top_quantile_hit_rate': quantile_hit_rates[-1],
            'bottom_quantile_hit_rate': quantile_hit_rates[0]
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
            'MultiHeadSelfAttention': MultiHeadSelfAttention,
            'CrossSectionalRankLoss': CrossSectionalRankLoss,
        }

        self.model = keras.models.load_model(
            model_file,
            custom_objects=custom_objects,
            compile=False
        )

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

    def train_model(
        self, X_train, y_train, X_val, y_val, input_shape,
        meta_train=None, meta_val=None, X_test=None, y_test=None,
        meta_test=None, model_file=None
    ):
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

        history = self.model.train(
            X_train, y_train, X_val, y_val,
            train_metadata=meta_train,
            val_metadata=meta_val,
            model_file=model_file
        )

        # 评估
        print("\n验证集评估:")
        val_metrics = self.model.evaluate(X_val, y_val, metadata=meta_val)
        for key, value in val_metrics.items():
            if key in ['quantile_hit_rates', 'quantile_returns']:
                print(f"{key}: {[f'{v:.4f}' for v in value]}")
            else:
                print(f"{key}: {value:.4f}")
        self.model.validation_metrics = val_metrics

        if X_test is not None and y_test is not None:
            print("\n最终测试集评估（不参与早停与模型选择）:")
            test_metrics = self.model.evaluate(
                X_test, y_test, metadata=meta_test
            )
            for key, value in test_metrics.items():
                if key in ['quantile_hit_rates', 'quantile_returns']:
                    print(f"{key}: {[f'{v:.4f}' for v in value]}")
                else:
                    print(f"{key}: {value:.4f}")
            self.model.test_metrics = test_metrics

        refit_epochs = int(getattr(config, 'PRODUCTION_REFIT_EPOCHS', 0) or 0)
        if refit_epochs > 0 and X_test is not None and meta_train is not None:
            print(
                f"\n生产模型回灌最近验证/测试数据: {refit_epochs} epochs "
                "（上述测试指标已在回灌前冻结）"
            )
            refit_data = MultiArrayDateGroupedBatchSequence(
                [
                    (X_train, y_train, meta_train['trade_date'].values),
                    (X_val, y_val, meta_val['trade_date'].values),
                    (X_test, y_test, meta_test['trade_date'].values),
                ],
                shuffle=True
            )
            refit_lr = float(getattr(config, 'PRODUCTION_REFIT_LR', 0.0001))
            self.model.model.optimizer.learning_rate.assign(refit_lr)
            self.model.model.fit(
                refit_data, epochs=refit_epochs, verbose=1
            )

        # 保存模型
        self.model.save(model_file)

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
        (
            X_train, y_train, X_val, y_val, X_test, y_test,
            feature_cols, meta_train, meta_val, meta_test
        ) = builder.prepare_train_data(stock_data)

        # 训练模型
        model = trainer.train_model(
            X_train, y_train,
            X_val, y_val,
            input_shape=(config.SEQUENCE_LENGTH, len(feature_cols)),
            meta_train=meta_train,
            meta_val=meta_val,
            X_test=X_test,
            y_test=y_test,
            meta_test=meta_test
        )


if __name__ == "__main__":
    main()
