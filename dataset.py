"""
数据集构建模块
功能：构建训练集、验证集、测试集，计算标签（次日涨跌幅），数据标准化
支持时序窗口数据构建，适配 CNN+LSTM+Attention 模型
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import pickle
import config
from features import FeatureEngineer


class DatasetBuilder:
    """数据集构建器（支持时序窗口）"""

    def __init__(self, use_robust_scaler=True):
        """
        初始化

        Args:
            use_robust_scaler: 是否使用 RobustScaler（对异常值更鲁棒）
        """
        self.feature_engineer = FeatureEngineer()
        self.scaler = None
        self.feature_cols = None
        self.use_robust_scaler = use_robust_scaler

    def create_target(self, df, horizon=None):
        """
        创建目标变量：未来第 N 日涨跌幅

        Args:
            df: 包含特征的数据
            horizon: 预测未来第几天（默认为 config.FORECAST_HORIZON）

        Returns:
            DataFrame: 添加目标列的数据
        """
        if horizon is None:
            horizon = config.FORECAST_HORIZON

        # 未来第 N 日收益率
        df['target_return'] = df['close'].shift(-horizon) / df['close'] - 1

        # 删除最后 N 行（没有未来数据）
        df = df[:-horizon].copy()

        # 目标变量
        df['target'] = df['target_return']

        return df

    def prepare_single_stock(self, df, ts_code):
        """
        准备单只股票的数据

        Args:
            df: 原始日线数据
            ts_code: 股票代码

        Returns:
            DataFrame: 包含特征和目标的数据
        """
        # 添加股票代码
        df['ts_code'] = ts_code

        # 添加特征
        df = self.feature_engineer.add_all_features(df)

        # 创建目标
        df = self.create_target(df)

        # 删除包含 NaN 的行
        df = df.dropna()

        return df

    def build_dataset(self, stock_data_dict):
        """
        构建完整数据集

        Args:
            stock_data_dict: {ts_code: DataFrame} 格式的股票数据字典

        Returns:
            DataFrame: 合并后的完整数据集
        """
        print("构建数据集...")

        all_dfs = []

        for ts_code, df in stock_data_dict.items():
            try:
                prepared_df = self.prepare_single_stock(df, ts_code)

                if len(prepared_df) > config.SEQUENCE_LENGTH:  # 确保有足够数据构建时序
                    all_dfs.append(prepared_df)

            except Exception as e:
                print(f"处理 {ts_code} 失败: {e}")
                continue

        if not all_dfs:
            raise ValueError("没有有效的数据")

        # 合并所有股票数据
        dataset = pd.concat(all_dfs, ignore_index=True)

        # 按股票代码和日期排序（重要：时序数据必须按股票分组）
        dataset = dataset.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

        print(f"数据集构建完成: {len(dataset)} 条样本")

        return dataset

    def split_dataset(self, dataset, val_split=None):
        """
        划分训练集和验证集（按时间顺序）

        Args:
            dataset: 完整数据集
            val_split: 验证集比例

        Returns:
            tuple: (train_df, val_df)
        """
        if val_split is None:
            val_split = config.VALIDATION_SPLIT

        # 按时间顺序划分（避免未来数据泄露）
        unique_dates = sorted(dataset['trade_date'].unique())
        split_idx = int(len(unique_dates) * (1 - val_split))
        split_date = unique_dates[split_idx]

        train_df = dataset[dataset['trade_date'] < split_date].copy()
        val_df = dataset[dataset['trade_date'] >= split_date].copy()

        print(f"训练集: {len(train_df)} 条样本")
        print(f"验证集: {len(val_df)} 条样本")

        return train_df, val_df

    def get_feature_columns(self, dataset):
        """
        获取特征列名

        Args:
            dataset: 数据集

        Returns:
            list: 特征列名列表
        """
        return self.feature_engineer.get_feature_columns(dataset)

    def create_sequences(self, df, feature_cols):
        """
        创建时序窗口数据

        Args:
            df: 数据集（必须已按 ts_code 和 trade_date 排序）
            feature_cols: 特征列名

        Returns:
            tuple: (X_sequences, y_targets, metadata)
                - X_sequences: (n_samples, sequence_length, n_features)
                - y_targets: (n_samples,)
                - metadata: DataFrame with ts_code, trade_date, close
        """
        X_list = []
        y_list = []
        meta_list = []

        # 按股票代码分组
        grouped = df.groupby('ts_code')

        for ts_code, group_df in grouped:
            group_df = group_df.sort_values('trade_date').reset_index(drop=True)

            # 提取特征和目标
            features = group_df[feature_cols].values
            targets = group_df['target'].values

            # 滑动窗口创建序列
            for i in range(len(group_df) - config.SEQUENCE_LENGTH + 1):
                # 时序特征窗口
                X_seq = features[i:i + config.SEQUENCE_LENGTH]

                # 目标：窗口最后一天的目标值
                y_target = targets[i + config.SEQUENCE_LENGTH - 1]

                # 元数据：最后一天的信息
                meta_row = group_df.iloc[i + config.SEQUENCE_LENGTH - 1][
                    ['ts_code', 'trade_date', 'close']
                ]

                X_list.append(X_seq)
                y_list.append(y_target)
                meta_list.append(meta_row)

        # 转换为 numpy 数组
        X_sequences = np.array(X_list)
        y_targets = np.array(y_list)
        metadata = pd.DataFrame(meta_list).reset_index(drop=True)

        return X_sequences, y_targets, metadata

    def fit_scaler(self, train_df, feature_cols=None):
        """
        训练标准化器

        Args:
            train_df: 训练数据
            feature_cols: 特征列名

        Returns:
            Scaler: 训练好的标准化器
        """
        if feature_cols is None:
            feature_cols = self.get_feature_columns(train_df)

        self.feature_cols = feature_cols

        # 初始化标准化器
        if self.use_robust_scaler:
            # RobustScaler 对异常值更鲁棒（使用中位数和四分位数）
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()

        # 训练
        X_train = train_df[feature_cols].values
        self.scaler.fit(X_train)

        print(f"标准化器训练完成，特征数量: {len(feature_cols)}")
        print(f"使用标准化器: {type(self.scaler).__name__}")

        return self.scaler

    def transform_features(self, df, feature_cols=None):
        """
        标准化特征（不改变 DataFrame 结构）

        Args:
            df: 数据集
            feature_cols: 特征列名

        Returns:
            DataFrame: 标准化后的数据集
        """
        if feature_cols is None:
            feature_cols = self.feature_cols

        if self.scaler is None:
            raise ValueError("请先调用 fit_scaler() 训练标准化器")

        df = df.copy()

        # 标准化特征列
        df[feature_cols] = self.scaler.transform(df[feature_cols].values)

        return df

    def prepare_train_data(self, stock_data_dict, val_split=None):
        """
        准备训练数据（完整流程，支持时序窗口）

        Args:
            stock_data_dict: 股票数据字典
            val_split: 验证集比例

        Returns:
            tuple: (X_train, y_train, X_val, y_val, feature_cols, meta_train, meta_val)
        """
        # 构建数据集
        dataset = self.build_dataset(stock_data_dict)

        # 划分训练集和验证集
        train_df, val_df = self.split_dataset(dataset, val_split)

        # 获取特征列
        feature_cols = self.get_feature_columns(train_df)

        # 训练标准化器
        self.fit_scaler(train_df, feature_cols)

        # 标准化数据
        train_df = self.transform_features(train_df, feature_cols)
        val_df = self.transform_features(val_df, feature_cols)

        # 创建时序窗口数据
        print("\n创建训练集时序窗口...")
        X_train, y_train, meta_train = self.create_sequences(train_df, feature_cols)

        print("创建验证集时序窗口...")
        X_val, y_val, meta_val = self.create_sequences(val_df, feature_cols)

        print(f"\n训练集: X={X_train.shape}, y={y_train.shape}")
        print(f"验证集: X={X_val.shape}, y={y_val.shape}")

        return X_train, y_train, X_val, y_val, feature_cols, meta_train, meta_val

    def prepare_predict_data(self, stock_data_dict):
        """
        准备预测数据（使用最近 N 天的数据）

        Args:
            stock_data_dict: 股票数据字典

        Returns:
            tuple: (X, df_info) - 特征和对应的股票信息
        """
        if self.scaler is None or self.feature_cols is None:
            raise ValueError("请先训练模型或加载标准化器")

        predict_dfs = []

        for ts_code, df in stock_data_dict.items():
            try:
                # 准备数据
                prepared_df = self.feature_engineer.add_all_features(df)
                prepared_df['ts_code'] = ts_code

                # 删除 NaN
                prepared_df = prepared_df.dropna()

                if len(prepared_df) < config.SEQUENCE_LENGTH:
                    continue

                # 只取最近 SEQUENCE_LENGTH 天
                recent_data = prepared_df.tail(config.SEQUENCE_LENGTH).copy()

                predict_dfs.append(recent_data)

            except Exception as e:
                print(f"准备预测数据 {ts_code} 失败: {e}")
                continue

        if not predict_dfs:
            raise ValueError("没有有效的预测数据")

        # 合并
        df_predict = pd.concat(predict_dfs, ignore_index=True)

        # 标准化特征
        df_predict = self.transform_features(df_predict, self.feature_cols)

        # 创建时序窗口（每只股票一个窗口）
        X_list = []
        info_list = []

        grouped = df_predict.groupby('ts_code')

        for ts_code, group_df in grouped:
            group_df = group_df.sort_values('trade_date').reset_index(drop=True)

            if len(group_df) == config.SEQUENCE_LENGTH:
                # 提取特征
                X_seq = group_df[self.feature_cols].values

                # 最后一天的信息
                last_row = group_df.iloc[-1]

                X_list.append(X_seq)
                info_list.append({
                    'ts_code': last_row['ts_code'],
                    'trade_date': last_row['trade_date'],
                    'close': last_row['close']
                })

        # 转换为 numpy 数组
        X = np.array(X_list)
        df_info = pd.DataFrame(info_list)

        print(f"预测数据准备完成: {len(X)} 只股票")

        return X, df_info

    def save_scaler(self, scaler_file=None, feature_cols_file=None):
        """
        保存标准化器和特征列

        Args:
            scaler_file: 标准化器文件路径
            feature_cols_file: 特征列文件路径
        """
        if scaler_file is None:
            scaler_file = config.SCALER_FILE
        if feature_cols_file is None:
            feature_cols_file = config.FEATURE_COLS_FILE

        # 保存标准化器
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)

        # 保存特征列
        with open(feature_cols_file, 'wb') as f:
            pickle.dump(self.feature_cols, f)

        print(f"标准化器已保存: {scaler_file}")
        print(f"特征列已保存: {feature_cols_file}")

    def load_scaler(self, scaler_file=None, feature_cols_file=None):
        """
        加载标准化器和特征列

        Args:
            scaler_file: 标准化器文件路径
            feature_cols_file: 特征列文件路径
        """
        if scaler_file is None:
            scaler_file = config.SCALER_FILE
        if feature_cols_file is None:
            feature_cols_file = config.FEATURE_COLS_FILE

        # 加载标准化器
        with open(scaler_file, 'rb') as f:
            self.scaler = pickle.load(f)

        # 加载特征列
        with open(feature_cols_file, 'rb') as f:
            self.feature_cols = pickle.load(f)

        print(f"标准化器已加载: {scaler_file}")
        print(f"特征列已加载，数量: {len(self.feature_cols)}")


def main():
    """测试数据集构建功能"""
    from data_loader import DataLoader

    loader = DataLoader()
    builder = DatasetBuilder()

    # 获取测试数据
    trade_date = loader.get_latest_trade_date()
    stock_list = loader.get_stock_list(trade_date)

    # 只取前5只股票测试
    test_stocks = stock_list.head(5)

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

        print(f"\n特征列: {feature_cols[:5]}...")
        print(f"时序窗口长度: {config.SEQUENCE_LENGTH}")
        print(f"目标统计: mean={y_train.mean():.4f}, std={y_train.std():.4f}")


if __name__ == "__main__":
    main()
