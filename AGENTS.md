# AGENTS.md — A 股超短线量化交易系统 v4.0

> 本文件面向 AI Coding Agent 编写。如果你正在阅读此文件，说明你对本项目一无所知，请从这里开始。

---

## 项目概览

本项目是一个基于深度学习和 XGBoost 的 **A 股超短线量化交易系统**，目标为预测 T+1 开盘至 T+6 收盘的五交易日收益，并每日推荐 Top N 只股票。项目采用模块化架构，数据流为：

```
Tushare API → 数据清洗 → 特征工程(80+维) → 时序窗口构建 → 模型训练/预测 → 选股输出
```

当前默认模型架构为 **CNN + GRU + Multi-Head Attention** 混合网络（时序窗口 20 天），同时支持 XGBoost 基准模型和其他预设深度学习架构。

**本地映射目录**: `Y:\quant_v4`  
**服务器目录**: `/home/zhouwei/quant_v4`（语法和运行检查必须在服务器执行）

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 编程语言 | Python 3 |
| 数据获取 | Tushare Pro API (`tushare>=1.3.0`) |
| 数据处理 | `pandas>=2.0.0`, `numpy>=1.24.0` |
| 特征工程/标准化 | `scikit-learn>=1.3.0` (RobustScaler) |
| 深度学习 | `tensorflow>=2.15.0` (Keras Functional API) |
| 日期工具 | `python-dateutil>=2.8.0` |

**模型保存格式**: Keras HDF5 (`.h5`)  
**标准化器保存格式**: `pickle` (`.pkl`)

---

## 项目结构

```
quant_v4/
├── config.py              # 全局配置（超参数、路径、Token、筛选条件）
├── data_loader.py         # Tushare 数据获取、清洗、股票筛选
├── features.py            # 特征工程（收益率、均线、RSI、MACD、布林带、ATR 等 80+ 维）
├── dataset.py             # 数据集构建：时序滑动窗口、标签生成、RobustScaler 标准化
├── model.py               # CNN+LSTM/GRU+Attention 模型定义、自定义损失函数、训练、评估
├── predictor.py           # 股票预测器 + 带成交约束和费用的回测引擎
├── baseline_models.py     # XGBoost 与线性基准模型
├── walkforward.py         # 滚动训练与样本外评估
├── main.py                # CLI 入口（训练、预测、回测、因子、基准、walk-forward）
├── model_selector.py      # 交互式模型架构切换工具（6 种预设配置）
├── test_model.py          # 模型架构冒烟测试（无真实数据，验证构建/前向/训练/保存加载）
├── test_pipeline.py       # 数据泄露、标签、元数据、选股与成交模拟测试
├── requirements.txt       # Python 依赖列表
├── prompt.txt             # 本项目的原始需求 Prompt
│
├── README.md              # 项目总览文档（v4.0）
├── README_v4.md           # CNN+LSTM+Attention 专项文档
├── ARCHITECTURE.md        # 架构升级 rationale、损失函数数学推导
├── UPGRADE_SUMMARY.md     # 升级变更日志、使用方法
│
├── data/daily_cache/      # 按交易日保存的日线数据缓存
├── cache/                 # 复权因子、daily_basic、资金流、龙虎榜等缓存
├── models/                # 模型产物目录
│   ├── quant_model.h5     # 训练好的 Keras 模型
│   ├── scaler.pkl         # 拟合好的 RobustScaler
│   ├── feature_cols.pkl   # 特征列名列表
│   └── model_meta.json    # 标签口径、训练截止日和特征哈希
│
└── [空文件] 1.24.0, 1.3.0, 2.0.0, 2.15.0, 2.8.0
    # 这些是 0 字节文件，疑似命令行误操作产生的垃圾文件，可安全删除
```

---

## 模块详解

### `config.py`
- 所有超参数集中管理，包括日期范围、股票筛选阈值、特征参数、网络结构参数、训练参数、回测参数。
- Tushare Token 只从环境变量或 `~/.config/quant_v4/tushare_token` 读取，禁止写入源码。
- 默认 `START_DATE = "20180101"`。
- **注意**: `MODEL_LAYERS`、`N_JOBS`、`LOG_FILE`/`LOG_LEVEL` 等配置项在当前代码中**未被实际使用**。
- 启动时会自动创建 `./data`、`./models`、`./cache` 目录。

### `data_loader.py`
- `DataLoader` 类封装 Tushare Pro API。
- 功能：获取交易日历、最新交易日、A 股列表、个股日线、全市场日线、流动性/波动率/涨跌停筛选。
- 数据清洗包含价格逻辑验证（high ≥ low, high ≥ close/open 等）。
- 大股票池按交易日批量拉取，小股票池保留逐股票串行路径。
- 日线、复权因子、daily_basic、资金流、龙虎榜和涨跌停数据均支持本地缓存读取。
- 历史市值按交易日合并，只允许向后填充已知值，禁止用当前市值回填历史。

### `features.py`
- `FeatureEngineer` 类构建 80+ 维特征：
  - 价格特征：1/3/5/10/20 日收益率、振幅、影线比率、K 线位置
  - 成交量特征：量比、成交额变化、价量相关性
  - 均线特征：MA5/10/20/30/60、价格偏离度、MA 多头排列标识
  - 技术指标：RSI(14)、MACD(12,26,9)、布林带(20,2)、ATR(14)
  - 动量与波动率特征

### `dataset.py`
- `DatasetBuilder` 类负责从原始数据到训练样本的完整转换。
- **关键设计**（防止数据泄露）：
  - 按日期做**时序划分**（非随机打乱），验证集 = 较晚的日期。
  - 按股票分组构建滑动窗口，**禁止跨股票拼接时序**。
  - 使用 `RobustScaler`（基于中位数/IQR）替代 StandardScaler。
- `prepare_train_data()`: 完整训练数据流水线。
- `prepare_predict_data()`: 为每只股票取最近 `SEQUENCE_LENGTH` 天，生成单条预测序列。

### `model.py`
- `MultiHeadSelfAttention`: 自定义 Keras Layer，包装 `keras.layers.MultiHeadAttention`，带残差连接和 LayerNormalization。
- `QuantModel`: 根据 `config.py` 开关动态构建网络：
  - CNN 模块：1D Conv → BatchNorm → ReLU → SpatialDropout1D → MaxPool（可选）
  - RNN 模块：LSTM 或 GRU，支持双向
  - Attention 模块：多头自注意力（需 `RETURN_SEQUENCES=True`）
  - 全连接层：3 层 Dense + BN + ReLU + Dropout
- **自定义损失函数**: 回归误差 + 方向约束 + 横截面排序损失，具体权重以 `config.py` 为准。
- **评估指标**: MAE、MSE、RMSE、direction_accuracy、IC、分位数收益
- **回调函数**: EarlyStopping、ReduceLROnPlateau、ModelCheckpoint
- `ModelTrainer`: 高层封装，负责 build → compile → train → evaluate → save。

### `predictor.py`
- `StockPredictor`: 加载模型 → 获取候选股票 → 预测 → 排序 → 添加股票名称/行业 → 格式化输出（中文列名）。
- `StockPredictor` 强制校验模型类型、训练截止日、标签价格口径、特征数量和特征哈希。
- `BacktestEngine` 按 T+1 开盘买入、T+6 收盘卖出模拟非重叠组合，计入整数手、费用、滑点、成交容量和延迟卖出。
- 普通回测拒绝模型训练期覆盖回测期的前视用法；正式评估优先使用 `walkforward` 模式。

### `main.py`
- CLI 入口，主要模式：
  - `train`: 训练模型（先检查数据 → 特征工程 → 构建序列 → 训练 → 保存）
  - `predict`: 预测选股（默认模式，输出中文表格并保存 CSV）
  - `backtest`: 运行回测
  - `baseline`: 训练 XGBoost/线性基准
  - `walkforward`: 滚动训练和样本外评估

### `model_selector.py`
- 交互式脚本，提供 6 种预设架构切换：
  1. 简单 DNN
  2. 纯 CNN
  3. 纯 LSTM
  4. CNN + LSTM
  5. CNN + LSTM + Attention（完整版）
  6. 轻量 CNN + GRU + Attention（当前默认配置）
- 运行后直接在运行时修改 `config` 模块的变量值。

### `test_model.py`
- 冒烟测试脚本，使用随机数据验证：
  - 自定义 Attention 层能否前向传播
  - 模型能否成功 build + compile
  - 前向传播输出 shape 是否正确
  - 能否完成 3 个 epoch 的训练
  - 模型 save/load 是否正常工作

---

## 构建与运行命令

本项目**无 formal build system**（无 `setup.py`、`pyproject.toml`、`Makefile`）。

### 安装依赖

```bash
pip install -r requirements.txt
```

### 训练模型

```bash
python main.py --mode train
```

流程：获取全市场历史数据 → 特征工程 → 构建时序窗口 → RobustScaler 标准化 → 训练 CNN+GRU+Attention → 保存模型/标准化器/特征列名/元数据。

### 预测选股

```bash
python main.py --mode predict
# 或省略参数（默认即为 predict）
python main.py
```

输出：中文格式 Top 10 股票列表，同时保存到 `./recommendations_YYYYMMDD.csv`。

### 回测

```bash
python main.py --mode backtest
```

### 切换模型架构

```bash
python model_selector.py
```

按提示输入 1~6 即可切换配置，随后运行 `python main.py --mode train` 重新训练。

---

## 测试

### 冒烟测试（无需真实数据、无需 Tushare Token）

```bash
python test_model.py
```

该脚本仅验证模型架构能否在随机数据上正常 build、forward、train、save/load。如果此测试通过，说明 TensorFlow 环境和模型代码基本正确。

### 注意

- 项目**无 pytest/unittest 正式测试框架**，但有 `test_pipeline.py` 离线回归测试。
- 每个 `.py` 模块末尾都有 `if __name__ == "__main__":` 测试桩，可用于快速手动验证该模块。

---

## 代码风格与约定

- **注释与文档字符串**: 几乎全部使用**中文**。新增代码或修改时请保持中文注释风格。
- **模块组织**: 每个核心模块定义一个主类，例如 `DataLoader`、`FeatureEngineer`、`DatasetBuilder`、`QuantModel`、`StockPredictor`。
- **配置管理**: 全局参数统一放在 `config.py`，各模块通过 `import config` 读取。不要在其他模块里硬编码超参数。
- **日期格式**: 所有日期字符串使用 `YYYYMMDD`（如 `"20250101"`）。
- **错误处理**: 主流程（`main.py`）使用 `try/except` + `traceback.print_exc()` 捕获并打印异常，防止单点失败导致整个程序崩溃。
- **随机种子**: 模型训练在 `QuantModel.__init__` 中固定 `numpy` 和 `tensorflow` 随机种子（`config.RANDOM_SEED = 42`）。
- **序列化**: 模型保存为 `.h5`，标准化器和特征列保存为 `.pkl`（使用 `pickle`）。

---

## 数据流与关键设计

1. **数据源**: Tushare Pro API（需有效 Token）。
2. **目标变量**: `close.shift(-6) / open.shift(-1) - 1`，即 T+1 开盘到 T+6 收盘的五交易日收益率。
3. **时序窗口**: 输入形状为 `(SEQUENCE_LENGTH, n_features)`，默认 20 天 × 80+ 特征。
4. **数据划分**: 按日期划分训练/验证/测试，并在边界留出 `PURGE_DAYS`，严禁随机打乱。
5. **标准化**: 训练集 fit `RobustScaler`，验证集和预测集只做 transform。
6. **预测流程**: 取每只股票最近 20 天 → 做特征工程 → scaler transform → 输入模型 → 输出预测收益率 → 排序取 Top N。

---

## 已知问题与限制

1. **旧模型不兼容**: 标签和元数据规则已更新，缺少元数据的旧 DL/XGBoost 模型会被拒绝，必须重新训练。
2. **完整 walk-forward 成本高**: 默认从 2018 年开始并评估全部折，大股票池运行时间和 API/缓存占用都较高。
3. **日线回测无法可靠模拟盘中止损/止盈**: `STOP_LOSS`、`TAKE_PROFIT` 暂不进入成交逻辑；需要分钟线才能确定触发顺序。
4. **SHAP 为可选依赖**: 未安装时不影响训练、预测和回测，只跳过模型解释。
5. **未使用的配置项**: `N_JOBS`、遗留的 `MODEL_LAYERS`、`LOG_FILE`/`LOG_LEVEL` 尚未完整接入。

---

## 安全注意事项

- **Token 安全**: Token 仅允许放在环境变量或权限为 `0600` 的用户私有文件中，禁止提交到仓库。
- **数据目录**: 项目运行时会在根目录创建/写入 `./data/`、`./models/`、`./cache/`、`./recommendations_*.csv`。确保运行目录有写权限。
- ** pickle 反序列化**: 模型加载使用 `pickle.load()` 读取 `scaler.pkl` 和 `feature_cols.pkl`。如果这些文件来自不可信来源，存在反序列化安全风险。但在本项目中这些文件由自身训练流程生成。

---

## 模型产物

训练成功后会在 `./models/` 目录生成以下文件，**四者必须同时存在且匹配**才能正常预测：

| 文件 | 说明 |
|------|------|
| `quant_model.h5` | Keras 模型权重与架构 |
| `scaler.pkl` | 拟合好的 RobustScaler |
| `feature_cols.pkl` | 特征列顺序列表 |
| `model_meta.json` | 训练区间、标签口径和特征哈希 |

如果修改了 `features.py` 或 `config.py` 中的特征相关参数，必须**重新训练**，否则特征维度/顺序不匹配会导致预测失败。

---

## Agent 操作建议

- 修改代码前，先在服务器运行 `python test_pipeline.py` 和 `python test_model.py` 确认环境正常。
- 所有语法检查必须通过 SSH 在服务器执行，不要把本地映射盘的 Python 环境当作验证环境。
- 若增加新特征，需同步修改 `config.py` 中的相关参数，并重新训练模型。
- 本项目为研究原型；生产化仍需完善日志、监控、数据质量告警、分钟级成交模拟和实盘风控。
