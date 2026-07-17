# A 股超短线量化交易系统 v4.0 (CNN+LSTM+Attention)

基于 **CNN + LSTM/GRU + Multi-Head Attention** 混合架构的 A 股超短线量化交易系统，专为 A 股市场的高噪声、微弱信号、非平稳、T+1约束等特点设计。

## 🎯 核心升级

### 模型架构升级
从简单的 DNN 模型升级为 **CNN + LSTM/GRU + Multi-Head Attention** 混合架构：

1. **CNN 层**（1D 卷积）
   - 提取局部特征模式（如连续上涨/下跌、量价配合等）
   - 多层卷积 + Batch Normalization + Spatial Dropout
   - 适合捕捉短期技术形态

2. **LSTM/GRU 层**（可选双向）
   - 捕捉长期时序依赖关系
   - 记忆历史信息，理解趋势变化
   - 双向结构能同时考虑过去和"未来"（序列内）的上下文

3. **Multi-Head Self-Attention**
   - 自动学习重要特征和时间点的权重
   - 多头机制捕捉不同维度的注意力模式
   - 残差连接 + Layer Normalization

4. **全连接层**
   - 整合所有提取的特征
   - Batch Normalization + Dropout 防止过拟合

### 损失函数升级
自定义组合损失函数：

```python
Total Loss = MSE Loss + Direction Loss + Ranking Loss
```

- **MSE Loss**: 预测准确性
- **Direction Loss**: 方向一致性（涨跌方向正确更重要）
- **Ranking Loss**: 排序正确性（确保高收益股票排在前面）

### 时序窗口
- 使用过去 **20 天**的数据作为输入序列
- 预测未来第 1 天的涨跌幅
- 严格时序划分，避免数据泄露

## 📊 模型架构图

```
输入 (20天 × N特征)
    ↓
┌─────────────────────┐
│   CNN 模块           │
│  - Conv1D × 3       │
│  - BatchNorm        │
│  - SpatialDropout   │
└─────────────────────┘
    ↓
┌─────────────────────┐
│   RNN 模块           │
│  - Bi-LSTM/GRU × 2  │
│  - BatchNorm        │
│  - Dropout          │
└─────────────────────┘
    ↓
┌─────────────────────┐
│   Attention 模块     │
│  - Multi-Head Attn  │
│  - Layer Norm       │
│  - GlobalAvgPool    │
└─────────────────────┘
    ↓
┌─────────────────────┐
│   全连接层           │
│  - Dense × 3        │
│  - BatchNorm        │
│  - Dropout          │
└─────────────────────┘
    ↓
输出（次日涨跌幅预测）
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- TensorFlow >= 2.15.0
- pandas, numpy
- scikit-learn
- tushare

### 2. 配置参数

通过环境变量或用户私有文件配置 Token；模型参数仍在 `config.py` 中配置：

```bash
export TUSHARE_TOKEN="your_token_here"
```

```python
# 模型架构
MODEL_TYPE = "cnn_lstm_attention"  # 或 "dnn"
SEQUENCE_LENGTH = 20  # 时序窗口长度

# CNN 参数
CNN_FILTERS = [64, 128, 128]
CNN_KERNEL_SIZES = [3, 3, 3]

# LSTM 参数
LSTM_UNITS = [128, 64]
USE_LSTM = True  # True=LSTM, False=GRU
BIDIRECTIONAL = True  # 双向RNN

# Attention 参数
USE_ATTENTION = True
NUM_ATTENTION_HEADS = 8
ATTENTION_KEY_DIM = 32

# 训练参数
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.0005
```

### 3. 训练模型

```bash
python main.py --mode train
```

训练过程包括：
- 获取历史数据（默认从 2020-01-01 开始）
- 特征工程（80+ 维特征）
- 构建时序窗口（20天序列）
- 数据标准化（RobustScaler）
- 模型训练（早停 + 学习率衰减）

### 4. 预测选股

```bash
python main.py --mode predict
```

输出示例：
```
==================================================
          今日推荐买入股票 (Top 10)
==================================================

股票代码     股票名称   行业      当前价格   预测涨幅
000001.SZ   平安银行   银行      12.50     3.45%
600036.SH   招商银行   银行      35.20     3.12%
...
```

### 5. 回测验证

```bash
python main.py --mode backtest
```

## 🔧 关键技术细节

### 1. 时序窗口构建

```python
# 对每只股票，使用滑动窗口
for i in range(len(data) - SEQUENCE_LENGTH + 1):
    X = data[i:i+SEQUENCE_LENGTH]  # 过去20天
    # 标签由 dataset.py 使用下一交易日开盘和第六交易日收盘构造
    y = data[i+SEQUENCE_LENGTH-1]['target_return']
```

### 2. 数据标准化

使用 **RobustScaler** 而非 StandardScaler：
- 基于中位数和四分位数
- 对异常值更鲁棒
- 更适合金融数据

### 3. 模型正则化

多层正则化防止过拟合：
- Batch Normalization（加速训练，稳定梯度）
- Dropout（全连接层）
- Spatial Dropout（CNN/RNN 层）
- L2 正则化（可选）

### 4. 评估指标

除了常规指标外，关注：
- **Direction Accuracy**: 方向准确率
- **IC (Information Coefficient)**: 预测值与真实值相关性
- **Quantile Returns**: 分位数收益（检查是否能区分高低收益股票）
- **Top Quantile Return**: 预测最高的 20% 平均收益

## 📁 项目结构

```
./
├── config.py          # 配置（新增 CNN/LSTM/Attention 参数）
├── data_loader.py     # 数据获取（无变化）
├── features.py        # 特征工程（无变化）
├── dataset.py         # 数据集构建（支持时序窗口）⭐
├── model.py           # CNN+LSTM+Attention 模型 ⭐⭐⭐
├── predictor.py       # 预测选股（适配时序输入）
├── main.py            # 程序入口（更新训练流程）
├── requirements.txt   # 依赖库
└── README.md          # 说明文档
```

## 🎓 模型优势

### 针对 A 股特点的设计

1. **高噪声**
   - RobustScaler 对异常值鲁棒
   - 多层正则化防止过拟合
   - Ranking Loss 关注相对排序而非绝对值

2. **微弱信号**
   - CNN 提取局部模式
   - Attention 自动学习重要特征
   - 组合损失函数强化信号

3. **非平稳**
   - LSTM/GRU 适应时变特性
   - 时序窗口捕捉动态变化
   - RobustScaler 减少分布偏移影响

4. **T+1 约束**
   - 严格时序划分
   - 只使用前一日数据预测
   - T+1 开盘到 T+6 收盘的五交易日收益率作为目标

## 📈 性能对比

| 模型 | 方向准确率 | IC | Top 20% 收益 |
|------|-----------|-----|-------------|
| 简单 DNN | ~52% | 0.05 | 1.2% |
| CNN+LSTM | ~55% | 0.08 | 1.8% |
| CNN+LSTM+Attention | ~58% | 0.12 | 2.3% |

*注：以上为示例数据，实际效果取决于训练数据和市场环境*

## ⚙️ 模型调优建议

### 1. 时序窗口长度
```python
SEQUENCE_LENGTH = 20  # 可尝试 10, 15, 20, 30
```
- 更短：响应更快，但信息不足
- 更长：信息更丰富，但计算量大

### 2. CNN 配置
```python
CNN_FILTERS = [64, 128, 128]  # 增加过滤器数量提取更多特征
CNN_KERNEL_SIZES = [3, 3, 3]  # 更大的卷积核捕捉更长的模式
```

### 3. LSTM vs GRU
```python
USE_LSTM = True  # LSTM：更强大但更慢
USE_LSTM = False  # GRU：更快但稍弱
```

### 4. Attention 头数
```python
NUM_ATTENTION_HEADS = 8  # 可尝试 4, 8, 16
```
- 更多头：捕捉更多模式，但计算量大
- 更少头：更快，但可能错过某些模式

### 5. 批次大小
```python
BATCH_SIZE = 64  # 时序模型建议使用较小 batch
```
- 更小：梯度更新更频繁，更稳定
- 更大：训练更快，但可能不稳定

## 🔬 进阶功能

### 1. 特征重要性分析

使用 Attention 权重分析哪些特征最重要：

```python
# 从模型中提取 Attention 层
attention_layer = model.get_layer('multi_head_attention')
# 可视化注意力权重
```

### 2. 模型集成

训练多个模型并集成：

```python
# 不同随机种子
# 不同时序窗口
# 不同架构
# 投票或加权平均
```

### 3. 在线学习

定期使用最新数据重新训练：

```python
# 每周重新训练
# 增量学习（Fine-tuning）
```

## ⚠️ 注意事项

1. **数据质量**
   - Tushare 数据可能有缺失或错误
   - 建议数据清洗和验证

2. **过拟合风险**
   - 模型复杂度高，容易过拟合
   - 必须使用验证集监控
   - 早停机制很重要

3. **计算资源**
   - CNN+LSTM+Attention 训练较慢
   - 建议使用 GPU（提速 5-10 倍）
   - 内存需求较大

4. **市场环境**
   - 模型在不同市场环境表现差异大
   - 建议分市场环境训练
   - 定期评估和更新

## 📚 参考文献

1. **Temporal Convolutional Networks** (TCN)
2. **Attention Is All You Need** (Transformer)
3. **Financial Time Series Forecasting with Deep Learning**
4. **Ranking Loss for Stock Prediction**

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

## ⚖️ 免责声明

本项目仅供学习研究使用，不构成任何投资建议。股市有风险，投资需谨慎。使用本系统产生的任何投资损失，作者不承担任何责任。

---

**祝您交易顺利！** 📈✨
