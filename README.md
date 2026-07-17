# A 股超短线量化交易系统 v4

这是一个面向 A 股横截面选股的研究系统。它使用 Tushare 数据构建量价、技术指标、资金流、龙虎榜和基本面特征，支持 CNN+GRU+Attention 与 XGBoost，并输出每日 Top N 候选股票。

当前唯一有效的交易时间轴是：

```text
T 日收盘数据完成 → 生成信号 → T+1 日开盘买入 → T+6 日收盘卖出
```

标签为 `close(T+6) / open(T+1) - 1`，持有五个交易日。任何修改特征、标签或股票池规则的操作都要求重新训练模型。

## 已实现的关键约束

- 训练、验证和测试按日期顺序划分，并在边界留出标签持有期，避免标签重叠泄露。
- 训练使用历史时点股票池、历史市值和前复权价格；不使用当前市值回填历史。
- 特征选择使用逐日横截面 Rank IC，而不是把所有日期混在一起计算相关性。
- 验证指标包含逐日 Rank IC、ICIR 和 Top N/分位数组合收益。
- 模型、标准化器、特征顺序和元数据必须匹配；旧模型或标签口径不一致时会拒绝预测。
- 回测使用次日开盘成交、100 股整数手、滑点、佣金、印花税、过户费和成交额容量限制。
- 普通回测会拒绝训练期覆盖回测期的前视模型；模型评估优先使用 walk-forward。
- 行业约束先分散风险；候选行业不足时会按模型排序回补，确保尽量填满 Top N。

## 环境与 Token

安装依赖：

```bash
pip install -r requirements.txt
```

Tushare Token 不写入源码。任选一种方式配置：

```bash
export TUSHARE_TOKEN="your-token"
```

或写入私有文件：

```text
~/.config/quant_v4/tushare_token
```

服务器当前使用的 Python 环境为：

```bash
/home/zhouwei/quant_v4/.conda_tf/bin/python
```

## 常用命令

所有命令都应在服务器项目目录 `/home/zhouwei/quant_v4` 中运行。

```bash
# 语法检查
/home/zhouwei/quant_v4/.conda_tf/bin/python -m py_compile *.py

# 离线测试
/home/zhouwei/quant_v4/.conda_tf/bin/python test_pipeline.py
/home/zhouwei/quant_v4/.conda_tf/bin/python test_model.py

# 训练深度学习模型
/home/zhouwei/quant_v4/.conda_tf/bin/python main.py --mode train

# 训练 XGBoost 基准模型
/home/zhouwei/quant_v4/.conda_tf/bin/python main.py --mode baseline

# 预测
/home/zhouwei/quant_v4/.conda_tf/bin/python main.py --mode predict --model dl
/home/zhouwei/quant_v4/.conda_tf/bin/python main.py --mode predict --model xgb

# 严格的历史区间回测；模型训练截止日必须早于回测起点
/home/zhouwei/quant_v4/.conda_tf/bin/python main.py --mode backtest --model dl

# 滚动训练、样本外评估
/home/zhouwei/quant_v4/.conda_tf/bin/python main.py --mode walkforward

# 收益导向模型搜索：模型、训练窗、股票池和持有期统一比较
/home/zhouwei/quant_v4/.conda_tf/bin/python main.py --mode optimize

# 小样本搜索链路冒烟；不会发布生产模型
/home/zhouwei/quant_v4/.conda_tf/bin/python main.py --mode optimize --quick --sample-size 80 --start-date 20240101

# 使用通过滚动样本外门槛的表格生产模型预测
/home/zhouwei/quant_v4/.conda_tf/bin/python main.py --mode predict --model tabular
```

快速联调可以追加 `--quick --sample-size 80 --no-advanced-data`，但快速结果不能用于实盘判断。

## 模型产物

深度学习模型需要以下四个相互匹配的文件：

```text
models/quant_model.h5
models/scaler.pkl
models/feature_cols.pkl
models/model_meta.json
```

XGBoost 需要模型文件、特征列文件和 `models/model_meta_xgb.json`。元数据记录训练截止日、标签价格口径和特征哈希，预测时会严格校验。

`optimize` 会比较 Ridge、XGBoost 和简单 MLP，并搜索24/36个月训练窗、
流动性 Top 1000/2000 股票池及5/10日持有期。候选只有在多数滚动折
Rank IC 为正且扣费后 Top 10 收益为正时，才会发布
`models/quant_model_tabular.pkl` 和 `models/model_meta_tabular.json`。

本次标签口径已经改变，因此旧模型不会继续被加载。请在正式预测前重新训练，避免把旧模型和新特征流水线混用。

## 评估原则

不要只看训练损失或单次方向准确率。至少同时检查：

- 样本外逐日 Rank IC 的均值、标准差和 ICIR；
- Top 5/10/20 扣除成本后的平均收益与胜率；
- 最大回撤、夏普、换手和容量；
- 不同年份、牛熊市和行业环境下的稳定性；
- 多个 walk-forward 折是否方向一致。

任何提高收益率的改动，都必须先在未参与训练的时间段验证。历史回测不代表未来收益，本项目不构成投资建议。
