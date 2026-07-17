# 服务器部署与 GPU 加速指南

## 一、环境准备（Linux 服务器）

### 1.1 检查 GPU 与驱动

```bash
nvidia-smi
# 确认输出类似：NVIDIA-SMI 535.xx  Driver Version: 535.xx  CUDA Version: 12.2
```

若未安装驱动，参考 [NVIDIA 官方驱动安装](https://www.nvidia.com/Download/index.aspx)。

### 1.2 安装 CUDA Toolkit 与 cuDNN

TensorFlow 2.15 对应 CUDA 12.x + cuDNN 8.9.x：

```bash
# Ubuntu/Debian 示例
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-2

# cuDNN（需注册 NVIDIA 开发者账号下载）
# 解压后复制到 CUDA 目录
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

### 1.3 安装 Python 依赖

```bash
cd quant_v4

# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate

# 安装基础依赖
pip install -r requirements.txt

# 如果服务器有 NVIDIA GPU，额外安装 GPU 版本 TensorFlow
pip install tensorflow[and-cuda]==2.15.0

# 验证 TensorFlow GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# 应输出类似：[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

# 验证 XGBoost GPU
python -c "import xgboost as xgb; print(xgb.build_info())"
# 查找 'USE_CUDA': 'ON' 表示 GPU 版已启用
```

---

## 二、GPU 配置说明

所有 GPU 相关配置集中在 `config.py`：

```python
USE_GPU = True      # True=启用 GPU, False=仅 CPU
GPU_ID = 0          # 多卡服务器可改为 1,2,3...
```

**当前已实现**：

| 模型 | GPU 加速方式 | 自动回退 |
|-----|------------|---------|
| **XGBoost** | `tree_method='gpu_hist'` | 若 GPU 不可用自动回退 `hist` |
| **TensorFlow (DL)** | 自动检测 GPU + 显存按需增长 | 无 GPU 时自动使用 CPU |

---

## 三、部署步骤

### 3.1 上传项目

```bash
# 本地压缩后上传
zip -r quant_v4.zip quant_v4/
scp quant_v4.zip user@your_server:/home/user/

# 服务器解压
ssh user@your_server
unzip quant_v4.zip
cd quant_v4
```

### 3.2 首次全量训练

```bash
# 激活环境
source venv/bin/activate

# 训练 XGBoost（全量 2021~now，约 250万样本，GPU 5~10分钟）
python main.py --mode baseline

# 或训练深度学习（GPU 显著加速）
python main.py --mode train
```

### 3.3 设置每日自动预测（cron）

```bash
crontab -e

# 添加以下行（工作日 16:35 自动执行预测）
35 16 * * 1-5 cd /home/user/quant_v4 && source venv/bin/activate && python main.py --mode predict --model xgb >> /home/user/quant_v4/logs/predict.log 2>&1

# 每周一 02:00 自动重训模型（增量数据）
0 2 * * 1 cd /home/user/quant_v4 && source venv/bin/activate && python main.py --mode baseline >> /home/user/quant_v4/logs/retrain.log 2>&1
```

创建日志目录：
```bash
mkdir -p /home/user/quant_v4/logs
```

---

## 四、内存/显存监控

训练时建议开另一个终端监控：

```bash
# 显存监控（每 2 秒刷新）
watch -n 2 nvidia-smi

# 内存监控
htop
# 或
free -h
```

**预期资源占用**：

| 阶段 | CPU 内存 | GPU 显存 | 耗时 |
|-----|---------|---------|------|
| 数据拉取 | ~2 GB | 0 | 20~40 分钟 |
| XGBoost 训练 | ~4~6 GB | ~1~2 GB | 5~10 分钟 |
| DL (CNN+LSTM+Attn) 训练 | ~3~5 GB | ~4~8 GB | 15~30 分钟 |
| 预测 | ~1 GB | ~0.5 GB | < 1 分钟 |

---

## 五、常见问题

### Q1: `nvidia-smi` 显示正常，但 TensorFlow 检测不到 GPU

```bash
# 检查 CUDA/cuDNN 版本匹配
python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info())"

# 确保 LD_LIBRARY_PATH 包含 CUDA
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Q2: XGBoost 报错 `gpu_hist` 不可用

```bash
# 重新安装带 CUDA 的 XGBoost
pip uninstall xgboost -y
pip install xgboost --force-reinstall

# 或从源码编译（性能最好）
git clone --recursive https://github.com/dmlc/xgboost
 cd xgboost
mkdir build && cd build
cmake .. -DGOOGLE_TEST=OFF -DUSE_CUDA=ON
make -j$(nproc)
cd ../python-package
pip install .
```

### Q3: 显存溢出 (OOM)

`config.py` 中已设置显存按需增长（`set_memory_growth`）。若仍溢出：
- 减小 `BATCH_SIZE`（默认 128，可改为 64）
- 减小 `SEQUENCE_LENGTH`（默认 20，可改为 10）
- 减小 `CNN_FILTERS` / `LSTM_UNITS`

### Q4: 数据拉取太慢 / Tushare 积分不足

```bash
# 检查积分
python -c "
import tushare as ts
ts.set_token('你的token')
pro = ts.pro_api()
print(pro.query('user_token'))
"

# 若积分不足，可购买 Tushare 积分包，或降低数据频率（如只取每周数据训练）
```

---

## 六、一键部署脚本（可选）

保存为 `deploy.sh`：

```bash
#!/bin/bash
set -e

PROJECT_DIR="/home/user/quant_v4"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p $LOG_DIR
cd $PROJECT_DIR

echo "=== 开始每日量化流程 ==="
echo "时间: $(date)"

# 1. 数据更新 + 预测
python main.py --mode predict --model xgb 2>&1 | tee "$LOG_DIR/predict_$(date +%Y%m%d).log"

# 2. 若今天是周一，重训模型
if [ "$(date +%u)" -eq 1 ]; then
    echo "今天是周一，执行模型重训..."
    python main.py --mode baseline 2>&1 | tee "$LOG_DIR/retrain_$(date +%Y%m%d).log"
fi

echo "=== 流程结束 ==="
```

赋予执行权限：
```bash
chmod +x deploy.sh
```
