"""
基准模型模块
功能：XGBoost（主力） + Ridge/ElasticNet（线性基准）
支持滚动训练、特征重要性、SHAP 解释、与深度学习对比
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error
)
import xgboost as xgb
import config
from scipy import stats

# 尝试导入 shap，若未安装则跳过
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("警告: shap 未安装，跳过 SHAP 解释功能")


def _daily_regression_metrics(y_pred, y_true, trade_dates):
    """按交易日计算横截面 Rank IC、Top N 和扣费后收益。"""
    frame = pd.DataFrame({
        'prediction': np.asarray(y_pred, dtype=float),
        'target': np.asarray(y_true, dtype=float),
        'trade_date': np.asarray(trade_dates),
    }).dropna()
    ic_values = []
    quantile_returns = []
    top_n_list = [int(n) for n in getattr(config, 'EVAL_TOP_N_LIST', [5, 10, 20])]
    top_n_returns = {top_n: [] for top_n in top_n_list}
    for _, day in frame.groupby('trade_date'):
        if len(day) < 10:
            continue
        if day['prediction'].std() > 1e-12 and day['target'].std() > 1e-12:
            ic, _ = stats.spearmanr(day['prediction'], day['target'])
            if np.isfinite(ic):
                ic_values.append(float(ic))
        order = np.argsort(day['prediction'].to_numpy())
        quantiles = np.array_split(order, 5)
        if all(len(idx) > 0 for idx in quantiles):
            target = day['target'].to_numpy()
            quantile_returns.append([
                float(target[idx].mean()) for idx in quantiles
            ])
        target = day['target'].to_numpy()
        for top_n in top_n_list:
            n = min(top_n, len(order))
            if n > 0:
                top_n_returns[top_n].append(float(target[order[-n:]].mean()))

    result = {
        'rank_ic': float(np.mean(ic_values)) if ic_values else np.nan,
        'rank_ic_std': float(np.std(ic_values)) if ic_values else np.nan,
    }
    if quantile_returns:
        means = np.mean(np.asarray(quantile_returns), axis=0)
        result['top_quantile_return'] = float(means[-1])
        result['bottom_quantile_return'] = float(means[0])
        result['quantile_returns'] = [float(value) for value in means]
        result['long_short_return'] = float(means[-1] - means[0])
    else:
        result.update({
            'top_quantile_return': np.nan,
            'bottom_quantile_return': np.nan,
            'quantile_returns': [],
            'long_short_return': np.nan,
        })
    round_trip_cost = (
        2 * config.BACKTEST_COMMISSION
        + 2 * config.BACKTEST_TRANSFER_FEE
        + 2 * config.BACKTEST_SLIPPAGE
        + config.BACKTEST_STAMP_DUTY
    )
    for top_n, values in top_n_returns.items():
        mean_return = float(np.mean(values)) if values else np.nan
        return_std = float(np.std(values)) if values else np.nan
        return_se = (
            return_std / np.sqrt(len(values)) if len(values) > 1 else np.nan
        )
        result[f'top_{top_n}_return'] = mean_return
        result[f'top_{top_n}_return_std'] = return_std
        result[f'top_{top_n}_return_se'] = return_se
        result[f'top_{top_n}_net_return'] = mean_return - round_trip_cost
        result[f'top_{top_n}_net_positive_ratio'] = (
            float(np.mean(np.asarray(values) > round_trip_cost)) if values else np.nan
        )
    selection_top_n = int(getattr(config, 'SELECTION_TOP_N', 10))
    result['selection_score'] = result.get(
        f'top_{selection_top_n}_net_return', np.nan
    ) - float(getattr(config, 'VALIDATION_RETURN_SE_PENALTY', 0.0)) * result.get(
        f'top_{selection_top_n}_return_se', 0.0
    )
    result['round_trip_cost'] = float(round_trip_cost)
    return result


class XGBoostModel:
    """XGBoost 模型（回归/分类自适应）"""

    def __init__(self, params=None):
        params = (params or {}).copy()
        base = config.XGB_PARAMS_BASE.copy()
        base.update(params)
        params = base

        # 根据目标类型设置 objective 和 eval_metric
        self.target_type = getattr(config, 'TARGET_TYPE', 'classification')
        if self.target_type == 'regression':
            params['objective'] = 'reg:squarederror'
            params['eval_metric'] = 'rmse'
        else:
            params['objective'] = 'binary:logistic'
            params['eval_metric'] = 'auc'

        # GPU 设置
        if config.USE_GPU:
            # XGBoost 2.x 使用 tree_method='hist' + device='cuda'
            params['tree_method'] = 'hist'
            params['device'] = 'cuda'
            print("XGBoost 尝试使用 GPU 加速 (device=cuda)，若不支持将自动回退 CPU")
        else:
            params['tree_method'] = 'hist'
            print("XGBoost 使用 CPU (hist)")
        self.params = params
        self.model = None
        self.feature_names = None
        self.feature_importance = None

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            early_stopping_rounds=None, verbose=False):
        """
        训练模型

        Args:
            X_train: ndarray/DataFrame
            y_train: ndarray/Series
            X_val: 验证特征（用于早停）
            y_val: 验证标签
        """
        if early_stopping_rounds is None:
            early_stopping_rounds = config.XGB_EARLY_STOPPING_ROUNDS

        self.feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else None

        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        # XGBoost 3.x 要求把 early_stopping_rounds 放在估计器参数中。
        fit_ok = False
        use_early_stopping = (
            early_stopping_rounds is not None
            and X_val is not None and y_val is not None
        )
        for attempt in range(4):
            model_params = self.params.copy()
            if use_early_stopping:
                model_params['early_stopping_rounds'] = early_stopping_rounds
            if self.target_type == 'regression':
                self.model = xgb.XGBRegressor(**model_params)
            else:
                self.model = xgb.XGBClassifier(**model_params)

            try:
                fit_kwargs = {
                    'eval_set': eval_set,
                    'verbose': verbose
                }
                self.model.fit(X_train, y_train, **fit_kwargs)
                fit_ok = True
                break
            except Exception as e:
                err_str = str(e)
                if any(key in err_str.lower() for key in ['gpu', 'cuda', 'device']):
                    # GPU 不可用，回退到 CPU
                    self.params['tree_method'] = 'hist'
                    self.params.pop('device', None)
                    self.params.pop('predictor', None)
                    print("GPU 不可用，XGBoost 回退到 CPU (hist)")
                    continue
                else:
                    raise

        if not fit_ok:
            raise RuntimeError("XGBoost 训练失败")

        # 记录特征重要性
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names or [f'f{i}' for i in range(X_train.shape[1])],
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

    def predict(self, X):
        """预测：回归=连续收益，分类=上涨概率"""
        if self.model is None:
            raise ValueError("模型未训练")
        predict_device = getattr(config, 'XGB_PREDICT_DEVICE', 'cpu')
        if predict_device:
            self.model.set_params(device=predict_device)
        if self.target_type == 'regression':
            return self.model.predict(X)
        else:
            return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test, y_test, trade_dates=None, raw_returns=None):
        """
        评估模型

        Returns:
            dict: 指标
        """
        y_pred = self.predict(X_test)

        if self.target_type == 'regression':
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            }
            if trade_dates is not None:
                metrics.update(
                    _daily_regression_metrics(
                        y_pred,
                        y_test if raw_returns is None else raw_returns,
                        trade_dates
                    )
                )
            else:
                valid = ~(np.isnan(y_pred) | np.isnan(y_test))
                rank_ic = np.nan
                if valid.sum() > 10:
                    rank_ic, _ = stats.spearmanr(y_pred[valid], y_test[valid])
                quantiles = np.array_split(np.argsort(y_pred), 5)
                quantile_returns = [float(y_test[q].mean()) for q in quantiles]
                metrics.update({
                    'rank_ic': rank_ic,
                    'top_quantile_return': quantile_returns[-1],
                    'bottom_quantile_return': quantile_returns[0],
                    'quantile_returns': quantile_returns,
                    'long_short_return': quantile_returns[-1] - quantile_returns[0],
                })
        else:
            y_binary = (y_pred > 0.5).astype(int)
            metrics = {
                'accuracy': accuracy_score(y_test, y_binary),
                'auc': roc_auc_score(y_test, y_pred) if len(np.unique(y_test)) > 1 else np.nan,
                'precision': precision_score(y_test, y_binary, zero_division=0),
                'recall': recall_score(y_test, y_binary, zero_division=0),
                'f1': f1_score(y_test, y_binary, zero_division=0),
            }
            n_quantiles = 5
            quantiles = np.array_split(np.argsort(y_pred), n_quantiles)
            quantile_hit_rates = [float(y_test[q].mean()) for q in quantiles]
            metrics['top_quantile_hit_rate'] = quantile_hit_rates[-1]
            metrics['bottom_quantile_hit_rate'] = quantile_hit_rates[0]

        return metrics

    def shap_summary(self, X_sample, max_display=20):
        """
        SHAP 特征重要性解释

        Args:
            X_sample: 用于计算 SHAP 的样本（建议 100~500 条）
            max_display: 展示前 N 个特征

        Returns:
            shap_values 或 None
        """
        if not HAS_SHAP:
            print("SHAP 未安装，跳过")
            return None
        if self.model is None:
            raise ValueError("模型未训练")

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)

        # 保存摘要图（如需可视化可调用 shap.summary_plot）
        return shap_values

    def save(self, model_file=None, feature_file=None):
        """保存模型"""
        if model_file is None:
            model_file = config.MODEL_FILE.replace('.h5', '_xgb.pkl')
        if feature_file is None:
            feature_file = config.FEATURE_COLS_FILE.replace('.pkl', '_xgb.pkl')

        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)
        with open(feature_file, 'wb') as f:
            pickle.dump(self.feature_names, f)

        print(f"XGBoost 模型已保存: {model_file}")
        print(f"特征列已保存: {feature_file}")

    def load(self, model_file=None, feature_file=None):
        """加载模型"""
        if model_file is None:
            model_file = config.MODEL_FILE.replace('.h5', '_xgb.pkl')
        if feature_file is None:
            feature_file = config.FEATURE_COLS_FILE.replace('.pkl', '_xgb.pkl')

        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
        with open(feature_file, 'rb') as f:
            self.feature_names = pickle.load(f)

        print("XGBoost 模型已加载")


class LinearModel:
    """线性基准模型：Ridge / ElasticNet / Logistic"""

    def __init__(self, model_type='ridge', alpha=1.0, l1_ratio=0.5):
        """
        Args:
            model_type: 'ridge', 'elasticnet', 'logistic'
            alpha: 正则化强度
            l1_ratio: ElasticNet 的 L1 比例
        """
        self.model_type = model_type
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.model = None
        self.feature_names = None
        self.coef_df = None

    def fit(self, X_train, y_train):
        """训练线性模型"""
        self.feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else None
        self.target_type = getattr(config, 'TARGET_TYPE', 'classification')

        # 回归模式下不使用 Logistic
        if self.target_type == 'regression' and self.model_type == 'logistic':
            print("警告: 回归目标下 Logistic 不可用，自动切换为 Ridge")
            self.model_type = 'ridge'

        if self.model_type == 'ridge':
            estimator = Ridge(alpha=self.alpha)
        elif self.model_type == 'elasticnet':
            estimator = ElasticNet(
                alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=5000
            )
        elif self.model_type == 'logistic':
            estimator = LogisticRegression(
                penalty='l2', C=1.0 / self.alpha, max_iter=1000,
                random_state=config.RANDOM_SEED, solver='lbfgs'
            )
        else:
            raise ValueError(f"未知模型类型: {self.model_type}")

        self.model = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', RobustScaler()),
            ('model', estimator),
        ])
        self.model.fit(X_train, y_train)

        # 记录系数
        if hasattr(self.model, 'coef_'):
            coefs = self.model.coef_
            if coefs.ndim > 1:
                coefs = coefs.flatten()
            self.coef_df = pd.DataFrame({
                'feature': self.feature_names or [f'f{i}' for i in range(len(coefs))],
                'coef': coefs,
                'abs_coef': np.abs(coefs)
            }).sort_values('abs_coef', ascending=False)

    def predict(self, X):
        """预测：回归=连续值，分类=概率"""
        if self.model is None:
            raise ValueError("模型未训练")
        if self.model_type == 'logistic':
            return self.model.predict_proba(X)[:, 1]
        else:
            # Ridge/ElasticNet 直接输出连续值（回归模式下不做 sigmoid）
            return self.model.predict(X)

    def evaluate(self, X_test, y_test, trade_dates=None, raw_returns=None):
        """评估"""
        y_pred = self.predict(X_test)
        target_type = getattr(self, 'target_type', getattr(config, 'TARGET_TYPE', 'classification'))

        if target_type == 'regression':
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            }
            if trade_dates is not None:
                metrics.update(
                    _daily_regression_metrics(
                        y_pred,
                        y_test if raw_returns is None else raw_returns,
                        trade_dates
                    )
                )
            else:
                valid = ~(np.isnan(y_pred) | np.isnan(y_test))
                rank_ic = np.nan
                if valid.sum() > 10:
                    rank_ic, _ = stats.spearmanr(y_pred[valid], y_test[valid])
                quantiles = np.array_split(np.argsort(y_pred), 5)
                quantile_returns = [float(y_test[q].mean()) for q in quantiles]
                metrics.update({
                    'rank_ic': rank_ic,
                    'top_quantile_return': quantile_returns[-1],
                    'bottom_quantile_return': quantile_returns[0],
                    'long_short_return': quantile_returns[-1] - quantile_returns[0],
                })
        else:
            y_binary = (y_pred > 0.5).astype(int)
            metrics = {
                'accuracy': accuracy_score(y_test, y_binary),
                'auc': roc_auc_score(y_test, y_pred) if len(np.unique(y_test)) > 1 else np.nan,
                'precision': precision_score(y_test, y_binary, zero_division=0),
                'recall': recall_score(y_test, y_binary, zero_division=0),
                'f1': f1_score(y_test, y_binary, zero_division=0),
            }
            n_quantiles = 5
            quantiles = np.array_split(np.argsort(y_pred), n_quantiles)
            quantile_hit_rates = [float(y_test[q].mean()) for q in quantiles]
            metrics['top_quantile_hit_rate'] = quantile_hit_rates[-1]
            metrics['bottom_quantile_hit_rate'] = quantile_hit_rates[0]

        return metrics

    def save(self, model_file=None):
        """保存"""
        if model_file is None:
            suffix = f"_{self.model_type}.pkl"
            model_file = config.MODEL_FILE.replace('.h5', suffix)
        with open(model_file, 'wb') as f:
            pickle.dump({'model': self.model, 'feature_names': self.feature_names}, f)
        print(f"线性模型已保存: {model_file}")

    def load(self, model_file=None):
        """加载"""
        if model_file is None:
            suffix = f"_{self.model_type}.pkl"
            model_file = config.MODEL_FILE.replace('.h5', suffix)
        with open(model_file, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.feature_names = data['feature_names']
        print("线性模型已加载")


class ModelComparator:
    """模型对比器：在同一样本外数据上对比多个模型"""

    def __init__(self):
        self.results = {}

    def add_result(self, model_name, metrics_dict):
        """添加模型结果"""
        self.results[model_name] = metrics_dict

    def compare(self):
        """
        对比所有模型

        Returns:
            DataFrame: 对比表格
        """
        if not self.results:
            return pd.DataFrame()

        df = pd.DataFrame(self.results).T
        target_type = getattr(config, 'TARGET_TYPE', 'classification')
        if target_type == 'regression':
            selection_col = f"top_{int(getattr(config, 'SELECTION_TOP_N', 10))}_net_return"
            if selection_col in df.columns:
                df = df.sort_values(selection_col, ascending=False)
            elif 'rank_ic' in df.columns:
                df = df.sort_values('rank_ic', ascending=False)
            elif 'long_short_return' in df.columns:
                df = df.sort_values('long_short_return', ascending=False)
        else:
            if 'auc' in df.columns:
                df = df.sort_values('auc', ascending=False)
        return df

    def print_comparison(self):
        """打印对比结果"""
        df = self.compare()
        if df.empty:
            print("无对比数据")
            return

        print("\n" + "=" * 80)
        print("模型对比报告（样本外）")
        print("=" * 80)
        print(df.to_string())
        print("=" * 80)

        target_type = getattr(config, 'TARGET_TYPE', 'classification')
        if target_type == 'regression':
            selection_col = f"top_{int(getattr(config, 'SELECTION_TOP_N', 10))}_net_return"
            if selection_col in df.columns and df[selection_col].notna().any():
                best_net = df[selection_col].idxmax()
                print(
                    f"\n扣费后收益最高: {best_net}, "
                    f"{selection_col} = {df.loc[best_net, selection_col]:.4f}"
                )
            if 'rank_ic' in df.columns and df['rank_ic'].notna().any():
                best = df['rank_ic'].idxmax()
                print(f"\n最佳模型（按 Rank IC）: {best}, Rank IC = {df.loc[best, 'rank_ic']:.4f}")
            if 'long_short_return' in df.columns and df['long_short_return'].notna().any():
                best_ls = df['long_short_return'].idxmax()
                print(f"多空收益最高: {best_ls}, 收益 = {df.loc[best_ls, 'long_short_return']:.4f}")
        else:
            if 'auc' in df.columns:
                best = df['auc'].idxmax()
                print(f"\n最佳模型（按 AUC）: {best}, AUC = {df.loc[best, 'auc']:.4f}")
            if 'top_quantile_hit_rate' in df.columns:
                best_top = df['top_quantile_hit_rate'].idxmax()
                print(f"Top 分层命中率最高: {best_top}, 命中率 = {df.loc[best_top, 'top_quantile_hit_rate']:.4f}")


def train_and_compare(X_train, y_train, X_val, y_val, X_test, y_test,
                      feature_names, models_to_train=None,
                      test_trade_dates=None, test_raw_returns=None):
    """
    便捷函数：训练多个模型并对比

    Args:
        X_train, y_train: 训练集
        X_val, y_val: 验证集（XGB 早停用）
        X_test, y_test: 测试集（样本外对比用）
        feature_names: 特征名列表
        models_to_train: 模型列表，默认 ['xgb', 'ridge', 'logistic']

    Returns:
        dict: {model_name: model_instance}
    """
    if models_to_train is None:
        models_to_train = ['xgb', 'ridge', 'logistic']

    comparator = ModelComparator()
    trained_models = {}

    target_type = getattr(config, 'TARGET_TYPE', 'classification')

    # XGBoost
    if 'xgb' in models_to_train:
        print("\n训练 XGBoost...")
        xgb_model = XGBoostModel()
        xgb_model.fit(X_train, y_train, X_val, y_val, verbose=False)
        if xgb_model.feature_names is None:
            xgb_model.feature_names = list(feature_names)
            xgb_model.feature_importance['feature'] = xgb_model.feature_names
        metrics = xgb_model.evaluate(
            X_test, y_test, trade_dates=test_trade_dates,
            raw_returns=test_raw_returns
        )
        comparator.add_result('XGBoost', metrics)
        trained_models['xgb'] = xgb_model
        if target_type == 'regression':
            print(f"  RMSE: {metrics['rmse']:.4f}, Rank IC: {metrics.get('rank_ic', np.nan):.4f}, "
                  f"多空收益: {metrics.get('long_short_return', np.nan):.4f}")
        else:
            print(f"  AUC: {metrics['auc']:.4f}, Top分层命中率: {metrics['top_quantile_hit_rate']:.4f}")
        print(f"  重要特征 Top 5:\n{xgb_model.feature_importance.head(5).to_string(index=False)}")

    # Ridge
    if 'ridge' in models_to_train:
        print("\n训练 Ridge...")
        ridge_model = LinearModel(model_type='ridge', alpha=1.0)
        ridge_model.fit(X_train, y_train)
        metrics = ridge_model.evaluate(
            X_test, y_test, trade_dates=test_trade_dates,
            raw_returns=test_raw_returns
        )
        comparator.add_result('Ridge', metrics)
        trained_models['ridge'] = ridge_model
        if target_type == 'regression':
            print(f"  RMSE: {metrics['rmse']:.4f}, Rank IC: {metrics.get('rank_ic', np.nan):.4f}")
        else:
            print(f"  AUC: {metrics['auc']:.4f}, Top分层命中率: {metrics['top_quantile_hit_rate']:.4f}")

    # Logistic（仅分类模式）
    if 'logistic' in models_to_train and target_type == 'classification':
        print("\n训练 Logistic...")
        log_model = LinearModel(model_type='logistic', alpha=1.0)
        log_model.fit(X_train, y_train)
        metrics = log_model.evaluate(X_test, y_test)
        comparator.add_result('Logistic', metrics)
        trained_models['logistic'] = log_model
        print(f"  AUC: {metrics['auc']:.4f}, Top分层命中率: {metrics['top_quantile_hit_rate']:.4f}")

    comparator.print_comparison()
    return trained_models, comparator


def main():
    """测试基准模型"""
    from data_loader import DataLoader
    from features import FeatureEngineer
    from dataset import DatasetBuilder

    print("测试基准模型模块...")
    loader = DataLoader()
    engineer = FeatureEngineer()
    builder = DatasetBuilder()

    trade_date = loader.get_latest_trade_date()
    stock_list = loader.get_stock_list(trade_date).head(50)

    stock_data = {}
    for _, row in stock_list.iterrows():
        ts_code = row['ts_code']
        df = loader.get_stock_daily(ts_code)
        if df is not None and len(df) >= config.MIN_HISTORY_DAYS:
            stock_data[ts_code] = df

    if not stock_data:
        print("无数据")
        return

    # 构建数据（保留原有流水线）
    dataset = builder.build_dataset(stock_data)
    train_df, val_df = builder.split_dataset(dataset)
    feature_cols = builder.get_feature_columns(train_df)

    # 展平时序为截面样本（XGBoost 不用时序窗口）
    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['target'].values

    # 用验证集模拟测试集（实际应用时用独立测试期）
    X_test, y_test = X_val, y_val

    train_and_compare(
        X_train, y_train, X_val, y_val, X_test, y_test,
        feature_cols, models_to_train=['xgb', 'ridge', 'logistic']
    )


if __name__ == "__main__":
    main()
