import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
# 需要安装: pip install xgboost lightgbm catboost
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings

# --- 全局配置 (必须与原BRB实验一致) ---
GLOBAL_SEED = 42
DATA_FILE = "Mental_Health_Composite_Index.csv"  # 使用您数据融合生成的最终文件
OUTPUT_FILE = "sota_baseline_predictions.csv"    # 保存预测结果供画图用

# 忽略警告
warnings.filterwarnings('ignore')

def get_sota_models():
    """定义最强基线模型组合"""
    return {
        # 1. SVM: 代表基于距离和核函数的模型 (异构对比)
        'SVM': SVC(kernel='rbf', class_weight='balanced', C=1.0, probability=True, random_state=GLOBAL_SEED),
        
        # 2. XGBoost: 经典SOTA，极其稳健
        'XGBoost': XGBClassifier(
            n_estimators=200, 
            learning_rate=0.05, 
            max_depth=6,
            use_label_encoder=False, 
            eval_metric='mlogloss', 
            random_state=GLOBAL_SEED,
            n_jobs=-1
        ),
        
        # 3. LightGBM: 速度快，往往精度稍高
        'LightGBM': LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            class_weight='balanced',
            random_state=GLOBAL_SEED,
            n_jobs=-1,
            verbose=-1
        ),
        
        # 4. CatBoost: 对抗过拟合能力强，增加基线的多样性
        'CatBoost': CatBoostClassifier(
            iterations=200,
            learning_rate=0.05,
            depth=6,
            auto_class_weights='Balanced',
            random_seed=GLOBAL_SEED,
            verbose=0, # 静默模式
            allow_writing_files=False
        )
    }

def main():
    print("="*40)
    print("=== 开始运行 SOTA 基线对比实验 (已修复维度问题) ===")
    print(f"=== 模型阵容: SVM, XGBoost, LightGBM, CatBoost ===")
    print("="*40)

    # 1. 数据加载
    try:
        df = pd.read_csv(DATA_FILE)
        print(f"✅ 成功加载数据: {DATA_FILE}, 样本数: {len(df)}")
        
        # 提取特征和标签
        target_col = 'stress_level'
        # 排除非特征列 (根据您的文件结构，可能包含 'mental_health_history' 作为特征，但要排除 'Mental_Health_Index' 等结果列如果存在)
        # 这里假设除 stress_level 外的都是特征。如果您的CSV包含无关列，请在此处过滤。
        # 假设 'stress_level' 是最后一列，或者我们可以显式drop
        if target_col not in df.columns:
             print(f"❌ 错误: 数据中找不到目标列 '{target_col}'")
             return

        X = df.drop(columns=[target_col])
        y = df[target_col].astype(int)
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        print("请确保 'Mental_Health_Composite_Index.csv' 在当前目录下")
        return

    # 2. 10折交叉验证 (严格对齐原实验)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=GLOBAL_SEED)
    
    all_predictions = []
    fold_metrics = []

    fold = 1
    for train_idx, test_idx in skf.split(X, y):
        print(f"正在运行第 {fold}/10 折...")
        
        X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        models = get_sota_models()
        
        # 准备 fold 的预测结果字典
        # 【关键修复】：确保 true_label 是 1D 数组
        fold_preds = {
            'fold': fold,
            'true_label': y_test.values.ravel() 
        }
        
        for name, model in models.items():
            # 数据预处理：SVM 需要标准化
            if name == 'SVM':
                scaler = StandardScaler()
                X_train_fit = scaler.fit_transform(X_train_raw)
                X_test_fit = scaler.transform(X_test_raw)
            else:
                X_train_fit = X_train_raw
                X_test_fit = X_test_raw
            
            # 训练与推理
            model.fit(X_train_fit, y_train)
            y_pred = model.predict(X_test_fit)
            
            # 【关键修复】：强制展平数组，解决 (N, 1) 导致的报错
            y_pred = np.array(y_pred).ravel()
            
            # 保存预测结果
            fold_preds[f'pred_{name}'] = y_pred
            
            # 计算关键指标
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
            # 专门计算“高风险 (High Stress, Label=2)”的召回率
            _, recall_vec, _, _ = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1, 2], zero_division=0)
            high_risk_recall = recall_vec[2] if len(recall_vec) > 2 else 0
            
            fold_metrics.append({
                'Fold': fold,
                'Model': name,
                'Weighted_F1': f1,
                'High_Risk_Recall': high_risk_recall
            })
            
        # 将本折的预测结果转换为 DataFrame 并存储
        try:
            fold_df = pd.DataFrame(fold_preds)
            all_predictions.append(fold_df)
        except ValueError as e:
            print(f"❌构造DataFrame失败 (Fold {fold}): {e}")
            # 打印调试信息
            for k, v in fold_preds.items():
                if hasattr(v, 'shape'):
                    print(f"Key: {k}, Shape: {v.shape}")
            return
        
        fold += 1

    # 3. 保存最终预测文件 (用于画图)
    final_pred_df = pd.concat(all_predictions, ignore_index=True)
    
    # 映射标签为中文，与原实验格式统一
    label_map = {0: '低', 1: '中', 2: '高'}
    final_pred_df['true_label'] = final_pred_df['true_label'].map(label_map)
    for model in get_sota_models().keys():
        final_pred_df[f'pred_{model}'] = final_pred_df[f'pred_{model}'].map(label_map)
        
    final_pred_df.to_csv(OUTPUT_FILE, index=False)
    
    # 4. 打印简报
    print("\n" + "="*40)
    print("=== 实验完成！结果摘要 (Mean Performance) ===")
    print("="*40)
    metrics_df = pd.DataFrame(fold_metrics)
    summary = metrics_df.groupby('Model')[['Weighted_F1', 'High_Risk_Recall']].mean()
    print(summary)
    print("\n✅ 详细预测结果已保存至:", OUTPUT_FILE)
    print("下一步：请运行绘图脚本 plot_final_comparison.py，将此文件与 SH-BRB 结果合并。")

if __name__ == '__main__':
    main()