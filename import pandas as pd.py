import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
#  计算补充基线的标准差
# --- 配置 ---
FILE_PATH = "sota_baseline_predictions.csv"  # 确保这个文件在当前目录下

def main():
    print("=== 正在计算 SOTA 基线模型的 Mean ± SD ===")
    
    try:
        df = pd.read_csv(FILE_PATH)
        
        # 获取模型列表 (排除 fold 和 true_label)
        models = [col.replace('pred_', '') for col in df.columns if col.startswith('pred_')]
        
        print(f"{'Model':<15} | {'Mean F1':<10} | {'Std Dev':<10} | {'Table Format (Copy this)'}")
        print("-" * 65)
        
        for model in models:
            fold_f1_scores = []
            
            # 遍历 10 个折，分别计算 F1
            for fold in sorted(df['fold'].unique()):
                fold_data = df[df['fold'] == fold]
                y_true = fold_data['true_label']
                y_pred = fold_data[f'pred_{model}']
                
                # 计算该折的 Weighted F1
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                fold_f1_scores.append(f1)
            
            # 计算均值和标准差
            mean_val = np.mean(fold_f1_scores)
            std_val = np.std(fold_f1_scores)
            
            # 格式化输出
            print(f"{model:<15} | {mean_val:.4f}     | {std_val:.4f}     | {mean_val:.4f} ± {std_val:.4f}")
            
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {FILE_PATH}。请确保你已经运行了之前的 baseline 实验。")

if __name__ == '__main__':
    main()