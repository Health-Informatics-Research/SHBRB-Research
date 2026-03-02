import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, f1_score, confusion_matrix, roc_curve, auc
from scipy.stats import wilcoxon

# --- 全局设置 ---
PREDICTIONS_FILE = "all_folds_predictions.csv"
WEIGHTS_FILE = "all_folds_rule_weights.csv"
MODELS = {
    'A': 'SH-BRB (完整模型)',
    'B': 'BRB-SEM (无病史调节)',
    'C': 'BRB-Heuristic (无SEM知识)',
    'D': 'Random Forest (机器学习基线)'
}

# 设置绘图风格和字体
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans'] # 尝试支持中文
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

def analyze_and_plot_rule_weights(df_weights):
    """分析并绘制最终平均规则权重图"""
    print("\n" + "="*25 + " 全局可解释性分析：最终规则权重 " + "="*25)
    
    # 筛选出我们关心的模型 (SEM-Based)
    sem_weights = df_weights[df_weights['model_type'] == 'SEM-Based'].copy()
    
    # 计算平均权重
    average_weights = sem_weights.groupby('rule_name')['weight'].mean().sort_values()
    print("各规则的平均权重值:")
    print(average_weights)
    
    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = sns.color_palette("viridis", len(average_weights))
    bars = ax.barh(average_weights.index, average_weights.values, color=colors)
    
    for bar in bars:
        ax.text(bar.get_width() * 1.01, 
                bar.get_y() + bar.get_height()/2,
                f'{bar.get_width():.3f}',
                va='center', ha='left', fontsize=10)
    
    ax.set_title('Global Interpretability: Final Average Rule Weights', fontsize=16, pad=20)
    ax.set_xlabel('Average Weight (Importance)', fontsize=12)
    ax.set_ylabel('Rule Name', fontsize=12)
    ax.set_xlim(0, average_weights.max() * 1.15)
    
    plt.tight_layout()
    plt.savefig("rule_weights_barchart.png", dpi=300)
    print("✅ 规则权重图已保存为 'rule_weights_barchart.png'")
    # plt.show() # 如果在服务器运行请注释此行

def plot_confusion_matrices(df):
    """【新增】绘制可视化的混淆矩阵热力图 (2x2布局)"""
    print("\n" + "="*25 + " 绘制混淆矩阵热力图 " + "="*25)
    labels_order = ['低', '中', '高']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, (model_code, model_name) in enumerate(MODELS.items()):
        pred_col = f'pred_{model_code}'
        ax = axes[idx]
        
        # 计算混淆矩阵
        cm = confusion_matrix(df['true_label'], df[pred_col], labels=labels_order)
        
        # 绘制热力图
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    xticklabels=labels_order, yticklabels=labels_order, 
                    annot_kws={"size": 14}, cbar=False)
        
        # 标题和标签
        # 提取模型简称，避免标题过长
        short_name = model_name.split(' ')[0] 
        ax.set_title(f'{model_code}: {short_name}', fontsize=14, fontweight='bold')
        ax.set_ylabel('真实标签 (True Label)', fontsize=12)
        ax.set_xlabel('预测标签 (Predicted Label)', fontsize=12)

    plt.tight_layout()
    plt.savefig("confusion_matrices_combined.png", dpi=300)
    print("✅ 混淆矩阵图已保存为 'confusion_matrices_combined.png'")
    # plt.show()

def plot_high_risk_roc_curve(df):
    """【新增】绘制针对'高压力'类别的ROC曲线"""
    print("\n" + "="*25 + " 绘制高风险识别 ROC 曲线 " + "="*25)
    
    plt.figure(figsize=(10, 8))
    
    # 将真实标签二值化：高=1，其他=0
    y_true_binary = (df['true_label'] == '高').astype(int)
    
    colors = {'A': '#d62728', 'B': '#ff7f0e', 'C': '#2ca02c', 'D': '#9467bd'}
    lines = {'A': '-', 'B': '--', 'C': '-.', 'D': ':'}
    widths = {'A': 2.5, 'B': 1.5, 'C': 1.5, 'D': 1.5}
    
    plotted_count = 0
    for model_code, model_name in MODELS.items():
        prob_col = f'prob_high_{model_code}'
        
        # 检查是否存在概率列
        if prob_col not in df.columns:
            print(f"⚠️ 警告: 数据中缺少 {prob_col}，无法绘制模型 {model_code} 的ROC。请确保运行了包含概率保存的新版训练脚本。")
            continue
            
        y_score = df[prob_col]
        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, 
                 color=colors[model_code], 
                 linestyle=lines[model_code],
                 linewidth=widths[model_code],
                 label=f'{model_code}: {model_name.split(" ")[0]} (AUC = {roc_auc:.3f})')
        plotted_count += 1
    
    if plotted_count > 0:
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        plt.title('ROC Curves for High-Risk Identification', fontsize=15)
        plt.legend(loc="lower right", fontsize=11, frameon=True)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("high_risk_roc_curve.png", dpi=300)
        print("✅ ROC曲线已保存为 'high_risk_roc_curve.png'")
        # plt.show()
    else:
        print("❌ 未能绘制任何ROC曲线，请检查数据文件。")

def analyze_detailed_metrics(df):
    """分析详细指标并打印LaTex表格"""
    print("\n" + "="*25 + " 详细性能指标 (Mean ± SD) " + "="*25)
    labels_order = ['低', '中', '高']
    for model_code, model_name in MODELS.items():
        pred_col = f'pred_{model_code}'
        reports = [classification_report(df[df['fold'] == fold]['true_label'], df[df['fold'] == fold][pred_col], labels=labels_order, output_dict=True, zero_division=0) for fold in sorted(df['fold'].unique())]
        
        print(f"\n--- {model_name} LaTex表格 ---")
        print("\\begin{tabular}{lccc}")
        print("\\toprule")
        print("类别 & Precision & Recall & F1-Score \\\\")
        print("\\midrule")
        for label in labels_order:
            p_vals = [r[label]['precision'] for r in reports if label in r]
            r_vals = [r[label]['recall'] for r in reports if label in r]
            f1_vals = [r[label]['f1-score'] for r in reports if label in r]
            p_str = f"${np.mean(p_vals):.3f} \\pm {np.std(p_vals):.3f}$"
            r_str = f"${np.mean(r_vals):.3f} \\pm {np.std(r_vals):.3f}$"
            f1_str = f"${np.mean(f1_vals):.3f} \\pm {np.std(f1_vals):.3f}$"
            print(f"{label} & {p_str} & {r_str} & {f1_str} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")

def perform_statistical_tests(df):
    """执行统计检验"""
    print("\n" + "="*25 + " 统计显著性检验 " + "="*25)
    f1_scores = {model_code: df.groupby('fold').apply(lambda g: f1_score(g['true_label'], g[f'pred_{model_code}'], average='weighted', zero_division=0)).tolist() for model_code in MODELS.keys()}
    
    print("--- 综合F1分数 (Mean ± SD) ---")
    for code, scores in f1_scores.items():
        print(f"模型 {code} ({MODELS[code]}): {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    print("\n--- Wilcoxon 配对检验 p-values ---")
    try:
        p_ab = wilcoxon(f1_scores['A'], f1_scores['B'])[1]
        print(f"模型A vs. 模型B (病史调节): p-value = {p_ab:.4f}{' *' if p_ab < 0.05 else ''}")
        p_ac = wilcoxon(f1_scores['A'], f1_scores['C'])[1]
        print(f"模型A vs. 模型C (SEM知识): p-value = {p_ac:.4f}{' *' if p_ac < 0.05 else ''}")
        p_ad = wilcoxon(f1_scores['A'], f1_scores['D'])[1]
        print(f"模型A vs. 模型D (Random Forest): p-value = {p_ad:.4f}{' *' if p_ad < 0.05 else ''}")
    except ValueError as e:
        print(f"无法执行检验 (可能是样本数不足): {e}")
    print("\n* 表示差异具有统计显著性 (p < 0.05)")

if __name__ == "__main__":
    try:
        # 1. 绘制规则权重图
        if pd.io.common.file_exists(WEIGHTS_FILE):
            weights_df = pd.read_csv(WEIGHTS_FILE)
            analyze_and_plot_rule_weights(weights_df)
        else:
            print(f"跳过规则权重图 (未找到 {WEIGHTS_FILE})")

        # 2. 加载预测结果
        if pd.io.common.file_exists(PREDICTIONS_FILE):
            results_df = pd.read_csv(PREDICTIONS_FILE)
            print(f"\n成功加载预测结果文件: {PREDICTIONS_FILE}, 共 {len(results_df)} 条样本")
            
            # --- 新增的可视化功能 ---
            plot_high_risk_roc_curve(results_df)   # 画 ROC 曲线
            plot_confusion_matrices(results_df)    # 画 混淆矩阵热力图
            
            # --- 原有的分析功能 ---
            analyze_detailed_metrics(results_df)   # 打印 LaTeX 表格
            perform_statistical_tests(results_df)  # 统计检验
        else:
            print(f"错误: 未找到结果文件: {PREDICTIONS_FILE}")
            print("请先运行 final_evaluation.py (含概率保存版) 来生成结果。")
            
    except Exception as e:
        print(f"发生未预期的错误: {str(e)}")
        import traceback
        traceback.print_exc()