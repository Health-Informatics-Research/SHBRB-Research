import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support

# --- 配置 ---
BRB_FILE = "all_folds_predictions.csv"  # 包含 SH-BRB(pred_A) 和 Random Forest(pred_D)
BASELINE_FILE = "sota_baseline_predictions.csv" # 包含 4个 SOTA 基线

def main():
    print("=== 开始生成最终学术级对比图表 (还原横向图4-1) ===")
    
    # 1. 读取并合并数据
    try:
        df_brb = pd.read_csv(BRB_FILE) 
        df_base = pd.read_csv(BASELINE_FILE) 
        
        merged_df = df_brb[['fold', 'true_label', 'pred_A', 'pred_D']].copy()
        merged_df.rename(columns={
            'pred_A': 'SH-BRB',
            'pred_D': 'Random Forest'
        }, inplace=True)
        
        baseline_cols = [c for c in df_base.columns if c.startswith('pred_')]
        for col in baseline_cols:
            model_name = col.replace('pred_', '')
            merged_df[model_name] = df_base[col]
            
        model_list = [c for c in merged_df.columns if c not in ['fold', 'true_label']]
        
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        return

    # 2. 按折 (Fold) 计算指标
    labels_order = ['低', '中', '高'] 
    f1_records = [] 
    hr_records = []
    
    for fold_id, group in merged_df.groupby('fold'):
        y_true = group['true_label']
        for model in model_list:
            y_pred = group[model]
            
            p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
            f1_records.append({'Fold': fold_id, 'Model': model, 'Score': f1, 'Metric': 'Weighted F1'})
            
            p_vec, r_vec, f1_vec, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels_order, zero_division=0)
            hr_records.append({'Fold': fold_id, 'Model': model, 'Score': p_vec[2], 'Metric': 'High-Risk Precision'})
            hr_records.append({'Fold': fold_id, 'Model': model, 'Score': r_vec[2], 'Metric': 'High-Risk Recall'})

    df_f1 = pd.DataFrame(f1_records)
    df_hr = pd.DataFrame(hr_records)

    sns.set_theme(style="whitegrid", font='SimHei', font_scale=1.1) 
    plt.rcParams['axes.unicode_minus'] = False
    
    mean_f1 = df_f1.groupby('Model')['Score'].mean().sort_values(ascending=False)
    sorted_models = mean_f1.index.tolist()
    std_f1 = df_f1.groupby('Model')['Score'].std()

    # ========================================================
    # 图 4-1: 综合 F1 对比 (横向还原版 + 误差棒 + ns)
    # ========================================================
    plt.figure(figsize=(10, 7))
    
    # 还原你原本的横向画法 (y="Model", x="Score") 和 magma 配色
    ax1 = sns.barplot(
        x="Score", 
        y="Model", 
        data=df_f1, 
        order=sorted_models,
        palette="magma",
        capsize=0.1,         # 垂直误差棒的帽宽
        errorbar='sd',       # 标准差误差棒
        errcolor="#333333",  # 误差棒颜色稍微柔和一点
        errwidth=1.5
    )
    
    # 设置合理的 X 轴范围，左侧留出基线，右侧给 ns 标记留出空间
    plt.xlim(0.84, mean_f1.max() + std_f1.max() + 0.015) 
    plt.xlabel("Weighted F1 Score", fontsize=12, fontweight='bold')
    plt.ylabel("Model", fontsize=12, fontweight='bold')
    
    # --- 画横向统计显著性标志 (ns) ---
    idx_shbrb = sorted_models.index('SH-BRB')
    idx_rf = sorted_models.index('Random Forest') 
    
    # 计算误差棒最右侧的 X 坐标
    max_x_shbrb = mean_f1['SH-BRB'] + std_f1['SH-BRB']
    max_x_rf = mean_f1['Random Forest'] + std_f1['Random Forest']
    
    # 大括号画在最长误差棒的右侧偏外一点
    x_bracket = max(max_x_shbrb, max_x_rf) + 0.003
    w_bracket = 0.0015 # 大括号的小拐角宽度
    
    # 画连接线 (横向躺着的 U 型)
    plt.plot([x_bracket - w_bracket, x_bracket, x_bracket, x_bracket - w_bracket], 
             [idx_shbrb, idx_shbrb, idx_rf, idx_rf], 
             lw=1.5, c='black')
    
    # 打上 ns 文本 (在横线的右侧)
    plt.text(x_bracket + 0.001, (idx_shbrb + idx_rf) * 0.5, 
             "ns", ha='left', va='center', color='black', fontsize=12)

    # 把数值标签写在柱子的根部内部，这样既清晰又不会和误差棒重叠
    for i, p in enumerate(ax1.patches):
        # 将数值写在 x=0.842 的位置 (柱子内部靠左)
        ax1.annotate(f"{mean_f1[sorted_models[i]]:.4f}", 
                     (0.842, p.get_y() + p.get_height() / 2.), 
                     ha='left', va='center', color='white', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig("Final_Overall_F1_Comparison_Full.png", dpi=300)
    print("✅ 图表1 (横向还原版+误差棒+ns) 已保存")


    # ========================================================
    # 图 4-3: 高风险识别能力对比 (保留你满意的高级区分配色)
    # ========================================================
    plt.figure(figsize=(14, 6))
    
    baseline_palette = sns.color_palette("Set2", n_colors=len(model_list)-1)
    custom_palette_hr = {'SH-BRB': '#e74c3c'}
    c_idx = 0
    for m in model_list:
        if m != 'SH-BRB':
            custom_palette_hr[m] = baseline_palette[c_idx]
            c_idx += 1
            
    ax2 = sns.barplot(
        x="Metric", 
        y="Score", 
        hue="Model", 
        data=df_hr, 
        palette=custom_palette_hr,
        errorbar=None 
    )
    
    plt.ylim(0.75, 1.05) 
    plt.ylabel("Score", fontsize=12, fontweight='bold')
    plt.xlabel("")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.3f', padding=3, fontsize=9)
        
    plt.tight_layout()
    plt.savefig("Final_High_Risk_Comparison_Full.png", dpi=300)
    print("✅ 图表2 (高级颜色区分版) 已保存")

if __name__ == '__main__':
    main()