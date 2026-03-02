import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 准备数据 (模型A vs 模型C) ---
# 数据来源于您之前提供的LaTex性能报告
data_ac = {
    'Model': [
        # Model A Data
        'A: SH-BRB (with SEM)', 'A: SH-BRB (with SEM)', 'A: SH-BRB (with SEM)',
        'A: SH-BRB (with SEM)', 'A: SH-BRB (with SEM)', 'A: SH-BRB (with SEM)',
        'A: SH-BRB (with SEM)', 'A: SH-BRB (with SEM)', 'A: SH-BRB (with SEM)',
        # Model C Data
        'C: BRB-Heuristic (no SEM)', 'C: BRB-Heuristic (no SEM)', 'C: BRB-Heuristic (no SEM)',
        'C: BRB-Heuristic (no SEM)', 'C: BRB-Heuristic (no SEM)', 'C: BRB-Heuristic (no SEM)',
        'C: BRB-Heuristic (no SEM)', 'C: BRB-Heuristic (no SEM)', 'C: BRB-Heuristic (no SEM)',
    ],
    'Class': [ 'Low', 'Medium', 'High'] * 6,
    'Metric': ['Precision']*3 + ['Recall']*3 + ['F1-Score']*3 + ['Precision']*3 + ['Recall']*3 + ['F1-Score']*3,
    'Score': [
        # Model A Scores (Precision, Recall, F1)
        0.970, 0.735, 0.993,  # Precision
        0.826, 0.980, 0.816,  # Recall
        0.891, 0.840, 0.894,  # F1-Scores
        # Model C Scores (Precision, Recall, F1)
        0.946, 0.732, 0.989,  # Precision
        0.823, 0.961, 0.816,  # Recall
        0.879, 0.830, 0.892,  # F1-Scores
    ]
}

df_ac = pd.DataFrame(data_ac)

# --- 2. 绘图 (使用与上一张图完全相同的模板) ---
g = sns.catplot(
    data=df_ac, 
    x='Class', 
    y='Score', 
    hue='Model', 
    col='Metric',
    kind='bar',
    height=4,
    aspect=0.8,
    palette=['#1f77b4', '#ff7f0e'] # 为模型A和C指定新的对比色
)

# --- 3. 为每个条形添加数值标签 ---
for ax in g.axes.flat:
    for patch in ax.patches:
        ax.annotate(f'{patch.get_height():.3f}',
                    (patch.get_x() + patch.get_width() / 2., patch.get_height()),
                    ha='center', va='center', xytext=(0, 5),
                    textcoords='offset points', fontsize=9)

# --- 4. 添加和调整图例 ---
sns.move_legend(g, "upper right", bbox_to_anchor=(.98, 0.98), title='Model')

# --- 5. 优化与标签 ---
g.set(ylim=(0.7, 1.08))
g.set_axis_labels("Stress Level Category", "Score")
g.set_titles("{col_name}", size=14)
g.fig.subplots_adjust(top=0.88, wspace=0.2)

# --- 6. 保存与显示 ---
plt.savefig("ablation_study_2_sem_knowl_final.png", dpi=300, bbox_inches='tight')
plt.show()

print("\n最终版消融研究II对比图已生成并保存为 'ablation_study_2_sem_knowl_final.png'")