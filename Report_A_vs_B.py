import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 数据部分 (保持不变) ---
data = {
    'Model': [
        'A: SH-BRB (with History Mod.)', 'A: SH-BRB (with History Mod.)', 'A: SH-BRB (with History Mod.)',
        'A: SH-BRB (with History Mod.)', 'A: SH-BRB (with History Mod.)', 'A: SH-BRB (with History Mod.)',
        'A: SH-BRB (with History Mod.)', 'A: SH-BRB (with History Mod.)', 'A: SH-BRB (with History Mod.)',
        'B: No History Mod.', 'B: No History Mod.', 'B: No History Mod.',
        'B: No History Mod.', 'B: No History Mod.', 'B: No History Mod.',
        'B: No History Mod.', 'B: No History Mod.', 'B: No History Mod.',
    ],
    'Class': [ 'Low', 'Medium', 'High'] * 6,
    'Metric': ['Precision']*3 + ['Recall']*3 + ['F1-Score']*3 + ['Precision']*3 + ['Recall']*3 + ['F1-Score']*3,
    'Score': [0.970, 0.735, 0.993, 0.826, 0.980, 0.816, 0.891, 0.840, 0.894,
              0.873, 0.860, 0.885, 0.866, 0.886, 0.857, 0.868, 0.870, 0.869]
}
df = pd.DataFrame(data)

# --- 2. 绘图部分 (核心修改) ---
# 【修改】移除 legend=False，让seaborn自动生成图例
g = sns.catplot(
    data=df, 
    x='Class', 
    y='Score', 
    hue='Model', 
    col='Metric',
    kind='bar',
    height=4,
    aspect=0.8,
    palette=['#1f77b4', '#9467bd']
)

# --- 3. 为每个条形添加数值标签 (保持不变) ---
for ax in g.axes.flat:
    for patch in ax.patches:
        ax.annotate(f'{patch.get_height():.3f}',
                    (patch.get_x() + patch.get_width() / 2., patch.get_height()),
                    ha='center', va='center', xytext=(0, 5),
                    textcoords='offset points', fontsize=9)

# --- 4. 【核心修正】正确地移动和调整图例 ---
# 使用 seaborn 的 move_legend 函数，这是最稳健的方法
sns.move_legend(g, "upper right", bbox_to_anchor=(.98, 0.98), title='Model')


# --- 5. 优化与标签 (保持不变) ---
g.set(ylim=(0.7, 1.08))
g.set_axis_labels("Stress Level Category", "Score")
g.set_titles("{col_name}", size=14)
g.fig.subplots_adjust(top=0.88, wspace=0.2)

# --- 6. 保存与显示 ---
plt.savefig("ablation_study_1_history_mod_final.png", dpi=300, bbox_inches='tight')
plt.show()