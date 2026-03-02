# ER-Data preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# 设置中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_columns', None)

def load_data(file_path):
    """加载原始数据"""
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    print("✅ 数据加载成功！")
    return df

def basic_checks(df):
    """数据基础检查"""
    # ... (此函数无需修改)
    print("\n=== 缺失值统计 ===")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    print(f"\n重复行数: {df.duplicated().sum()}")
    print("\n目标变量分布（stress_level）:")
    print(df['stress_level'].value_counts())
    return df

def handle_missing_values(df):
    """处理缺失值"""
    # ... (此函数无需修改)
    print("\n=== 处理缺失值 ===")
    continuous_cols = ['blood_pressure', 'sleep_quality', 'self_esteem', 
                     'depression', 'headache', 'anxiety_level', 'study_load']
    categorical_cols = ['mental_health_history', 'bullying']
    for col in continuous_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"填充 {col} (连续): 中位数={median_val:.2f}")
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"填充 {col} (分类): 众数={mode_val}")
    return df

def remove_duplicates(df):
    """删除重复值"""
    # ... (此函数无需修改)
    initial = len(df)
    df = df.drop_duplicates()
    print(f"\n删除重复行: {initial - len(df)}")
    return df

def adjust_direction(df):
    """统一调整变量方向"""
    # ... (此函数无需修改)
    print("\n=== 统一调整变量方向 ===")
    reverse_vars = [
        'academic_performance', 'sleep_quality', 'teacher_student_relationship', 
        'social_support', 'living_conditions', 'safety', 'basic_needs', 
        'extracurricular_activities', 'self_esteem'
    ]
    # 暂存原始self_esteem用于反转
    original_self_esteem = df['self_esteem'].copy()
    for var in reverse_vars:
        if var in df.columns and var != 'self_esteem':
            df[var] = 6 - df[var]
            print(f"已反转 {var} 方向 (1-5分量表)")
    
    if 'self_esteem' in df.columns:
        # 特殊处理：self_esteem是0-30分量表
        max_val = original_self_esteem.max()
        min_val = original_self_esteem.min()
        df['self_esteem'] = max_val + min_val - original_self_esteem
        print(f"已反转 self_esteem 方向 (范围 {min_val}-{max_val})")
    return df

def handle_outliers(df):
    """处理异常值"""
    # ... (此函数无需修改)
    print("\n=== 处理异常值 ===")
    continuous_cols = ['anxiety_level', 'blood_pressure', 'study_load', 'self_esteem']
    for col in continuous_cols:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df[col] = np.clip(df[col], lower_bound, upper_bound)
            print(f"{col}: 异常值已通过缩尾处理")
    return df
    
# --- 新增函数 ---
def generate_descriptive_stats(df, output_path="descriptive_stats_for_paper.csv"):
    """为论文生成并保存描述性统计表格"""
    print("\n=== 生成描述性统计 (用于论文表1) ===")
    
    # 定义变量及其所属维度
    var_dims = {
        '心理': ['anxiety_level', 'self_esteem', 'depression'],
        '生理': ['headache', 'blood_pressure', 'sleep_quality', 'breathing_problem'],
        '学业': ['academic_performance', 'study_load', 'teacher_student_relationship', 'future_career_concerns'],
        '社交': ['social_support', 'peer_pressure', 'extracurricular_activities', 'bullying'],
        '环境': ['noise_level', 'living_conditions', 'safety', 'basic_needs'],
        '背景与结果': ['mental_health_history', 'stress_level']
    }
    
    stats = df.describe().transpose()
    stats = stats[['mean', 'std', 'min', 'max']]
    stats.columns = ['均值 (Mean)', '标准差 (Std. Dev.)', '最小值 (Min)', '最大值 (Max)']
    
    # 创建最终的表格DataFrame
    table_df = []
    for dim, var_list in var_dims.items():
        for var in var_list:
            if var in stats.index:
                row = stats.loc[var]
                row['维度'] = dim
                row['变量名 (Variable)'] = var
                table_df.append(row)
                
    final_table = pd.DataFrame(table_df)
    final_table = final_table[['维度', '变量名 (Variable)', '均值 (Mean)', '标准差 (Std. Dev.)', '最小值 (Min)', '最大值 (Max)']]
    
    # 保存为CSV
    final_table.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 描述性统计表格已保存至: {output_path}")
    print("表格预览:")
    print(final_table)
    return final_table

def standardize_data(df):
    """Z-score标准化"""
    # ... (此函数无需修改)
    print("\n=== 数据标准化 ===")
    continuous_cols = [
        'anxiety_level', 'self_esteem', 'depression', 'blood_pressure', 'study_load', 
        'sleep_quality', 'academic_performance', 'teacher_student_relationship', 
        'future_career_concerns', 'social_support', 'peer_pressure', 
        'extracurricular_activities', 'noise_level', 'living_conditions', 'safety', 
        'basic_needs', 'headache', 'breathing_problem'
    ]
    existing_cols = [col for col in continuous_cols if col in df.columns]
    if existing_cols:
        scaler = StandardScaler()
        df[existing_cols] = scaler.fit_transform(df[existing_cols])
        print(f"已标准化 {len(existing_cols)} 个变量")
    return df

def save_and_visualize(df, output_path):
    """保存结果并可视化"""
    # ... (此函数无需修改)
    df.to_csv(output_path, index=False)
    print(f"\n💾 预处理完成! 数据已保存至: {output_path}")
    
def main():
    df = load_data('mental_pressure_levelData.csv')
    df = basic_checks(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = adjust_direction(df)
    df = handle_outliers(df)
    
    # --- 关键修改：在标准化之前，调用新函数生成描述性统计 ---
    generate_descriptive_stats(df)
    
    df_standardized = standardize_data(df.copy()) # 注意：传入副本进行标准化
    
    save_and_visualize(df_standardized, 'preprocessed_data.csv')

if __name__ == '__main__':
    print("=== 开始数据预处理 ===")
    main()
    print("=== 预处理完成 ===")