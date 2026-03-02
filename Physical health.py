# 文件名: Physical health.py
# 功能: ER融合实验：生理组熵权法权重分配（已修正版本）

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

def load_processed_data():
    """加载预处理好的数据"""
    try:
        df = pd.read_csv('preprocessed_data.csv')
        print("✅ 预处理数据加载成功！")
    except FileNotFoundError:
        print("❌ 错误：未找到 'preprocessed_data.csv'。请确保预处理已运行。")
        exit()
    
    physio_vars = ['headache', 'blood_pressure', 'sleep_quality', 'breathing_problem']
    
    print("\n=== 生理组变量统计信息 ===")
    print(df[physio_vars].describe().round(3))
    
    missing = df[physio_vars].isnull().sum()
    print("\n✅ 无缺失值" if missing.sum() == 0 else f"\n⚠️ 缺失值:\n{missing[missing>0]}")
    
    return df, physio_vars

def prepare_positive_data(df, cols):
    """
    为熵权法准备正值数据。
    熵权法要求所有输入值>0。我们通过线性变换将数据缩放到[0.001, 1]区间。
    """
    df_positive = df.copy()
    
    # 检查数据是否已经是正值
    if (df[cols] > 0).all().all():
        print("所有数据已为正值，跳过正向化处理。")
        return df_positive

    print("\n=== 执行数据正向化（熵权法要求）===")
    for col in cols:
        min_val = df[col].min()
        max_val = df[col].max()
        range_val = max_val - min_val
        
        if range_val == 0:
            # 如果列中所有值都相同，则赋一个恒定正值
            df_positive[col] = 0.5
            print(f"变量 '{col}' 值恒定，已处理。")
        else:
            # 线性变换到[0.001, 1.0]
            df_positive[col] = 0.001 + 0.999 * (df[col] - min_val) / range_val
            
    print("✅ 数据正向化完成。")
    return df_positive

def entropy_weight(df, cols):
    """使用熵权法计算权重"""
    # 确保数据是正的
    X = df[cols].values
    if (X <= 0).any():
        raise ValueError("熵权法输入数据必须全部为正值。")

    # 计算指标比重
    P = X / np.sum(X, axis=0)
    
    # 计算熵值
    epsilon = 1e-12
    entropy = -np.sum(P * np.log(P + epsilon), axis=0) / np.log(len(X))
    
    # 计算差异系数
    diversity = 1 - entropy
    
    # 计算权重
    weights = diversity / diversity.sum()
    
    weight_dict = dict(zip(cols, weights))
    print("\n=== 熵权法计算结果 ===")
    print(f"各指标熵值: {np.round(entropy, 4)}")
    print(f"差异系数: {np.round(diversity, 4)}")
    print("\n=== 生理组变量权重 ===")
    for var, w in weight_dict.items():
        print(f"{var}: {w:.4f}")
    
    # 保存权重
    model_dir = Path("first_fusion_models")
    model_dir.mkdir(exist_ok=True)
    weight_series = pd.Series(weight_dict, name='Entropy_Weights')
    joblib.dump(weight_series, model_dir / "Physical_weights.pkl")
    return weight_dict

def calculate_physio_score(df_raw, df_positive, weight_dict, physio_vars):
    """
    【已修正】计算生理健康得分。
    使用与权重来源相同的数据(df_positive)进行计算，然后将得分标准化并附加到原始数据(df_raw)上。
    """
    print("\n=== 开始计算生理健康得分（修正版） ===")
    # 核心修正：在 df_positive (用于计算权重的数据)上进行加权求和
    score = sum(df_positive[var] * w for var, w in weight_dict.items())
    print("✅ 加权得分计算完成。")
    
    # 对计算出的得分进行Z-score标准化
    scaler = StandardScaler()
    standardized_score = scaler.fit_transform(score.values.reshape(-1, 1))
    
    # 创建结果DataFrame，使用原始数据的值，但附加新计算的标准化得分
    # df_positive的索引是经过预处理的，应与df_raw的索引一致
    result_df = df_raw[physio_vars].copy()
    result_df['Physical health_score'] = standardized_score
    
    print("\n=== 生理健康标准化得分分布 ===")
    print(result_df['Physical health_score'].describe().round(3))
    return result_df

def main():
    # 1. 加载数据
    df_raw, physio_vars = load_processed_data()
    
    # 2. 准备正值数据（仅用于熵权法计算）
    df_positive = prepare_positive_data(df_raw, physio_vars)
    
    # 3. 基于正值数据计算权重
    weights = entropy_weight(df_positive, physio_vars)
    
    # 4. 计算得分【修正】
    #    传入原始数据用于最终保存，传入正值数据用于计算
    result_df = calculate_physio_score(df_raw, df_positive, weights, physio_vars)
    
    # 5. 保存结果
    output_path = 'Physical health_score.csv'
    result_df.to_csv(output_path, index=False)
    print(f"\n💾 数据已保存至: {output_path}")
    print(f"包含列: {result_df.columns.tolist()}")

if __name__ == '__main__':
    main()