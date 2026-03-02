"""
ER融合实验：心理组CFA权重分配（最终版）
"""
import pandas as pd
import numpy as np
from semopy import Model
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

def load_processed_data():
    df = pd.read_csv('preprocessed_data.csv')
    print("✅ 预处理数据加载成功！")
    
    psych_vars = ['anxiety_level', 'self_esteem', 'depression']
    print("\n=== 心理组变量统计信息 ===")
    print(df[psych_vars].describe().round(3))
    
    missing = df[psych_vars].isnull().sum()
    if missing.sum() == 0:
        print("\n✅ 无缺失值")
    else:
        print(f"\n⚠️ 缺失值:\n{missing[missing>0]}")
    
    return df, psych_vars

def fit_cfa_model(df, model_spec):
    mod = Model(model_spec)
    mod.fit(df)
    print("\n=== 标准化因子载荷 ===")
    params = mod.inspect(std_est=True)
    loadings = params[(params['op'] == '~') & (params['rval'] == 'Mental_Health')]
    print(loadings[['lval', 'Estimate']])
    return loadings.set_index('lval')['Estimate']

def calculate_weights(loadings):
    # 保留方向信息（不取绝对值）
    total = sum(abs(loadings))
    weights = {var: abs(loadings[var])/total for var in loadings.index}
    
    print("\n=== 心理组变量权重 ===")
    for var, w in weights.items():
        direction = '正向' if loadings[var] > 0 else '负向'
        print(f"{var}: {w:.3f} (方向: {direction})")
    
    return weights

def calculate_score(df, weights, loadings, psych_vars):  # 添加loadings参数
    # 计算原始得分（保留方向）
    score = pd.Series(0.0, index=df.index)
    for var in weights:
        sign = 1 if loadings[var] > 0 else -1
        score += df[var] * sign * weights[var]
    
    # Z-score标准化
    scaler = StandardScaler()
    standardized_score = scaler.fit_transform(score.values.reshape(-1, 1))
    
    # 创建结果DataFrame
    df_score = df[psych_vars].copy()
    df_score['Psychological state_score'] = standardized_score
    
    print("\n=== 心理健康标准化得分分布 ===")
    print(df_score['Psychological state_score'].describe().round(3))
    return df_score

def main():
    # 加载数据
    df, psych_vars = load_processed_data()
    
    # 定义CFA模型
    model_spec = '''
    Mental_Health =~ anxiety_level + self_esteem + depression
    '''
    print("\n=== CFA模型定义 ===")
    print(model_spec.strip())
    
    # 拟合模型并获取载荷
    loadings = fit_cfa_model(df, model_spec)
    
    # 计算权重
    weights = calculate_weights(loadings)
    
    # 计算得分（传入载荷）
    result_df = calculate_score(df, weights, loadings, psych_vars)
    
    # 保存结果
    output_path = 'Psychological state_score.csv'
    result_df.to_csv(output_path, index=False)
    print(f"\n💾 数据已保存至: {output_path}")

if __name__ == '__main__':
    main()