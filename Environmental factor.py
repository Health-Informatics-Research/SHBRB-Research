"""
环境组PCA权重分配（最终优化版）
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

class EnvironmentalPCA:
    def __init__(self):
        # 定义变量及业务方向（1表示高分=环境质量差，-1表示相反）
        self.var_directions = {
            'noise_level': 1,          # 噪声越高环境越差
            'living_conditions': -1,   # 居住条件越好环境越好
            'safety': -1,              # 安全性越高环境越好
            'basic_needs': -1          # 基本需求满足度越高环境越好
        }
        self.env_vars = list(self.var_directions.keys())
        self.model_dir = Path("first_fusion_models")
        self.model_dir.mkdir(exist_ok=True)
        
    def load_data(self):
        """数据加载与校验"""
        df = pd.read_csv('preprocessed_data.csv')
        print("✅ 预处理数据加载成功！")
        
        # 检查环境变量
        missing_vars = [var for var in self.env_vars if var not in df.columns]
        if missing_vars:
            raise ValueError(f"缺失环境变量: {missing_vars}")
        
        print("\n=== 环境组变量统计信息 ===")
        print(df[self.env_vars].describe().round(3))
        return df
    
    def adjust_direction(self, df):
        """方向调整：确保所有变量高分=环境差"""
        df_adj = df[self.env_vars].copy()
        for var, direction in self.var_directions.items():
            if direction == -1:
                df_adj[var] = -df_adj[var]  # 反转负向指标
        return df_adj
    
    def fit_pca(self, df):
        """PCA建模"""
        pca = PCA(n_components=1)  # 只保留第一主成分
        pca.fit(df)
        
        # 确保主成分方向：高分=环境差
        if pca.components_[0].mean() < 0:
            pca.components_ = -pca.components_
        
        # 保存模型
        joblib.dump(pca, self.model_dir / "Environmental_pca.pkl")
        return pca
    
    def calculate_weights(self, pca):
        """权重计算（确保所有权重为正）"""
        # 获取第一主成分载荷
        loadings = pca.components_[0]
        
        # 取绝对值并归一化
        abs_loadings = np.abs(loadings)
        weights = abs_loadings / abs_loadings.sum()
        
        # 转换为Series
        return pd.Series(weights, index=self.env_vars)
    
    def calculate_score(self, df, weights):
        """计算环境得分（高分=环境差）"""
        # 计算原始得分
        score = (df * weights).sum(axis=1)
        
        # Z-score标准化
        scaler = StandardScaler()
        standardized_score = scaler.fit_transform(score.values.reshape(-1, 1))
        
        # 创建结果DataFrame
        result_df = df.copy()
        result_df['Environmental factor_score'] = standardized_score
        
        print("\n=== 环境压力标准化得分分布 ===")
        print(result_df['Environmental factor_score'].describe().round(3))
        return result_df
    
    def run_pipeline(self):
        """完整处理流程"""
        try:
            # 加载数据
            raw_df = self.load_data()
            
            # 方向调整
            df_adj = self.adjust_direction(raw_df)
            
            # PCA建模
            pca = self.fit_pca(df_adj)
            
            # 权重计算
            weights = self.calculate_weights(pca)
            print("\n=== 环境变量权重 ===")
            print(weights.round(4))
            
            # 计算得分
            result_df = self.calculate_score(df_adj, weights)
            
            # 保存结果（只保留环境变量和得分）
            output_cols = self.env_vars + ['Environmental factor_score']
            result_df[output_cols].to_csv('Environmental factor_score.csv', index=False)
            print(f"\n💾 数据已保存至: Environmental factor_score.csv")
            
            return True
        except Exception as e:
            print(f"处理失败: {str(e)}")
            return False

if __name__ == '__main__':
    EnvironmentalPCA().run_pipeline()