import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

class SocialPCA:
    def __init__(self):
        # 定义变量方向（调整后所有变量：高分=社会适应差）
        self.var_directions = {
            'social_support': -1,    # 原始高分=支持多→反转
            'peer_pressure': 1,       # 原始高分=压力大→保持
            'extracurricular_activities': -1,  # 原始高分=活动多→反转
            'bullying': 1              # 原始高分=欺凌多→保持
        }
        self.social_vars = list(self.var_directions.keys())
        self.model_dir = Path("first_fusion_models")
        self.model_dir.mkdir(exist_ok=True)
        
    def load_data(self):
        """加载数据并校验变量"""
        df = pd.read_csv('preprocessed_data.csv')
        print("✅ 预处理数据加载成功！")
        
        # 检查社会变量
        missing_vars = [var for var in self.social_vars if var not in df.columns]
        if missing_vars:
            raise ValueError(f"缺失社会变量: {missing_vars}")
            
        print("\n=== 社会组变量统计信息 ===")
        print(df[self.social_vars].describe().round(3))
        return df
    
    def adjust_direction(self, df):
        """方向调整：确保所有变量高分=社会适应差"""
        df_adj = df[self.social_vars].copy()
        for var, direction in self.var_directions.items():
            if direction == -1:
                df_adj[var] = -df_adj[var]  # 反转负向指标
        return df_adj
    
    def fit_pca(self, df):
        """PCA建模 + 主成分方向修正"""
        pca = PCA(n_components=1)
        pca.fit(df)
        
        # 确保主成分方向：高分=社会适应差
        # 使用所有变量载荷均值作为方向判断
        if pca.components_[0].mean() < 0:
            pca.components_ = -pca.components_
        
        # 保存模型
        joblib.dump(pca, self.model_dir / "Social_pca.pkl")
        return pca
    
    def calculate_weights(self, pca):
        """权重计算（确保所有权重为正）"""
        # 获取第一主成分载荷
        loadings = pca.components_[0]
        
        # 取绝对值并归一化
        abs_loadings = np.abs(loadings)
        weights = abs_loadings / abs_loadings.sum()
        
        # 转换为Series
        return pd.Series(weights, index=self.social_vars)
    
    def calculate_score(self, df, weights):
        """计算社会适应得分（高分=适应差）"""
        # 计算原始得分
        score = (df * weights).sum(axis=1)
        
        # Z-score标准化
        scaler = StandardScaler()
        standardized_score = scaler.fit_transform(score.values.reshape(-1, 1))
        
        # 创建结果DataFrame
        result_df = df.copy()
        result_df['Social_relations_score'] = standardized_score
        
        print("\n=== 社会适应标准化得分分布 ===")
        print(result_df['Social_relations_score'].describe().round(3))
        return result_df
    
    def run_pipeline(self):
        """完整处理流程"""
        try:
            # 1. 加载数据
            raw_df = self.load_data()
            
            # 2. 方向调整
            df_adj = self.adjust_direction(raw_df)
            
            # 3. PCA建模
            pca = self.fit_pca(df_adj)
            
            # 4. 权重计算
            weights = self.calculate_weights(pca)
            print("\n=== 社会变量权重 ===")
            print(weights.round(4))
            
            # 5. 计算得分
            result_df = self.calculate_score(df_adj, weights)
            
            # 6. 保存结果（只保留社会变量和得分）
            output_cols = self.social_vars + ['Social_relations_score']
            result_df[output_cols].to_csv('Social_relations_score.csv', index=False)
            print(f"\n💾 数据已保存至: Social_relations_score.csv")
            
            return True
        except Exception as e:
            print(f"处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == '__main__':
    SocialPCA().run_pipeline()