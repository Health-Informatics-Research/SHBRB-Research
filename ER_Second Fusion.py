"""
心理健康综合评估系统 - 高阶因子模型二次融合 (最终修复版)

功能：基于高阶因子模型的综合评估，生成心理健康综合指数
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from semopy import Model, calc_stats
import warnings
import traceback

# 忽略semopy可能产生的FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)


# 定义各组配置
GROUP_CONFIG = {
    'Psychological': {
        'score_file': 'Psychological state_score.csv',
        'score_column': 'Psychological state_score',
        'method': 'CFA',
    },
    'Physical': {
        'score_file': 'Physical health_score.csv',
        'score_column': 'Physical health_score',
        'method': 'Entropy',
    },
    'Environmental': {
        'score_file': 'Environmental factor_score.csv',
        'score_column': 'Environmental factor_score',
        'method': 'PCA',
    },
    'Academic': {
        'score_file': 'Academic pressure_score.csv',
        'score_column': 'Academic pressure_score',
        'method': 'SEM',
    },
    'Social': {
        'score_file': 'Social_relations_score.csv',
        'score_column': 'Social_relations_score',
        'method': 'PCA',
    }
}

class MentalHealthAnalyzer:
    def __init__(self):
        # 创建模型目录
        self.model_dir = Path("second_fusion_models")
        self.model_dir.mkdir(exist_ok=True)
        
        # 初始化数据结构
        self.group_scores = {}  # 存储各维度得分
        self.composite_index = None  # 综合指标
        self.higher_order_weights = None  # 高阶因子模型权重
        self.validation_report = []  # 验证报告
        
        # 加载原始数据（用于最终结果合并）
        try:
            self.raw_data = pd.read_csv('preprocessed_data.csv')
            print("✅ 原始数据加载成功")
        except Exception as e:
            print(f"❌ 原始数据加载失败: {str(e)}")
            self.raw_data = None

    def load_group_scores(self):
        """加载各维度的得分文件"""
        print("\n=== 加载各组得分 ===")
        all_loaded = True
        
        for group, config in GROUP_CONFIG.items():
            try:
                # 加载得分文件
                df_score = pd.read_csv(config['score_file'])
                
                # 验证得分列是否存在
                if config['score_column'] not in df_score.columns:
                    raise ValueError(f"得分列 '{config['score_column']}' 不存在")
                
                # 提取得分列
                score_data = df_score[config['score_column']]
                
                # 检查数据有效性
                if len(score_data) == 0:
                    raise ValueError("得分数据为空")
                
                # 存储得分
                self.group_scores[group] = score_data
                print(f"✅ {group} 组得分加载成功 ({len(score_data)} 样本)")
                
                # 记录验证信息
                self.validation_report.append({
                    'Group': group,
                    'Status': 'Success',
                    'Method': config['method'],
                    'Mean': score_data.mean().round(4),
                    'Std': score_data.std().round(4),
                    'Min': score_data.min().round(4),
                    'Max': score_data.max().round(4)
                })
                
            except Exception as e:
                print(f"❌ {group} 组得分加载失败: {str(e)}")
                self.validation_report.append({
                    'Group': group,
                    'Status': 'Failed',
                    'Method': config['method'],
                    'Error': str(e)
                })
                all_loaded = False
        
        return all_loaded

    def standardize_scores(self):
        """确保所有组得分标准化（均值为0，标准差为1）"""
        print("\n=== 得分标准化检查 ===")
        for group, scores in self.group_scores.items():
            mean = scores.mean()
            std = scores.std()
            
            # 检查标准化情况
            if abs(mean) > 0.01 or abs(std - 1) > 0.01:
                print(f"⚠️ {group}组: 均值={mean:.4f}, 标准差={std:.4f} → 执行标准化")
                self.group_scores[group] = (scores - mean) / std
                new_mean = self.group_scores[group].mean()
                new_std = self.group_scores[group].std()
                print(f"    标准化后: 均值={new_mean:.4f}, 标准差={new_std:.4f}")
            else:
                print(f"✅ {group}组: 已标准化 (均值≈{mean:.4f}, 标准差≈{std:.4f})")

    def fit_higher_order_model(self):
        """使用高阶因子模型计算维度权重"""
        try:
            print("\n=== 拟合高阶因子模型 ===")
            
            # 准备数据 - 各维度得分
            df_scores = pd.DataFrame(self.group_scores)
            
            # 检查数据完整性
            if df_scores.isnull().sum().sum() > 0:
                print("⚠️ 存在缺失值，使用中位数填充")
                df_scores = df_scores.fillna(df_scores.median())
            
            # 构建高阶因子模型 - 使用更简洁的定义
            model_spec = '''
                # 高阶心理健康因子
                Overall_Mental_Health =~ Psychological + Physical + Environmental + Academic + Social
            '''
            model = Model(model_spec)
            res = model.fit(df_scores, obj='MLW')
            
            # 检查模型拟合结果
            if res is None:
                raise RuntimeError("模型拟合失败，未返回结果")
            
            # 获取完整的参数表
            params = model.inspect(std_est=True)
            print("\n=== 完整参数表 ===")
            print(params)
            
            # 修复参数提取逻辑 - 使用正确的操作符和变量关系
            # 在semopy中，测量模型使用'~'操作符表示
            # 潜变量是预测变量，指标是因变量
            loadings = params[
                (params['op'] == '~') & 
                (params['rval'] == 'Overall_Mental_Health')
            ]
            
            if loadings.empty:
                # 尝试备选提取方式
                print("⚠️ 使用备选方式提取载荷")
                # 查找所有以潜变量为预测变量的参数
                loadings = params[
                    (params['op'] == '~') & 
                    (params['rval'].str.contains('Overall_Mental_Health'))
                ]
                if loadings.empty:
                    raise ValueError("无法提取因子载荷，请检查模型定义")
            
            # 确保载荷方向一致（高分=心理健康差）
            if loadings['Est. Std'].mean() < 0:
                print("⚠️ 因子载荷方向反转（确保高分=心理健康差）")
                loadings['Est. Std'] = -loadings['Est. Std']
                
            # 提取载荷值
            loadings_vals = loadings.set_index('lval')['Est. Std']
            
            # 打印载荷信息
            print("\n=== 标准化因子载荷 ===")
            print(loadings[['lval', 'Est. Std']])
            
            # 计算权重（取绝对值并归一化）
            abs_loadings = loadings_vals.abs()
            self.higher_order_weights = abs_loadings / abs_loadings.sum()
            
            # 打印权重
            print("\n=== 高阶因子模型权重 ===")
            print(self.higher_order_weights.round(4))
            
            # 计算适配度指标
            try:
                stats = calc_stats(model)
                stats = stats.dropna(axis=1, how='all')
                print("\n=== 模型适配度指标 ===")
                print(stats.T.round(3))
                
                # 保存适配度指标
                fit_indices = stats.iloc[0].to_dict()
            except Exception as e:
                print(f"适配度指标计算失败: {str(e)}")
                fit_indices = {}
            
            # 保存关键模型信息
            model_info = {
                'model_spec': model_spec,
                'loadings': loadings_vals.to_dict(),
                'weights': self.higher_order_weights.to_dict(),
                'fit_indices': fit_indices,
                'data_means': df_scores.mean().to_dict(),
                'data_stds': df_scores.std().to_dict()
            }
            
            joblib.dump(model_info, self.model_dir / "Higher_Order_CFA.pkl")
            print("✅ 高阶因子模型关键信息已保存")
            
            return True
            
        except Exception as e:
            print(f"❌ 高阶因子模型拟合失败: {str(e)}")
            traceback.print_exc()
            return False

    def calculate_composite_index(self):
        """计算综合心理健康指数"""
        print("\n=== 计算综合指数 ===")
        
        # 初始化综合得分
        composite = pd.Series(0.0, index=self.group_scores['Psychological'].index)
        
        # 加权求和
        for group, weight in self.higher_order_weights.items():
            composite += self.group_scores[group] * weight
            print(f"  + {weight:.4f} * {group}")
        
        # 标准化综合指数（Z-score）
        composite = (composite - composite.mean()) / composite.std()
        self.composite_index = composite
        
        # 打印分布
        print("\n=== 综合指数分布 ===")
        print(composite.describe().round(3))
        
        return composite

    def visualize_results(self):
            """可视化结果"""
            print("\n=== 生成可视化结果 ===")
            
            try:
                # 1. 权重分布图 (Weight Distribution Plot)
                plt.figure(figsize=(10, 6))
                weights_df = pd.DataFrame({
                    'Dimension': self.higher_order_weights.index,
                    'Weight': self.higher_order_weights.values
                }).sort_values('Weight', ascending=False)
                
                sns.barplot(x='Dimension', y='Weight', data=weights_df, palette='viridis')
                plt.title('Distribution of Mental Health Dimension Weights', fontsize=14)
                plt.ylabel('Weight')
                plt.xlabel('Dimension')
                plt.xticks(rotation=15)
                plt.tight_layout()
                plt.savefig('higher_order_weights.png', dpi=300)
                plt.close()
                print("✅ 权重分布图已保存")
                
                # 2. 综合指标分布 (Composite Indicator Distribution) - (对应您论文的图2.2)
                plt.figure(figsize=(10, 6))
                sns.histplot(self.composite_index, kde=True, bins=30, color='skyblue')
                plt.title('Distribution of Composite Mental Health Index', fontsize=14)
                plt.xlabel('Composite Score')
                plt.ylabel('Frequency')
                plt.axvline(self.composite_index.mean(), color='red', linestyle='dashed', linewidth=1)
                plt.grid(axis='y', alpha=0.3)
                plt.savefig('composite_distribution.png', dpi=300)
                plt.close()
                print("✅ 综合指标分布图已保存")
                
                # 3. 维度得分相关热力图 (Correlation Heatmap)
                df_scores = pd.DataFrame(self.group_scores)
                plt.figure(figsize=(10, 8))
                corr = df_scores.corr()
                sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
                plt.title('Correlation Heatmap of Mental Health Dimensions', fontsize=14)
                plt.tight_layout()
                plt.savefig('dimension_correlation.png', dpi=300)
                plt.close()
                print("✅ 维度相关性热力图已保存")
                
            except Exception as e:
                print(f"❌ 可视化失败: {str(e)}")
                traceback.print_exc()

    def save_results(self):
        """保存最终结果"""
        print("\n=== 保存结果 ===")
        
        try:
            # 创建结果DataFrame
            result_df = pd.DataFrame()
            
            # 添加各维度得分
            for group in self.group_scores:
                result_df[f"{group}_Score"] = self.group_scores[group]
            
            # 添加综合得分
            result_df['Mental_Health_Index'] = self.composite_index
            
            # 添加原始压力等级和目标变量
            if self.raw_data is not None:
                result_df['stress_level'] = self.raw_data['stress_level']
                result_df['mental_health_history'] = self.raw_data['mental_health_history']
            
            # 保存为CSV
            result_df.to_csv('Mental_Health_Composite_Index.csv', index=False)
            print("✅ 综合指数文件已保存: Mental_Health_Composite_Index.csv")
            
            # 保存验证报告
            report_df = pd.DataFrame(self.validation_report)
            report_df.to_csv('Group_Score_Validation_Report.csv', index=False)
            print("✅ 验证报告已保存: Group_Score_Validation_Report.csv")
            
            # 保存权重信息
            weights_df = pd.DataFrame({
                'Dimension': self.higher_order_weights.index,
                'Weight': self.higher_order_weights.values
            })
            weights_df.to_csv('Dimension_Weights.csv', index=False)
            print("✅ 维度权重文件已保存: Dimension_Weights.csv")
            
            return True
        except Exception as e:
            print(f"❌ 结果保存失败: {str(e)}")
            traceback.print_exc()
            return False

    def run(self):
        """执行完整流程"""
        print("\n" + "="*50)
        print("=== 开始二次融合（高阶因子模型） ===")
        print("="*50)
        
        # 步骤1: 加载各组得分
        if not self.load_group_scores():
            print("❌ 得分加载失败，终止流程")
            return False
        
        # 步骤2: 标准化得分
        self.standardize_scores()
        
        # 步骤3: 拟合高阶因子模型
        if not self.fit_higher_order_model():
            print("❌ 高阶模型拟合失败，终止流程")
            return False
        
        # 步骤4: 计算综合指数
        self.calculate_composite_index()
        
        # 步骤5: 可视化结果
        self.visualize_results()
        
        # 步骤6: 保存结果
        self.save_results()
        
        print("\n" + "="*50)
        print("=== 二次融合成功完成 ===")
        print("="*50)
        return True

if __name__ == '__main__':
    analyzer = MentalHealthAnalyzer()
    if analyzer.run():
        print("心理健康综合评估完成！")
    else:
        print("流程执行失败，请检查错误日志。")