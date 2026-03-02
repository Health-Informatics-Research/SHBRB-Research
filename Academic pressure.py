"""
ER融合实验：学业组SEM模型权重分配
"""
# 1. 导入库
import pandas as pd
import numpy as np
from semopy import Model, calc_stats
from sklearn.preprocessing import StandardScaler
import warnings
import joblib
from pathlib import Path

# 忽略semopy可能产生的FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# 2. 数据加载与预处理
def load_and_preprocess():
    try:
        # 加载数据
        try:
            df = pd.read_csv('preprocessed_data.csv')
        except FileNotFoundError:
            print(" 错误：未找到数据文件 'preprocessed_data.csv'。请确保文件存在于脚本运行目录。")
            exit()

        print(f"✅ 数据加载成功（样本量：{len(df)}）")

        # 关键变量检查
        required_vars = ['academic_performance',         # 学业表现
                         'study_load',                   # 学习负担
                         'teacher_student_relationship', # 师生关系
                         'future_career_concerns']       # 未来职业的担忧
        missing_vars = [var for var in required_vars if var not in df.columns]
        if missing_vars:
            raise ValueError(f"缺失必要字段：{missing_vars}")

        # 确保关键变量是数值类型
        for var in required_vars:
             if not pd.api.types.is_numeric_dtype(df[var]):
                 try:
                     df[var] = pd.to_numeric(df[var])
                     print(f" 变量 '{var}' 已转换为数值类型。")
                 except ValueError:
                     raise TypeError(f"变量 '{var}' 包含无法转换为数值的值，请检查数据。")


        # 数据描述
        print("\n=== 变量描述统计 ===")
        print(df[required_vars].describe().round(2))

        # 缺失值处理
        initial_count = len(df)
        df_clean = df[required_vars].dropna()
        final_count = len(df_clean)

        if final_count < initial_count:
            print(f"\n⚠️ 删除含有缺失值的样本：{initial_count - final_count}个。剩余样本量：{final_count}")
        if final_count == 0:
             raise ValueError("所有样本都包含缺失值，无法进行分析。")

        return df_clean

    except Exception as e:
        print(f"\n 数据预处理失败：{str(e)}")
        exit()

# 3. SEM模型定义
def define_sem_model():
    # 修正后的测量模型（交换参照指标）
    model_spec = '''
    # 测量模型（固定职业担忧为正向指标）
    Academic_Stress =~ 1*future_career_concerns + academic_performance

    # 结构模型保持不变
    Academic_Stress ~ study_load + teacher_student_relationship

    # 允许预测变量相关
    study_load ~~ teacher_student_relationship
    '''
    print("\n=== SEM模型定义 ===")
    print(model_spec.strip())
    return model_spec

# 4. 模型拟合与诊断
def fit_sem_model(data, model_spec):
    try:
        # 数据标准化
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(data),
                                 columns=data.columns,
                                 index=data.index) # 保留原始索引

        # 模型初始化与拟合
        mod = Model(model_spec)
        # fit 方法返回一个结果对象
        res = mod.fit(df_scaled, obj='MLW') # 使用 MLW (Wishart) 目标函数

        # 检查拟合结果对象是否存在
        if res is None:
             raise RuntimeError("模型拟合未能返回结果对象。")

        # 从结果对象获取目标函数值
        # 检查 res 对象是否有 fun 属性
        if hasattr(res, 'fun'):
            fun_val = res.fun
            print(f"\n 模型拟合完成（目标函数值：{fun_val:.4f}）")
        else:
            print("\n 未能从拟合结果中获取目标函数值 (fun 属性不存在)。")
            # 可以尝试其他可能的方式或跳过打印

        # 简单的检查：查看参数估计是否合理（例如，没有NaN）
        params_check = mod.inspect()
        if params_check['Estimate'].isnull().any():
            print("\n 模型拟合可能存在问题：参数估计中包含NaN值。")
            # 可能需要进一步处理

        # 适配度指标
        print("\n=== 模型适配度指标计算 ===")
        try:
            stats = calc_stats(mod)
            print("\n=== 模型适配度指标（可用项） ===")
            stats = stats.dropna(axis=1, how='all')
            print(stats.T.round(3))
        except Exception as e:
            print(f" 计算适配度指标时出错：{e}")
            print("   继续执行，但适配度指标可能不完整。")

         # pkl修改后的保存逻辑
        model_dir = Path("first_fusion_models")
        model_dir.mkdir(exist_ok=True)
    
        # pkl只保存可序列化的参数
        params = mod.inspect(std_est=True)
        
        # 确保提取的路径包含Academic_Stress
        # ============== 修改开始 ============== 
        # 提取测量模型参数（=~ 操作符对应的参数）
        measurement_indicators = ['academic_performance', 'future_career_concerns']

        indicators = ['academic_performance', 'future_career_concerns']
        # 直接使用测量模型操作符
        measurement_coef = params.loc[
            (params['op'] == '=~') &
            (params['lval'] == 'Academic_Stress') &
            (params['rval'].isin(indicators)),
            ['rval', 'Est. Std']].rename(columns={'rval':'indicator'})

        # 提取结构模型参数（潜变量被预测的部分）
        path_coef = params[
            (params['op'] == '~') &
            (params['lval'] == 'Academic_Stress')
        ][['rval', 'Est. Std']].rename(columns={'rval': 'predictor'})

        # 保存完整的模型参数
        joblib.dump(
            {
                'measurement_coef': measurement_coef,  # 测量模型参数（含符号修正）
                'path_coef': path_coef                # 结构模型参数（保持不变）
            }, 
        model_dir / "Academic_sem_params.pkl"
        )
    # ============== 修改结束 ==============

        return mod, df_scaled, scaler # 返回模型对象、标准化数据和scaler

    except AttributeError as ae: # 捕获特定的 AttributeError
         print(f"\n 模型拟合时发生属性错误：{str(ae)}")
         import traceback
         traceback.print_exc()
         return None, None, None
    except Exception as e:
        print(f"\n 模型拟合错误：{str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

# 5. 权重计算系统 
def calculate_weights(model): 
    if model is None:
        print("\n 模型对象无效，无法计算权重。")
        return None

    try:
        # 获取标准化参数估计
        params = model.inspect(std_est=True)

        # 修正: 提取测量模型参数 (indicator ~ latent 形式)
        # 查找操作符为 '~' 且右侧变量 (rval) 是潜变量 'Academic_Stress' 的行
        loadings_df = params[
            (params['op'] == '~') &
            (params['rval'] == 'Academic_Stress') &
            (params['lval'].isin(['academic_performance', 'future_career_concerns'])) # 确保只选指标
        ][['lval', 'Est. Std']].copy() # 提取指标 (lval) 和标准化估计

        # 检查是否找到了预期的指标载荷
        if loadings_df.empty:
            raise ValueError("未能在参数表中找到 'indicator ~ Academic_Stress' 的测量模型参数。"
                             "请仔细检查模型定义和 inspect() 输出。")

        expected_indicators = {'academic_performance', 'future_career_concerns'}
        found_indicators = set(loadings_df['lval'])
        if found_indicators != expected_indicators:
             print(f" 警告：找到的指标 ({found_indicators}) 与模型定义的指标 ({expected_indicators}) 不完全匹配。")
             # 可能需要根据实际情况决定是否继续

        # 将 lval 重命名为 'indicator' 以提高可读性
        loadings_df = loadings_df.rename(columns={'lval': 'indicator'})

        # 转换为字典: {indicator: std_loading}
        # 处理可能存在的 NaN 标准化载荷 
        if loadings_df['Est. Std'].isnull().any():
            print(" 警告：标准化载荷中存在 NaN 值，将跳过这些指标的权重计算。")
            loadings_df = loadings_df.dropna(subset=['Est. Std'])
            if loadings_df.empty:
                 raise ValueError("所有标准化载荷均为 NaN，无法计算权重。")

        weights = loadings_df.set_index('indicator')['Est. Std']

        # 使用平方和归一化（更科学）
        total_sq_weight = (weights**2).sum()**0.5
        if total_sq_weight > 0:
            norm_weights = (weights / total_sq_weight).round(3)
        else:
            norm_weights = weights.round(3)
        
        # 打印原始载荷供验证
        print("\n=== 原始标准化载荷 ===")
        print(loadings_df[['indicator', 'Est. Std']])
        
        return norm_weights.to_dict()

    except Exception as e:
        print(f"\n❌ 权重计算失败：{str(e)}")
        import traceback
        traceback.print_exc()
        return None

# 6. 得分计算 (修正版 - 返回标准化得分)
def calculate_scores(weights, scaled_data): # 不再需要 raw_data 和 scaler
    if not weights:
        print("\n⚠️ 权重无效，无法计算得分。")
        return None

    # 检查权重中的变量是否存在于 scaled_data 的列中
    missing_cols = [var for var in weights.keys() if var not in scaled_data.columns]
    if missing_cols:
        print(f"\n❌ 得分计算错误：以下变量在 scaled_data 中缺失：{missing_cols}")
        return None

    try:
        # 计算标准化得分 (加权和)
        # 使用 0 初始化 Series，确保索引与 scaled_data 一致
        score_scaled = pd.Series(0.0, index=scaled_data.index)
        print("\n=== 计算标准化得分 ===")
        for var, w in weights.items():
            print(f"  -> 添加项: {w:.3f} * {var}")
            score_scaled += w * scaled_data[var]

        print("✅ 标准化得分计算完成。")
        # 返回标准化的得分 Series
        return score_scaled.round(6)   #!!!!结果保留小数位数

    except Exception as e:
        print(f"\n❌ 得分计算错误：{str(e)}")
        return None

# 主执行流程
if __name__ == '__main__':
    print("--- 开始执行 SEM 分析流程 ---")

    # 数据准备
    df_raw = load_and_preprocess() # 得到清洗后的原始数据

    # 模型配置
    model_spec = define_sem_model()

    # 模型拟合
    # fit_sem_model 返回的是 model 对象, 标准化后的数据 df_scaled, 和 scaler 对象
    mod, df_scaled, scaler = fit_sem_model(df_raw, model_spec)

    # 检查拟合是否成功
    if mod is None or df_scaled is None:
         print("\n--- 模型拟合失败，流程终止 ---")
         exit()

    # 权重计算
    weights = calculate_weights(mod) # 传入 model 对象

    # 得分生成
    if weights:
        # 使用标准化数据计算得分
        academic_score = calculate_scores(weights, df_scaled)

        if academic_score is not None:
            # 将标准化得分添加到原始数据 DataFrame 中 (注意索引对齐)
            # df_raw 是预处理（dropna）后的数据，df_scaled 的索引与之对应
            df_output = df_raw.copy() # 创建副本以添加得分
            df_output['Academic pressure_score'] = academic_score

            # 结果保存
            output_path = 'Academic pressure_score.csv'
            try:
                df_output.to_csv(output_path, index=False)
                print(f"\n 包含标准化得分的结果已保存至：{output_path}")

                # 得分分布验证 (现在是标准化得分)
                print("\n=== 学业压力标准化得分分布 ===")
                print(df_output['Academic pressure_score'].describe().round(3))
            except Exception as e:
                 print(f"\n❌ 保存结果文件失败：{str(e)}")

        else:
            print("\n 未能成功计算得分。")
    else:
        print("\n 未能成功计算权重，无法生成得分。")

    print("\n--- SEM 分析流程结束 ---")
   

