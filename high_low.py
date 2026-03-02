import pandas as pd
import logging
from config import Config
from hierarchy_manager import HierarchyManager
from rule_generator import RuleGenerator
from rule_optimizer import RuleOptimizer
from brb_engine import BRBEngine

# --- 日志配置 ---
# 配置日志，使其直接打印在控制台，方便查看
logging.basicConfig(level=logging.INFO, format='%(asctime)s - INFO - %(message)s')

def train_final_model(full_data, ref_points):
    """
    在全部数据上训练一个最终模型，用于可解释性分析。
    """
    logging.info("开始在完整数据集上训练最终模型...")
    # 假设我们总是使用SEM知识来训练最终模型
    rule_gen = RuleGenerator()
    initial_rules = rule_gen.generate_rules(use_sem_knowledge=True)
    
    optimizer = RuleOptimizer(hm)
    # 这里的max_iter可以根据需要调整，30次通常足够
    final_rules = optimizer.optimize_rules(initial_rules, full_data, ref_points, max_iter=30, seed=42)
    logging.info("最终模型训练完成。")
    return final_rules

# --- 主程序入口 ---
if __name__ == "__main__":
    
    # --- 1. 初始化和数据准备 ---
    hm = HierarchyManager()
    data = pd.read_csv(Config.FUSED_DATA)
    data['stress_level'] = data['stress_level'].astype(int)
    ref_points = hm.generate_ref_points(data)

    # --- 2. 训练最终模型 ---
    # 我们需要一个训练好的模型来进行案例分析
    final_trained_rules = train_final_model(data, ref_points)
    
    # --- 3. 创建推理引擎 ---
    # 确保使用启用病史调节的模式
    engine = BRBEngine(final_trained_rules, ref_points, use_history_moderator=True)

    # --- 4. 定义并分析案例 ---
    # 这是您之前日志中确定的高风险和低风险样本的索引号
    # 您可以修改这个列表来分析任何您感兴趣的样本
    case_study_indices = [628, 736] 

    print("\n" + "="*25 + " 可解释性案例分析 " + "="*25)

    for i, sample_index in enumerate(case_study_indices):
        print(f"\n--- 案例 {i+1}：分析样本索引 {sample_index} ---")
        
        # 从数据集中提取单个样本
        sample_series = data.iloc[sample_index]
        sample_dict = sample_series.to_dict()
        
        print("分析样本信息:")
        print(sample_series)
        
        # --- 5. 【核心】调用带追踪的推理功能 ---
        print("\nBRB推理过程追踪:")
        trace_output = engine.infer(sample_dict, trace=True)
        
        # 打印追踪结果
        print(trace_output)