import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from config import Config
from hierarchy_manager import HierarchyManager
from rule_generator import RuleGenerator
from rule_optimizer import RuleOptimizer
from brb_engine import BRBEngine
import logging
import os

# --- 全局设置 ---
GLOBAL_SEED = 42
OUTPUT_PREDICTIONS_FILE = "all_folds_predictions.csv"
OUTPUT_WEIGHTS_FILE = "all_folds_rule_weights.csv"

# --- 全局对象 ---
hm = None

# --- 函数定义 ---
def train_brb(train_data, ref_points, use_sem_knowledge):
    """封装BRB规则生成和优化的过程"""
    rule_gen = RuleGenerator()
    initial_rules = rule_gen.generate_rules(use_sem_knowledge=use_sem_knowledge)
    
    optimizer = RuleOptimizer(hm)
    # 矩阵加速后，max_iter=30 会飞快完成
    final_rules = optimizer.optimize_rules(initial_rules, train_data, ref_points, max_iter=30, seed=GLOBAL_SEED)
    return final_rules

def evaluate_brb(engine, test_data):
    """
    [修改] 使用 batch_infer 加速评估过程
    同时返回预测标签列表和概率分布数组
    """
    # 直接调用矩阵化接口，速度提升100倍
    predictions, probabilities = engine.batch_infer(test_data)
    
    # predictions 已经是 list 或 array
    # probabilities 是 numpy array
    return predictions, probabilities

def main():
    """主函数：执行10折交叉验证并保存所有结果"""
    global hm
    hm = HierarchyManager()

    # --- 1. 数据准备 ---
    if not os.path.exists(Config.FUSED_DATA):
        print(f"错误：找不到数据文件 {Config.FUSED_DATA}，请先运行数据融合脚本。")
        return

    data = pd.read_csv(Config.FUSED_DATA)
    data['stress_level'] = data['stress_level'].astype(int)
    ref_points = hm.generate_ref_points(data)
    X = data.drop('stress_level', axis=1)
    y = data['stress_level']
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='experiment_log.log', filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info("="*20 + " 开始10折交叉验证 (矩阵加速版 + 概率保存) " + "="*20)

    # --- 2. 交叉验证设置 ---
    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=GLOBAL_SEED)
    
    all_results_list = []
    all_weights_list = []
    
    fold_num = 1
    # --- 3. 执行10折交叉验证循环 ---
    for train_index, test_index in skf.split(X, y):
        logging.info(f"\n--- 正在进行第 {fold_num}/{n_splits} 折 ---")
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # 必须重置索引，否则concat会出错
        train_data = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
        test_data = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)
        
        # --- 训练 ---
        logging.info(f"[Fold {fold_num}] 正在为SEM-Based模型训练...")
        sem_based_rules = train_brb(train_data, ref_points, use_sem_knowledge=True)
        
        logging.info(f"[Fold {fold_num}] 正在为Heuristic模型训练...")
        heuristic_rules = train_brb(train_data, ref_points, use_sem_knowledge=False)

        # --- 记录当前折的规则权重 ---
        for rule in sem_based_rules:
            all_weights_list.append({
                'fold': fold_num,
                'model_type': 'SEM-Based',
                'rule_name': rule['antecedents'][0],
                'weight': rule['weight']
            })
        for rule in heuristic_rules:
             all_weights_list.append({
                'fold': fold_num,
                'model_type': 'Heuristic',
                'rule_name': rule['antecedents'][0],
                'weight': rule['weight']
            })
        
        # --- 评估 ---
        # 1. 模型A: SH-BRB
        engine_A = BRBEngine(sem_based_rules, ref_points, use_history_moderator=True)
        preds_A, probs_A = evaluate_brb(engine_A, test_data) # 获取概率
        
        # 2. 模型B: No History
        engine_B = BRBEngine(sem_based_rules, ref_points, use_history_moderator=False)
        preds_B, probs_B = evaluate_brb(engine_B, test_data)
        
        # 3. 模型C: Heuristic
        engine_C = BRBEngine(heuristic_rules, ref_points, use_history_moderator=True)
        preds_C, probs_C = evaluate_brb(engine_C, test_data)
        
        # 4. 模型D: Random Forest
        rf_features = hm.hierarchy['micro_antecedents'] + ['mental_health_history']
        rf_model = RandomForestClassifier(n_estimators=100, random_state=GLOBAL_SEED, class_weight='balanced')
        # RF需要原始矩阵格式
        rf_model.fit(X_train[rf_features], y_train)
        
        preds_rf_indices = rf_model.predict(X_test[rf_features])
        preds_D = [ref_points['stress_level']['labels'][i] for i in preds_rf_indices]
        probs_rf = rf_model.predict_proba(X_test[rf_features])

        # 保存结果 (概率取索引2，即高压力)
        fold_results = pd.DataFrame({
            'fold': fold_num,
            'true_label': y_test.map({0: '低', 1: '中', 2: '高'}).values, # 确保也是array
            'pred_A': preds_A,
            'pred_B': preds_B,
            'pred_C': preds_C,
            'pred_D': preds_D,
            'prob_high_A': probs_A[:, 2], 
            'prob_high_B': probs_B[:, 2],
            'prob_high_C': probs_C[:, 2],
            'prob_high_D': probs_rf[:, 2] 
        })
        all_results_list.append(fold_results)
        
        logging.info(f"第 {fold_num} 折完成.")
        fold_num += 1

    # --- 4. 保存所有结果到CSV文件 ---
    final_results_df = pd.concat(all_results_list, ignore_index=True)
    final_results_df.to_csv(OUTPUT_PREDICTIONS_FILE, index=False)
    logging.info("\n" + "="*25 + f" 所有预测结果已成功保存到 {OUTPUT_PREDICTIONS_FILE} " + "="*25)
    
    final_weights_df = pd.DataFrame(all_weights_list)
    final_weights_df.to_csv(OUTPUT_WEIGHTS_FILE, index=False)
    logging.info("="*25 + f" 所有规则权重已成功保存到 {OUTPUT_WEIGHTS_FILE} " + "="*25)

if __name__ == "__main__":
    main()