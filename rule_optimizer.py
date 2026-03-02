import copy
from sklearn.metrics import f1_score
from scipy.optimize import differential_evolution
from brb_engine import BRBEngine
import logging
import pandas as pd

class RuleOptimizer:
    def __init__(self, hierarchy_manager):
        self.hm = hierarchy_manager
        self.data_for_objective = None
        self.rules_for_objective = None
        self.ref_points_for_objective = None
        logging.info("规则优化器初始化完成 (矩阵加速版)。")

    def _objective_function(self, weights):
        # 1. 将优化算法生成的权重应用到规则上
        current_rules = self.apply_weights(weights, copy.deepcopy(self.rules_for_objective))
        
        # 2. [核心加速点] 使用 batch_infer 进行全量预测
        # 初始化引擎 (这一步很快，只是传递配置)
        engine = BRBEngine(current_rules, self.ref_points_for_objective, use_history_moderator=True)
        
        # 直接传入整个 DataFrame 进行矩阵计算
        # y_pred 是字符串标签列表 ['高', '低'...]
        y_pred, _ = engine.batch_infer(self.data_for_objective)
        
        # 3. 计算指标 (F1 Score)
        # 这里的 y_true 是数字 [0,1,2]，需要映射成文字
        target_col = self.hm.hierarchy['target']
        y_true_labels = self.data_for_objective[target_col].map({0: '低', 1: '中', 2: '高'})
        
        f1 = f1_score(y_true_labels, y_pred, average='weighted', zero_division=0)
        
        # 差分进化是求最小值，所以返回 1-F1
        return 1 - f1

    def optimize_rules(self, rules, train_data, ref_points, max_iter=30, seed=None):
        logging.info(f"开始对 {len(rules)} 条规则进行权重优化 (矩阵加速模式, max_iter={max_iter})...")
        
        self.data_for_objective = train_data # 这是一个DataFrame
        self.rules_for_objective = copy.deepcopy(rules)
        self.ref_points_for_objective = ref_points
        
        # 设定权重范围
        bounds = [(0.01, 2.5) for _ in self.rules_for_objective]
        initial_weights = [r['weight'] for r in self.rules_for_objective]
        
        # workers=-1 启用多核并行，配合矩阵计算速度极快
        result = differential_evolution(
            self._objective_function, bounds, maxiter=max_iter, popsize=15, 
            tol=0.01, workers=-1, disp=True, x0=initial_weights, seed=seed
        )
        
        final_rules = self.apply_weights(result.x, rules)
        logging.info(f"权重优化完成！训练集最优F1: {1 - result.fun:.4f}")
        return final_rules

    def apply_weights(self, weights, rules):
        for i, rule in enumerate(rules):
            rule['weight'] = weights[i]
        return rules