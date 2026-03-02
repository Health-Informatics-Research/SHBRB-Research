import numpy as np
import pandas as pd
from config import Config
import logging

class BRBEngine:
    def __init__(self, rules, ref_points, use_history_moderator=True):
        self.rules = rules
        self.ref_points = ref_points
        self.use_history_moderator = use_history_moderator
        self.macro_rules = [r for r in rules if r['type'].startswith('macro')]
        self.micro_rules = [r for r in rules if r['type'].startswith('micro')]
        # 初始化日志只打印一次，避免刷屏
        # logging.info("分层BRB推理引擎初始化完成。")

    def batch_infer(self, df):
        """
        [核心优化] 向量化批量推理接口
        输入: df (DataFrame), 包含所有特征数据
        输出: predictions (list), final_beliefs (numpy array)
        原理: 利用NumPy广播机制替代for循环，数学结果完全一致
        """
        N = len(df)
        if N == 0: return [], np.array([])
        
        # 1. 准备上下文数据 (N,)
        history = df.get('mental_health_history', pd.Series(np.zeros(N))).values
        
        # --- 微观推理阶段 (Vectorized) ---
        # 预分配矩阵 (N, 3) 用于累加
        micro_belief_agg = np.zeros((N, 3))
        micro_weight_sum = np.zeros(N)
        
        for rule in self.micro_rules:
            antecedent = rule['antecedents'][0]
            # 获取整列输入数据 (N,)
            vals = df[antecedent].values
            
            # --- 计算激活权重 (Gaussian RBF) ---
            # refs: 变为 (1, 3) 以便广播
            refs = np.array(self.ref_points[antecedent]['ref_values'])[None, :]
            # vals: 变为 (N, 1)
            # 计算欧氏距离平方: (N, 1) - (1, 3) -> (N, 3)
            dist_sq = (vals[:, None] - refs) ** 2
            # 高斯激活
            activations = np.exp(-dist_sq) 
            
            # --- 归一化激活度 ---
            row_sums = activations.sum(axis=1)
            # 处理全0情况 (避免除以0)
            mask_nonzero = row_sums > 0
            # 只有和>0的行才做除法
            activations[mask_nonzero] /= row_sums[mask_nonzero, None]
            # 和为0的行设为均匀分布 (与原代码逻辑一致)
            activations[~mask_nonzero] = [1/3, 1/3, 1/3]
            
            # --- 规则权重动态调整 ---
            # 创建全为 rule['weight'] 的数组 (N,)
            w = np.full(N, rule['weight'])
            
            if self.use_history_moderator:
                # 向量化条件判断: 有病史 且 属于敏感规则
                if antecedent in ['Academic_Score', 'Psychological_Score']:
                    mask_mod = (history == 1)
                    w[mask_mod] *= 1.15
            
            # --- 置信度分布方向映射 ---
            # belief: (N, 3)
            if rule.get('is_positive', True):
                belief = activations
            else:
                belief = activations[:, ::-1] # 反转列顺序 [2,1,0]
            
            # --- 证据融合 (ER算法的简化加权部分) ---
            # w[:, None] 将 (N,) 广播为 (N, 1) 以便与 (N, 3) 相乘
            micro_belief_agg += belief * w[:, None]
            micro_weight_sum += w
            
        # --- 计算微观输出 (综合逆境指数) ---
        mask_w_zero = micro_weight_sum == 0
        # 避免除0，暂时设为1
        micro_weight_sum[mask_w_zero] = 1.0 
        adversity_belief = micro_belief_agg / micro_weight_sum[:, None]
        # 权重和为0的样本设为默认值
        adversity_belief[mask_w_zero] = [1/3, 1/3, 1/3]
        
        # --- 宏观推理阶段 (Vectorized) ---
        macro_belief_agg = np.zeros((N, 3))
        macro_weight_sum = np.zeros(N)
        
        for rule in self.macro_rules:
            antecedent = rule['antecedents'][0]
            w = np.full(N, rule['weight'])
            
            if antecedent == 'General_Adversity':
                # 规则输入是上一层的输出 (N, 3)
                input_belief = adversity_belief
                
                # 构建变换矩阵 T: (3, 3)
                # 原逻辑: activations[0]*dist_low + activations[1]*dist_mid ...
                # 这等价于矩阵乘法
                bd = rule['belief_distributions']
                # T的行分别对应 [低分布, 中分布, 高分布]
                T = np.array([bd['低'], bd['中'], bd['高']])
                
                # 矩阵乘法: (N, 3) @ (3, 3) -> (N, 3)
                rule_output_belief = input_belief @ T
                
            else: # mental_health_history
                vals = df[antecedent].values # (N,) 0或1
                
                # 权重调节
                if self.use_history_moderator:
                    mask_hist = (vals == 1)
                    w[mask_hist] *= 1.2
                
                # 构造输出分布
                bd = rule['belief_distributions']
                dist_high = np.array(bd['高'])
                dist_low = np.array(bd['低'])
                
                # 快速构造 (N, 3)
                rule_output_belief = np.zeros((N, 3))
                # vals==0 的行赋 dist_low
                rule_output_belief[vals == 0] = dist_low
                # vals==1 的行赋 dist_high
                rule_output_belief[vals == 1] = dist_high
            
            # 聚合
            macro_belief_agg += rule_output_belief * w[:, None]
            macro_weight_sum += w
            
        # --- 最终聚合 ---
        mask_mw_zero = macro_weight_sum == 0
        macro_weight_sum[mask_mw_zero] = 1.0
        final_beliefs = macro_belief_agg / macro_weight_sum[:, None]
        final_beliefs[mask_mw_zero] = [1/3, 1/3, 1/3]
        
        # --- 决策阈值应用 (Vectorized) ---
        # 计算得分: 中*1 + 高*2 (低*0)
        # scores: (N,)
        scores = final_beliefs[:, 1] + 2 * final_beliefs[:, 2]
        
        if self.use_history_moderator:
            # 动态阈值: 有病史用 (0.7, 1.3), 无病史用 (0.8, 1.5)
            thresh_mid = np.where(history == 1, 0.7, 0.8)
            thresh_high = np.where(history == 1, 1.3, 1.5)
        else:
            thresh_mid = np.full(N, 0.8)
            thresh_high = np.full(N, 1.5)
            
        pred_labels = np.zeros(N, dtype=int)
        # 逻辑判断
        # 0: Low (score < mid) - 默认为0
        # 1: Mid (mid <= score < high)
        mask_mid = (scores >= thresh_mid) & (scores < thresh_high)
        pred_labels[mask_mid] = 1
        # 2: High (score >= high)
        mask_high = (scores >= thresh_high)
        pred_labels[mask_high] = 2
        
        # 映射回字符串标签
        label_map = np.array(self.ref_points['stress_level']['labels'])
        predictions = label_map[pred_labels]
        
        return predictions, final_beliefs

    def infer(self, sample, trace=False):
        """
        兼容旧代码的单样本接口。
        内部将其包装成单行DataFrame调用batch_infer，确保逻辑唯一。
        """
        # 如果需要trace，这里还是保留原逻辑比较方便打印日志
        # 但为了保证实验结果绝对一致，建议实验时不开启trace
        df_single = pd.DataFrame([sample])
        preds, beliefs = self.batch_infer(df_single)
        
        if trace:
            # 如果需要trace，这里只是简单的返回结果，不打印详细路径
            # 因为矩阵化计算很难逐条打印trace log
            return f"Trace not available in accelerated mode. Result: {preds[0]}", beliefs[0]
            
        return preds[0], beliefs[0]