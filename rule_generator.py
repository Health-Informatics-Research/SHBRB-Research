# rule_generator.py
import pandas as pd
from config import Config
import logging

class RuleGenerator:
    def __init__(self):
        try:
            self.macro_paths = pd.read_csv(Config.SEM_MACRO_RULES_FILE)
            self.micro_paths = pd.read_csv(Config.SEM_MICRO_RULES_FILE)
            self.sem_knowledge_loaded = True
        except FileNotFoundError:
            self.sem_knowledge_loaded = False
            logging.warning("SEM结果文件未找到，将只能生成启发式规则。")
        
        self.belief_template = Config.BELIEF_DISTRIBUTION_TEMPLATE
        logging.info("分层规则生成器初始化完成。")

    def generate_rules(self, use_sem_knowledge=True):
        if use_sem_knowledge and not self.sem_knowledge_loaded:
            raise ValueError("请求使用SEM知识，但SEM结果文件加载失败。")

        rules = []
        if use_sem_knowledge:
            rules.extend(self._generate_sem_macro_rules())
            rules.extend(self._generate_sem_micro_rules())
            logging.info(f"成功生成 {len(rules)} 条基于SEM知识的分层规则。")
        else:
            rules.extend(self._generate_heuristic_macro_rules())
            rules.extend(self._generate_heuristic_micro_rules())
            logging.info(f"成功生成 {len(rules)} 条基于启发式知识的分层规则。")
        
        for i, rule in enumerate(rules):
            rule['id'] = f"rule_{i:03d}"
        return rules

    def _generate_sem_macro_rules(self):
        macro_rules = []
        for _, row in self.macro_paths.iterrows():
            antecedent = row['rhs']
            weight = abs(row['std.all'])
            belief = self.belief_template['strong_positive']
            macro_rules.append({
                'antecedents': [antecedent], 'consequent': Config.HIERARCHY['target'],
                'weight': weight,
                'belief_distributions': {'高': belief, '中': [0.2, 0.6, 0.2], '低': belief[::-1]},
                'type': 'macro_sem'
            })
        return macro_rules

    def _generate_sem_micro_rules(self):
        micro_rules = []
        for _, row in self.micro_paths.iterrows():
            antecedent = row['rhs']
            weight = abs(row['std.all'])
            is_positive = row['std.all'] > 0
            micro_rules.append({
                'antecedents': [antecedent], 'consequent': Config.HIERARCHY['intermediate_variable'],
                'weight': weight, 'is_positive': is_positive, 'type': 'micro_sem'
            })
        return micro_rules

    def _generate_heuristic_macro_rules(self):
        macro_rules = []
        belief = self.belief_template['strong_positive']
        for antecedent in Config.HIERARCHY['macro_antecedents']:
            macro_rules.append({
                'antecedents': [antecedent], 'consequent': Config.HIERARCHY['target'],
                'weight': 1.0,
                'belief_distributions': {'高': belief, '中': [0.2, 0.6, 0.2], '低': belief[::-1]},
                'type': 'macro_heuristic'
            })
        return macro_rules

    def _generate_heuristic_micro_rules(self):
        micro_rules = []
        for antecedent in Config.HIERARCHY['micro_antecedents']:
            micro_rules.append({
                'antecedents': [antecedent], 'consequent': Config.HIERARCHY['intermediate_variable'],
                'weight': 1.0,
                'is_positive': True,
                'type': 'micro_heuristic'
            })
        return micro_rules