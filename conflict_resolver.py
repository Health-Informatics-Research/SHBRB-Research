# conflict_resolver.py
from collections import defaultdict
import numpy as np
import logging
from scipy.spatial.distance import jensenshannon

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ConflictResolver:
    def __init__(self):
        self.conflict_rules = set()
        logging.info("冲突解决器初始化完成")
    
    def detect_rule_conflicts(self, rules):
        self.conflict_rules = set()
        CONFLICT_THRESHOLD = 0.5
        
        for i, r1 in enumerate(rules):
            for j, r2 in enumerate(rules):
                if i >= j: 
                    continue
                    
                if set(r1['antecedents']) == set(r2['antecedents']) and r1['consequent'] == r2['consequent']:
                    js_div = jensenshannon(r1['belief'], r2['belief'])
                    
                    if js_div > CONFLICT_THRESHOLD:
                        self.conflict_rules.add(r1['id'])
                        self.conflict_rules.add(r2['id'])
                        r1['is_conflict'] = True
                        r2['is_conflict'] = True
        
        if self.conflict_rules:
            logging.warning(f"检测到 {len(self.conflict_rules)} 条冲突规则")
        else:
            logging.info("未检测到规则冲突")
        
        return self.conflict_rules

    def resolve_rule_conflicts(self, rules):
        if not self.conflict_rules:
            return
        
        non_conflict_rules = []
        conflict_groups = defaultdict(list)
        
        for rule in rules:
            if rule['id'] in self.conflict_rules:
                key = tuple(sorted(rule['antecedents']))
                conflict_groups[key].append(rule)
            else:
                non_conflict_rules.append(rule)
        
        for key, group in conflict_groups.items():
            best_rule = max(group, key=lambda x: (
                x.get('priority', 0),
                x['weight'],
                -len(x['antecedents'])
            ))
            non_conflict_rules.append(best_rule)
            logging.info(f"解决冲突: 保留 {best_rule['id']} (权重={best_rule['weight']:.2f})")
        
        rules.clear()
        rules.extend(non_conflict_rules)
        logging.info(f"保留 {len(rules)} 条无冲突规则")