# hierarchy_manager.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from config import Config
import logging

class HierarchyManager:
    def __init__(self):
        self.hierarchy = Config.HIERARCHY
        self.ref_points = {}
        logging.info("分层管理器初始化完成。")

    def generate_ref_points(self, data: pd.DataFrame):
        self.ref_points['stress_level'] = {'labels': ['低', '中', '高'], 'ref_values': [0, 1, 2]}
        self.ref_points['General_Adversity'] = {'labels': ['低', '中', '高']}
        self.ref_points['mental_health_history'] = {'labels': ['无', '有'], 'ref_values': [0, 1]}

        for var in self.hierarchy['micro_antecedents']:
            if var in data.columns:
                discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
                binned_data = discretizer.fit_transform(data[[var]])
                
                ref_values = [np.median(data.loc[binned_data.ravel() == i, var]) for i in range(3)]
                self.ref_points[var] = {
                    'labels': ['低', '中', '高'],
                    'ref_values': ref_values
                }
        logging.info("已为所有前提变量生成参考点。")
        return self.ref_points