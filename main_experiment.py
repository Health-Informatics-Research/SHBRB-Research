# main_experiment.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from config import Config
from hierarchy_manager import HierarchyManager
from rule_generator import RuleGenerator
from rule_optimizer import RuleOptimizer # 导入优化器
from brb_engine import BRBEngine
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # 1. 加载数据
    data = pd.read_csv(Config.FUSED_DATA)
    # 确保目标变量是数值类型
    data['stress_level'] = data['stress_level'].astype(int)
    logging.info(f"加载BRB输入数据 {len(data)} 条。")

    # 2. 初始化管理器和生成器
    hm = HierarchyManager()
    ref_points = hm.generate_ref_points(data)
    
    rule_gen = RuleGenerator()
    initial_rules = rule_gen.generate_all_rules()

    # 3. 划分数据集
    X = data[hm.hierarchy['micro_antecedents'] + ['mental_health_history']]
    y = data[hm.hierarchy['target']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    logging.info(f"数据集划分: 训练={len(train_data)}, 测试={len(test_data)}")

    # 4. 【核心步骤】优化规则
    optimizer = RuleOptimizer(hm)
    # 使用训练数据对初始规则进行优化
    final_rules = optimizer.optimize_rules(initial_rules, train_data, ref_points, max_iter=50)

    # 5. 使用优化后的规则初始化引擎，并在测试集上进行评估
    engine = BRBEngine(final_rules, ref_points)
    
    predictions = []
    for _, row in test_data.iterrows(): # 使用测试集的X部分
        sample = row.to_dict()
        pred_label, _ = engine.infer(sample)
        predictions.append(pred_label)

    # 转换测试集的真实标签以进行比较
    y_test_labels = test_data[hm.hierarchy['target']].map({0: '低', 1: '中', 2: '高'})
    
    print("\n" + "="*25 + " 最终模型评估 (优化后) " + "="*25)
    report = classification_report(y_test_labels, predictions, labels=['低', '中', '高'])
    print(f"在测试集上的分类报告:\n{report}")

if __name__ == "__main__":
    main()