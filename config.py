# config.py
class Config:
    RAW_DATA = "preprocessed_data.csv"
    FUSED_DATA = "Mental_Health_Composite_Index.csv"
    
    SEM_MACRO_RULES_FILE = "final_sem_main_rules.csv"
    SEM_MICRO_RULES_FILE = "path_model_for_rules.csv"

    HIERARCHY = {
        'micro_antecedents': [
            'Psychological_Score', 'Physical_Score', 'Environmental_Score', 
            'Academic_Score', 'Social_Score'
        ],
        'macro_antecedents': ['General_Adversity', 'mental_health_history'],
        'intermediate_variable': 'General_Adversity',
        'target': 'stress_level'
    }
    
    BELIEF_DISTRIBUTION_TEMPLATE = {
        'strong_positive': [0.1, 0.2, 0.7],
        'medium_positive': [0.2, 0.4, 0.4],
        'strong_negative': [0.7, 0.2, 0.1],
        'medium_negative': [0.4, 0.4, 0.2],
    }

    BRB_RULES_FINAL = "models/brb_rules_final.pkl"