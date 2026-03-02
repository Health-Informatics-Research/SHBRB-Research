# SHBRB-Research
# SH-BRB: A Semantically-Grounded and Explainable AI Framework

> **Note to Reviewers:** This repository is entirely **anonymized** to strictly comply with the double-blind peer-review policy. All author identities, institutional affiliations, and identifying metadata have been intentionally removed.

## 📌 Overview
This repository contains the official implementation of the **SH-BRB (Structural Equation Modeling & Heuristic Belief Rule Base)** framework. 
The SH-BRB framework is a novel, theory-guided, and highly interpretable AI system designed for dynamic mental health assessment. It seamlessly integrates the psychometric structural validation of Structural Equation Modeling (SEM) with the transparent, non-linear reasoning capabilities of a Belief Rule Base (BRB) expert system.

## 🚀 Key Features
- **SEM-Driven Initialization:** Automated protocol to map factor structures and path coefficients from validated SEM into the BRB's initial topology and rule weights.
- **Microscopic Reasoning:** Gaussian Radial Basis Function (RBF) activation for fine-grained psychological feature extraction.
- **Macroscopic Context-Aware Modulation:** A dynamic mechanism that adaptively adjusts rule weights and high-risk decision boundaries based on individual contexts (e.g., medical history).
- **Interpretability:** 100% transparent reasoning paths ("White-box" model) ensuring clinical logic consistency.

## 📂 Repository Structure
```text
📦 SH-BRB-Research
 ┣ 📂 data/               # Placeholder for dataset (Data is not fully uploaded due to privacy constraints, synthetic/sample data provided)
 ┣ 📂 src/                # Core implementation source code
 ┃ ┣ 📜 data_fusion.py    # Multi-source heterogeneous data preprocessing & PCA
 ┃ ┣ 📜 sem_extraction.py # Scripts for extracting SEM path coefficients
 ┃ ┣ 📜 sh_brb_engine.py  # The core SH-BRB dynamic inference engine
 ┃ ┗ 📜 baselines.py      # SOTA baseline models (XGBoost, CatBoost, RF, LightGBM, SVM)
 ┣ 📂 utils/              # Helper functions, evaluation metrics (F1, Precision, Recall)
 ┣ 📜 requirements.txt    # Python dependencies
 ┗ 📜 README.md           # This documentation file
