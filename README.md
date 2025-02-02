# Sentiment Analysis of Autonomous Vehicle Tweets: A Comparative Study

## Summary
This repository contains research and analysis on the comparative performance of various sentiment analysis models applied to multilabel and multiclass classifications of tweets (negative, neutral, and positive) related to autonomous vehicles (AVs). The study explores several sentiment analysis models including:
- Support Vector Machines (SVM)
- Long Short-Term Memory networks (LSTM)
- BERTweet (base)
- RoBERTa (base)
- RoBERTa (base-latest)

**Keywords**: `sentiment-analysis`, `deep-learning`, `pytorch`, `autonomous-vehicles`, `tweets-analysis`, `nlp`, `bert`, `roberta`, 
`transformers`, `svm`, `lstm`, `huggingface-transformers`, `machine-learning`, `multilabel-classification`, `multiclass-classification`, 
`neural-networks`, `optuna`, `bayesian-optimization`


## Method
The research process encompassed several key components:

<img src="https://raw.githubusercontent.com/titgsky/Sentiment-Analysis-of-Tweets-on-Autonomous-Vehicles-from-2012-to-2021/main/Model_building_work_flow_ensemble_model.png" alt="Model Building Workflow" width="700" align="center"/>
*Figure 1: Model Building Workflow for Ensemble Model*

- Data splitting for robust model evaluation
- Bayesian optimization for hyperparameter tuning
- Ensemble learning approach with 5-fold stratified cross-validation
- Comprehensive evaluation using multiple classification metrics

## Objectives
Our study aimed to:
1. Compare the effectiveness of different models in multi-label and multi-class classification tasks
2. Evaluate model performance in classifying AV-related tweet sentiments

## Framework and Dependencies

### Core Framework
- Python version: 3.9.21
- PyTorch version: 2.5.1 with GPU support (CUDA version: 12.1)

### Required Libraries
```
Pandas version: 2.2.3
NumPy version: 1.26.4
Scikit-learn version: 1.6.0
Matplotlib version: 3.9.4
Seaborn version: 0.13.2
Optuna version: 4.1.0
Torchinfo version: 1.8.0
JupyterLab version: 4.3.4
```

## Repository Organization
The project is structured into the following directories:

### Data Sets
Contains tweet IDs for project replication:
- Complete dataset (6 million tweets) available on Zenodo (https://zenodo.org/records/14636994)
- Filtered dataset (3 million tweets) available on Zenodo (https://zenodo.org/records/14636994)
- Manually labeled dataset (1,198 randomly selected tweet IDs), available here, used for:
  - Training
  - Validation
  - Testing

How to cite:

Sauvayre, R., Fernandes Novo, M., Dehondt, M., Gable, J. S. M., Aalah, A., & Chauvière, C. (2025). Sentiment Analysis of Tweets on Autonomous Vehicles from 2012 to 2021 (V2.0.0). Zenodo. https://doi.org/10.5281/zenodo.14636994

### Emotional and Valence Dictionary
- Resource used for textual data filtering
- Applied in preprocessing phase

### Model Implementation
Each model type includes implementations for both multilabel and multiclass classification.

The Jupyter notebooks using PyTorch framework are available in the folder 'Jupyternotebooks'

## Key Findings
Our analysis revealed significant insights:

### Optimal Classification Approach
Multilabel classification emerged as the superior choice, with BERTweet (base) achieving:
- Accuracy: 78.21%
- Precision: 75.94%
- Recall: 68.18%
- F1-score: 71.66%

### Alternative Model Performance
RoBERTa latest demonstrated strong potential as an alternative solution, particularly in:
- Distinguishing between sentiment classes
- Maintaining consistent performance across different sentiment categories
