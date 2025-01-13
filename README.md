# Sentiment Analysis of Autonomous Vehicle Tweets: A Comparative Study

## Summary
This repository contains research and analysis on the comparative performance of various sentiment analysis models applied to multilabel and multiclass classifications of tweets (negative, neutral, and positive) related to autonomous vehicles (AVs). The study explores several sentiment analysis models including:
- Support Vector Machines (SVM)
- Long Short-Term Memory networks (LSTM)
- BERTweet (base)
- RoBERTa (base)
- RoBERTa (base-latest)

## Method
The research process encompassed several key components:

<img src="https://github.com/titgsky/Sentiment-Analysis-of-Tweets-on-Autonomous-Vehicles-from-2012-to-2021/blob/main/Model_building_work_flow_ensemble_model.png" alt="Model Building Workflow" width="600"/>
*Figure 1: Model Building Workflow for Ensemble Model*

- Data splitting for robust model evaluation
- Bayesian optimization for hyperparameter tuning
- Ensemble learning approach with 5-fold stratified cross-validation
- Comprehensive evaluation using multiple classification metrics

## Objectives
Our study aimed to:
1. Compare the effectiveness of different models in multi-label and multi-class classification tasks
2. Evaluate model performance in classifying AV-related tweet sentiments

## Repository Organization
The project is structured into the following directories:

### Framework
PyTorch

### Data Sets
Contains tweet IDs for project replication:
- Complete dataset (6 million tweets)
- Filtered dataset (3 million tweets)
- Manually labeled dataset (1,198 randomly selected tweet IDs) used for:
  - Training
  - Validation
  - Testing

### Emotional and Valence Dictionary
- Resource used for textual data filtering
- Applied in preprocessing phase

### Model Directories
Each model type includes implementations for both:
- Multilabel classification
- Multiclass classification

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
