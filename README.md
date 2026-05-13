# Diabetes Prediction using Machine Learning & Deep Learning

## Overview
This project predicts diabetes progression (regression) and diabetes onset (binary classification) using diagnostic measures. It compares **10+ machine learning algorithms** and **deep learning models** to find the best-performing approach.

![Predict Diabetic](https://github.com/ParisaMohammadi9094/Diabetic_Prediction_using_Machine_Learning/assets/18152407/e560ddee-7b03-4425-938e-a2b5718a6b4f)

## Key Features
- **Dual Prediction Tasks**:
  - Regression: Predict diabetes progression score (quantitative)
  - Classification: Predict diabetes onset (binary: high/low progression)
- **10+ Machine Learning Algorithms**:
  - Linear Regression, Ridge, Lasso, ElasticNet
  - K-Nearest Neighbors (KNN)
  - Support Vector Machines (SVR/SVC)
  - Decision Trees, Random Forest, Gradient Boosting
  - XGBoost
- **Deep Learning Models**:
  - Artificial Neural Network (ANN) with dropout layers and early stopping
- **Comprehensive Evaluation**:
  - Regression: MSE, RMSE, MAE, R², Cross-validation scores
  - Classification: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Model Comparison Table** – Sorted by performance (R² for regression, Accuracy for classification)
- **Caching System** – Speeds up re-training during hyperparameter tuning
- **Visualizations**:
  - Actual vs Predicted plots
  - Model comparison bar charts
  - Confusion matrices (for classification)
  - ROC curves

## Dataset
The dataset is sourced from `sklearn.datasets.load_diabetes`:
- **Features**: 10 baseline variables (age, sex, BMI, blood pressure, and 6 blood serum measurements)
- **Target (Regression)**: Quantitative measure of disease progression one year after baseline
- **Target (Classification)**: Binary label (1 = progression above median, 0 = below median)
- **Samples**: 442 patients


## Requirements
- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- xgboost
- tensorflow (for deep learning)
- joblib

