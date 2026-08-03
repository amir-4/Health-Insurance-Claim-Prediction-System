# 💳 Health Insurance Claim Prediction

<p align="center">

  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas" />
  <img src="https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge&logo=numpy" />
  <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn" />
  <img src="https://img.shields.io/badge/XGBoost-Gradient%20Boosting-189AB4?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Plotly-Visualization-3F4F75?style=for-the-badge&logo=plotly" />
  <img src="https://img.shields.io/badge/Streamlit-Deployment-FF4B4B?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/Joblib-Model%20Serialization-2E8B57?style=for-the-badge" />

</p>

<p align="center">

### 🚀 An End-to-End Supervised Machine Learning Project for Predicting Health Insurance Claim Costs

</p>

---

## 📌 Project Overview

**Health Insurance Claim Prediction** is an end-to-end supervised machine learning project developed to estimate health insurance claim costs based on customer demographic, health, lifestyle, and geographic characteristics.

The project covers the complete machine learning workflow — from **data exploration and feature engineering to model development, optimization, evaluation, error analysis, and deployment**.

The final solution is deployed as an interactive **Streamlit web application**, allowing users to enter customer information and receive an estimated insurance claim.

---

## 🎯 Project Objectives

The main objectives of this project are to:

- Understand the factors associated with insurance claim costs.
- Perform exploratory data analysis to identify meaningful patterns.
- Prepare numerical and categorical features for machine learning.
- Engineer additional features that may improve predictive performance.
- Build and compare multiple regression models.
- Apply cross-validation to evaluate model generalization.
- Optimize the best-performing model using hyperparameter tuning.
- Analyze model errors and learning behavior.
- Save the trained model and preprocessing pipeline.
- Deploy the final solution through an interactive Streamlit application.

---

# 🔄 Machine Learning Workflow

```text
                ┌─────────────────────┐
                │    Data Import      │
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │   Data Cleaning     │
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │        EDA          │
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │ Feature Engineering │
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │ Train / Test Split  │
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │   Preprocessing     │
                │ ColumnTransformer   │
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │ Model Development   │
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │ Cross-Validation     │
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │ Hyperparameter      │
                │ Optimization        │
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │ Model Evaluation    │
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │ Error Analysis      │
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │ Model Serialization │
                └──────────┬──────────┘
                           ↓
                ┌─────────────────────┐
                │ Streamlit Deployment│
                └─────────────────────┘
