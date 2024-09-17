# Heart Disease Prediction & Order Prediction Project

## Overview
This project consists of two parts:

1. **Heart Disease Prediction using Machine Learning**: 
   - A detailed analysis of a dataset to predict heart disease using machine learning algorithms, including feature preprocessing, model training, and evaluation.
   
2. **Order Prediction with SQL**:
   - SQL queries to analyze order data from a hypothetical database, including predictions of orders and comparison with actual orders.

## Part 1: Heart Disease Prediction

### Dataset
- The dataset consists of 202,121 records of health-related information.
- Key columns include:
  - **EmployeeID**: Unique ID for employees.
  - **HeartDisease**: Target variable, indicating if a person has heart disease.
  - **BMI, Smoking, AlcoholDrinking, Stroke, PhysicalActivity, SleepTime, etc.**: Various health-related attributes.

### Libraries Used
- `pandas`, `numpy`: For data manipulation and analysis.
- `seaborn`, `matplotlib`: For data visualization.
- `imblearn.over_sampling.SMOTE`: To handle imbalanced data using oversampling.
- `sklearn`: For preprocessing, model training, and evaluation.
- `scipy`: For statistical tests (chi-squared test, t-test).
- `sqlite3`: To create a database connection and execute SQL queries.

### Steps:

1. **Data Preprocessing**:
   - Missing values handled with `SimpleImputer`.
   - Categorical variables encoded using `LabelEncoder`.
   - Imbalance in target classes handled using `SMOTE`.
   
2. **Feature Scaling**:
   - Features were scaled using `StandardScaler` for normalization.

3. **Model Training**:
   - Random Forest Classifier was used to predict heart disease.
   - Data split into training and testing sets (80% training, 20% testing).
   - Imputation applied to handle missing data.

4. **Model Evaluation**:
   - **Accuracy**: 0.947
   - **Precision**: 0.988
   - **Recall**: 0.904
   - **F1 Score**: 0.944

5. **Chi-Squared & T-Test Results**:
   - A significant relationship was found between heart disease and features like `Smoking`, `AlcoholDrinking`, `BMI`, etc.

### Conclusion:
- The trained model demonstrated high performance, effectively predicting heart disease with strong precision and recall.
- The use of SMOTE to balance classes and imputation for missing data improved overall performance.

---
## Part 2: SQL Queries for Order Prediction

### SQL Queries Overview:
In this part, we work with SQL queries to analyze and predict orders. The queries are designed to extract insights from the order data by grouping clients based on the time since their last order and comparing predicted orders with actual orders.

