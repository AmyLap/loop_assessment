########################################################################################
# QUESTION ONE
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer

data = pd.read_csv("loop_case_study.csv")

head = data.head()
"""
   EmployeeID HeartDisease    BMI Smoking AlcoholDrinking Stroke  ...  Diabetic  PhysicalActivity SleepTime Asthma KidneyDisease SkinCancer
0       13423           No  16.60     Yes              No     No  ...       Yes               Yes         5    Yes            No        Yes
1       33574           No  20.34      No              No    Yes  ...        No               Yes         7     No            No         No
2       65906           No  26.58     Yes              No     No  ...       Yes               Yes         8    Yes            No         No
3       92780           No  24.21      No              No     No  ...        No                No         6     No            No        Yes
4       51415           No  23.71      No              No     No  ...        No               Yes         8     No            No         No

"""
shape = data.shape  # (202121, 17)
info = data.info()
nulls = data.isnull()  # No nulls

dtypes_dict = dict(data.dtypes)
categorical_columns = [category for category in dtypes_dict.keys() if dtypes_dict[category].char == "O"]
numerical_categories = [category for category in dtypes_dict.keys() if dtypes_dict[category].name in ['float64', 'int64'] and category != "EmployeeID"]
stats = data.describe().T
"""
                   count       mean       std    min    25%    50%    75%    max
BMI             202121.0  28.275822  6.344578  12.02  23.91  27.28  31.32  94.85
PhysicalHealth  202121.0   3.326389  7.898421   0.00   0.00   0.00   2.00  30.00
MentalHealth    202121.0   3.868549  7.899459   0.00   0.00   0.00   3.00  30.00
SleepTime       202121.0   7.110083  1.437298   1.00   6.00   7.00   8.00  24.00
"""
heart_disease_summary = data.groupby("HeartDisease").mean(numeric_only=True).round(2)
"""
                BMI  PhysicalHealth  MentalHealth  SleepTime   No  Yes
HeartDisease----------------------------------------------------------
No            28.18            2.92           3.8       7.10  1.0  0.0
Yes           29.34            7.73           4.6       7.17  0.0  1.0
"""

#####################################################################################

# 1.
# Unsupervised as we have a complete set of data

# 2.
# HeartDisease
y = data["HeartDisease"]
x = data
x_columns = x.columns

# 3.
# Machine Learning Models can not work on categorical variables in the form of strings,
# thus, we need to change it into numerical form.
# Hotencoding
y_encoded = LabelEncoder().fit_transform(y)
for category in categorical_columns:
    x[category] = LabelEncoder().fit_transform(x[category])

data["EmployeeID"] = data["EmployeeID"].astype('category')
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
x_without_y = x.drop("HeartDisease", axis=1)
x_without_y_columns = x_without_y.columns

# 4
counts = y.value_counts()
# counts.plot(kind='bar')
# plt.show()

"""
    No     184901
    Yes     17220
    Name: HeartDisease, dtype: int64
"""
# A large disparity between the two classes ('those with heart disease' and
# 'those without heart disease') can be seen in the above data
# and in 'Figure_1.png'.

# This indicates that there is a class imbalance as the majority
# class ("No") dominates the dataset, and the minority class ("Yes")
# is under-represented.

# Typically, I would try multiple techniques and compare their
# performance to determine the most effective approach for the
# given problem. However, due to time constraints, I have chosen
# to apply the SMOTE oversampling technique from the imblearn.over_sampling
# module to handle the imbalance. This will help address the class imbalance issue
# by generating synthetic samples for the minority class .

# Apply SMOTE oversampling
smote = SMOTE(random_state=55)
x_balanced, y_balanced = smote.fit_resample(x_without_y, y_encoded)
x_y_balanced, y_balanced = smote.fit_resample(x, y_encoded)

# 5. & 8.
# To balance the data I remove "EmployeeID" as it it is
# categorical and should not be convert to a float.
categorical_significance = {}
for category in categorical_columns:

    if category != "HeartDisease":
        contingency_table = pd.crosstab(x_balanced[category], y_balanced)
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        categorical_significance[category] = {
            # "contingency_table": contingency_table,
            "chi2": round(chi2, 2),
            "p_value": round(p_value, 4)
        }

for category in numerical_categories:
    # Separate the numerical variable based on the HeartDisease class
    balanced_x_HD = x_y_balanced[x_y_balanced["HeartDisease"] == 1][category]
    balanced_x_no_HD = x_y_balanced[x_y_balanced["HeartDisease"] == 0][category]

    # Perform the independent t-test
    t_statistic, p_value = ttest_ind(balanced_x_HD, balanced_x_no_HD)
    categorical_significance[category] = {
        "t_statistic": round(t_statistic),
        "p_value": round(p_value, 2)
    }

# sns.pairplot(
#     x_y_balanced[["BMI", "PhysicalHealth", "MentalHealth", "HeartDisease"]],
#     hue = "HeartDisease",
#     height = 3,
#     palette = "Set1")

"""
{'Smoking': {'chi2': 479.32, 'p_value': 0.0
    }, 'AlcoholDrinking': {'chi2': 10909.76, 'p_value': 0.0
    }, 'Stroke': {'chi2': 375.83, 'p_value': 0.0
    }, 'DiffWalking': {'chi2': 1078.57, 'p_value': 0.0
    }, 'Sex': {'chi2': 2634.49, 'p_value': 0.0
    }, 'AgeCategory': {'chi2': 77613.12, 'p_value': 0.0
    }, 'Diabetic': {'chi2': 34447.51, 'p_value': 0.0
    }, 'PhysicalActivity': {'chi2': 41849.39, 'p_value': 0.0
    }, 'Asthma': {'chi2': 8832.47, 'p_value': 0.0
    }, 'KidneyDisease': {'chi2': 32.71, 'p_value': 0.0
    }, 'SkinCancer': {'chi2': 2000.85, 'p_value': 0.0
    }, 'BMI': {'t_statistic': 48, 'p_value': 0.0
    }, 'PhysicalHealth': {'t_statistic': 117, 'p_value': 0.0
    }, 'MentalHealth': {'t_statistic': -14, 'p_value': 0.0
    }, 'SleepTime': {'t_statistic': -57, 'p_value': 0.0}
}
"""
# These results and 'Figure_2.png' suggest that there is a significant relationship between heart disease
# and all the given categories. Thus, the only column I have chosen to remove is "EmployeeID"

# 6.
# I have chosen to use standardization (z-score scaling) as a feature scaling method
# to ensure that the features are centred around zero and have a similar scale,
# It also prevent sensitives to outliers.
scaler = StandardScaler()
scaled_features_1 = scaler.fit_transform(x_balanced)

# 7
# Check for nan values
scaled_nans = np.isnan(scaled_features_1)

# 9.
X_train, X_test, y_train, y_test = train_test_split(
    scaled_features_1,
    y_balanced,
    test_size=0.2,
    stratify=y_balanced,
    random_state=55)

# # Handle missing values in the training data (X_train) before training the classifier
# # Convert the training data to a DataFrame
# X_train = pd.DataFrame(X_train, columns=x_balanced.columns)
# y_train = pd.Series(y_train)
# train_data = X_train
# train_data["HeartDisease"] = y_train
# train_data = train_data.dropna()
# X_train = train_data.drop("HeartDisease", axis=1)
# y_train = train_data["HeartDisease"]

# X_test = pd.DataFrame(X_test, columns=x_balanced.columns)
# y_test = pd.Series(y_test)
# test_data = X_test
# test_data["HeartDisease"] = y_test
# test_data = test_data.dropna()
# X_test = test_data.drop("HeartDisease", axis=1)
# y_test = test_data["HeartDisease"]


classify = RandomForestClassifier(n_estimators=200, random_state=55)
# classify.fit(X_train, y_train)
# predict_y = classify.predict(X_test)

# # 10.
# accuracy = accuracy_score(y_test, predict_y) # 0.9037358061157785
# precision = precision_score(y_test, predict_y) # 0.6420966420966421
# recall = recall_score(y_test, predict_y) # 0.18135554013416608
# f1 = f1_score(y_test, predict_y) # 0.2828282828282828

# 11.
# Dropping the nan values from the train and test sets may have resulted in the low
# precision, recall and f1 scores. I have chosen to use imputatation to handle the
# missing fields rather than dropping the fields.
# Handle missing values
imputer = SimpleImputer(strategy="mean")
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=x_without_y_columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=x_without_y_columns)

# Build the classifier
classify = RandomForestClassifier(n_estimators=150, random_state=55)
classify.fit(X_train_imputed, y_train)

# Make predictions on the test set
predict_y = classify.predict(X_test_imputed)

# Evaluate the classifier
new_accuracy = accuracy_score(y_test, predict_y)  # 0.9466610781357743
new_precision = precision_score(y_test, predict_y)  # 0.9883514664143803
new_recall = recall_score(y_test, predict_y)  # 0.9039777182877694
new_f1 = f1_score(y_test, predict_y)  # 0.9442835957912578
#
# 12.
# The updated model with the new scores shows improved performance.
# The higher accuracy (0.947) indicates that the model is correctly predicting most of the samples.
# The precision (0.988) indicates that the model is highly likely to be correct  when it predicts a positive case of heart disease.
# The recall (0.904) indicates that the model is able to identify a large portion of the actual positive cases from the dataset.
# The F1-score (0.944), a harmonic mean of precision and recall, provides a measure of the model's performance.
# Based on these results, this model will be sufficient for its intended purpose.

# 13.
# Overfitting occurs when a machine learning model provides accurate predictions for
# training data but in-accurate predictions for new data. It is a common issue in
# machine learning. This issue occurs when a model over-learns the training date,
# and thus resulting in in poor performance on new data.

# Some of the approaches one can take to avoid overfitting, is to ensure that there is a large,
# diverse training dataset, to focus on feature selection, and to stop the learning early before
# the noise in the data is learnt by the model

# 14.
# Statstical analysis, and exploritory data evaluation can also help identify good candidates
# for the health-allowance. They other option is to give the allowance to those that lead a
# healthier lifestyle as an incentive.

###########################################################################################
# QUESTION TWO

# 1.
"""
SELECT
  client_id,
  branch_name,
  MAX(order_date) AS last_order_date,
  DATEDIFF(CURRENT_DATE(), MAX(order_date)) AS n_days_since_last_order,
  CASE
    WHEN DATEDIFF(CURRENT_DATE(), MAX(order_date)) BETWEEN 0 AND 13 THEN '1-2 weeks'
    WHEN DATEDIFF(CURRENT_DATE(), MAX(order_date)) BETWEEN 14 AND 20 THEN '2-3 weeks'
    WHEN DATEDIFF(CURRENT_DATE(), MAX(order_date)) BETWEEN 21 AND 27 THEN '3-4 weeks'
    ELSE 'More than 4 weeks'
  END AS days_since_last_order_category
FROM
  orders
GROUP BY
  client_id,
  branch_name
ORDER BY
  n_days_since_last_order DESC;


CREATE TABLE orders (
  client_id VARCHAR(10),
  branch_name VARCHAR(10),
  order_date TIMESTAMP
);


INSERT INTO orders (client_id, branch_name, order_date)
VALUES
('123XPH', 'CPT', '2023-07-01 15:23:15'),
('124XPH', 'JHB', '2023-06-30 15:23:15'),
('123XPH', 'CPT', '2023-06-03 15:23:15'),
('123XPH', 'CPT', '2023-05-20 15:23:15'),
('124XPH', 'JHB', '2023-07-02 15:23:15'),
('124XPH', 'JHB', '2023-07-08 15:23:15'),
('124XPH', 'JHB', '2022-12-15 15:23:15'),
('125XPH', 'JHB', '2022-12-01 15:23:15')
;

#RESEULTS
"client_id"	"branch_name"  	"last_order_date"	 "n_days_since_last_order"	"days_since_last_order_category"
"125XPH"	    "JHB"	   "2022-12-01T15:23:15Z"	        224	             "More than 4 weeks"
"123XPH"	    "CPT"      "2023-07-01T15:23:15Z"	        12	                "1-2 weeks"
"124XPH"	    "JHB"	   "2023-07-08T15:23:15Z"          	5	                "1-2 weeks"

"""

# 2.
"""

"""
