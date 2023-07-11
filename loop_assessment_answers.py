import pandas as pd

data = pd.read_csv("loop_case_study.csv")

head = data.head()
shape = data.shape
info = data.info()
nulls = data.isnull()
nulls_sum = data.isnull().sum()
dtypes_dict = dict(data.dtypes)
objects = [category for category in dtypes_dict.keys() if dtypes_dict[category].char == "O"]
for category in objects:
    data[category] = data[category].astype('category')
    
stats = data.describe().T
