import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns

data = pd.read_csv('datasets/CarPrice_Assignment.csv')

print(data.head())
print(data.shape)
print(data.info())
string_columns = data.columns[data.dtypes == "object"]

for column in string_columns:
    print("Unique values for " + column)
    print(data[column].unique())
    print("=======================")

print(data.describe())

data["CompanyName"] = data["CarName"].apply(lambda x: x.split(' ')[0])
data.drop(columns=['car_ID', 'CarName'], inplace=True)

print(data.info())

string_columns = data.columns[data.dtypes == "object"]
for column in string_columns:
    print("Unique values for " + column)
    print(data[column].unique())
    print("=======================")
