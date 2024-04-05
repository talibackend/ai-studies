import sklearn.metrics
import numpy as np
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import helper
import sklearn

df = pd.read_csv("datasets/HR-Employee-Attrition.csv")
print(df.shape)

def transformYesNo(field):
    if field == "Yes":
        return 1
    else:
        return 0



df["Attrition"] = df["Attrition"].apply(lambda x: transformYesNo(x))
df["OverTime"] = df["OverTime"].apply(lambda x: transformYesNo(x))

df = df.drop(["Over18", "EmployeeCount", "StandardHours"], axis=1)

df = helper.normalizeCategoryFields(df, ["BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus"])

print(df.head(20))
print(df.info())

highlyCorrelatedColumns = helper.printCorrelations(df, 0.8)

print(highlyCorrelatedColumns)

# Drop highly correlated fields

df = df.drop(highlyCorrelatedColumns, axis=1)

print(df.shape)