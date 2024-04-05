import numpy as np
from matplotlib import pyplot
import sklearn
import seaborn as sns
import pandas as pd
import helper

df = pd.read_csv("datasets/HR-Employee-Attrition.csv")

print(df.head(20))
print(df.info())
print(df.describe())

# pyplot.figure(figsize=(8, 6))
# ax = sns.countplot(df, x="Attrition")
# ax.legend()
# pyplot.show()

# No Attrition is largely higher than Attrition

helper.printAllStringColumns(df)

pyplot.figure(figsize=(8, 6))
ax = sns.countplot(df, x="Attrition")
ax.legend()

pyplot.figure(figsize=(8, 6))
ax = sns.countplot(df, x="Gender")
ax.legend()

pyplot.figure(figsize=(8, 6))
ax = sns.countplot(df, x="OverTime")
ax.legend()

pyplot.figure(figsize=(8, 6))
ax = sns.countplot(df, x="EducationField")
ax.legend()

pyplot.figure(figsize=(8, 6))
ax = sns.countplot(df, x="Department")
ax.legend()

pyplot.figure(figsize=(8, 6))
ax = sns.countplot(df, x="MaritalStatus")
ax.legend()

pyplot.figure(figsize=(8, 6))
ax = sns.countplot(df, x="BusinessTravel")
ax.legend()

pyplot.show()
