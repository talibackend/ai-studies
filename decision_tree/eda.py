
import numpy as np
from matplotlib import pyplot
import sklearn
import seaborn as sns
import pandas as pd
import helper


df = pd.read_csv("datasets/iris_csv.csv")

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
ax = sns.countplot(df, x="class")
ax.legend()
pyplot.show()

# Classes are evenly distributed