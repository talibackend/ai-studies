
import numpy as np
from matplotlib import pyplot
import sklearn
import seaborn as sns
import pandas as pd
import helper


df = pd.read_csv("datasets/credit-card-default.csv")

print(df.head(20))
print(df.info())
print(df.describe())


helper.printAllStringColumns(df)

# pyplot.figure(figsize=(8, 6))
# ax = sns.countplot(df, x="defaulted")
# ax.legend()
# pyplot.show()

# Classes are not evenly distributed, 0 has a much higher percentage than 1