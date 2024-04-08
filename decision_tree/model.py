
import numpy as np
from matplotlib import pyplot
import sklearn
import seaborn as sns
import pandas as pd
import helper
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


df = pd.read_csv("datasets/iris_csv.csv")

train_data, test_data = train_test_split(df, test_size=0.3, train_size=0.7, random_state=10)

train_data_y = train_data["class"]
train_data_x = train_data.drop("class", axis=1)

model_1 = DecisionTreeClassifier()
model_1 = model_1.fit(train_data_x, train_data_y)

predictions = model_1.predict(train_data_x)

print(sklearn.metrics.accuracy_score(train_data_y, predictions))
print(sklearn.metrics.confusion_matrix(train_data_y, predictions))
print(sklearn.metrics.classification_report(train_data_y, predictions))