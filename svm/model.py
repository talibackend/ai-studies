
import numpy as np
from matplotlib import pyplot
import sklearn
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn import datasets

iris = datasets.load_iris()
iris_ds = pd.DataFrame()
iris_ds["petallength"] = iris["data"][:, 2]
iris_ds["petalwidth"] = iris["data"][:, 3]
iris_ds["target"] = iris["target"]
iris_ds["target"] = iris_ds["target"].apply(lambda x: 1 if x == 2 else 0)

iris_ds = iris_ds.astype(np.float64)

print(iris_ds["petallength"])
print(iris_ds["petalwidth"])
print(iris_ds["target"])
print(iris_ds.info())

# pyplot.figure(figsize=(8, 6))
# ax = sns.scatterplot(iris_ds, x="petallength", y="petalwidth", hue="target")
# ax.legend()
# pyplot.show()

# scaler = StandardScaler()
# iris_ds[["petallength", "petalwidth"]] = scaler.fit_transform(iris_ds[["petallength", "petalwidth"]])

# print(iris_ds)

y = iris_ds["target"]
x = iris_ds.drop(columns="target")

model = Pipeline([("Scaling", StandardScaler()), ("Linear SVC Model", LinearSVC(C=1, loss="hinge"))])
model = model.fit(x, y)

# print(x)
# print(y)

# model = LinearSVC(C=1, loss="hinge")
# model = model.fit(x, y)

# print(model.coef_)
# print(model.intercept_)

test_ds = pd.DataFrame()
test_ds["petallength"] = [5.5, 4.4, 2.1]
test_ds["petalwidth"] = [1.7, 1.2, 0.4]

print(test_ds)
print(model.predict(test_ds))