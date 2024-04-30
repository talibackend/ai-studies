
import numpy as np
from matplotlib import pyplot
import sklearn
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn import datasets

ds = datasets.make_moons(noise=0.1)

print(ds)

moons_df = pd.DataFrame()
moons_df["feature_1"] = ds[0][:, 0]
moons_df["feature_2"] = ds[0][:, 1]
moons_df["target"] = ds[1]

print(moons_df)


y = moons_df["target"]
x = moons_df.drop(columns="target")

# pyplot.figure(figsize=(8, 6))
# ax = sns.scatterplot(moons_df, x="feature_1", y="feature_2", hue="target")
# ax.legend()
# pyplot.show()

# model = Pipeline([
#     ("Polynomial Features", PolynomialFeatures(degree=3)),
#     ("Scaling", StandardScaler()),
#     ("Model Creation", LinearSVC(C=10, loss="hinge"))
# ])

# model = model.fit(x, y)

test_df = pd.DataFrame()
test_df["feature_1"] = [0.5, 0.6, 1.5, 0.5]
test_df["feature_2"] = [-0.6, 1, 0.75, 0.5]

# print(model.predict(test_df))

# Kernel Trick
model_2 = Pipeline([
    ("Scaling", StandardScaler()),
   ("Model Creation", SVC(kernel="poly", degree=3, coef0=1, C=10))
])
model_2 = model_2.fit(x, y)

print(model_2.predict(test_df))