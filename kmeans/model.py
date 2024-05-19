
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
from matplotlib import pyplot


mall_df = pd.read_csv('datasets/Mall_Customers.csv')

# mall_df["Is_male"] = mall_df["Gender"].apply(lambda x: True if x == "Male" else False)
# mall_df["Is_female"] = mall_df["Gender"].apply(lambda x: True if x == "Female" else False)

mall_df = mall_df.drop(axis=1, columns=["Gender", "CustomerID"])

print(mall_df)
print(mall_df.describe())
print(mall_df.info())

# pyplot.figure(figsize=(8, 6))
# ax = sns.pairplot(mall_df, hue="Gender")
# pyplot.show()

# km_inertias = []
# km_scores = []

# for i in range(3, 10):
#     model = KMeans(n_clusters=i).fit(mall_df)
#     km_inertias.append(model.inertia_)
#     km_scores.append(silhouette_score(mall_df, model.labels_))
    # print(f"K = {i}, Inertia = {model.inertia_}, Score = {silhouette_score(mall_df, model.labels_)}")

# pyplot.figure(figsize=(8, 6))
# ax = sns.lineplot(y=km_inertias, x=range(3, 10))
# ax.legend()
# pyplot.show()

# ax = sns.lineplot(y=km_scores, x=range(3, 10))
# ax.legend()
# pyplot.show()

# Optimal K is 6

model = KMeans(n_clusters=6).fit(mall_df)
mall_df["label"] = model.labels_

print(mall_df["label"].unique())
print(mall_df["label"].value_counts())

pyplot.figure(figsize=(8, 6))

pyplot.show()