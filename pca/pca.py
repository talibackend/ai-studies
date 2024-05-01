
import numpy as np
from matplotlib import pyplot
import sklearn
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

housing_df = pd.read_csv('datasets/newhousing.csv')
y = housing_df["price"]
housing_df.drop(columns="price", inplace=True)

print(housing_df)
print(housing_df.info())

dup_df = housing_df.copy()
scaler = StandardScaler()
dup_df[dup_df.columns] = scaler.fit_transform(dup_df[dup_df.columns])

print(dup_df)

# pca = PCA(random_state=100)
# pca = pca.fit(dup_df)

# print(pca.components_)
# print(pca.explained_variance_ratio_)

# x_values = []

# var_cumu = np.cumsum(pca.explained_variance_ratio_)
# print(var_cumu)

# for i in range(1, len(pca.explained_variance_ratio_) + 1):
#     x_values.append(i)

# pyplot.figure(figsize=(8, 6))
# ax = sns.barplot(x=x_values, y=pca.explained_variance_ratio_)
# ax.legend()

# pyplot.figure(figsize=(8, 6))
# ax = sns.lineplot(x=x_values, y=var_cumu)
# ax.legend()

# pyplot.show()

pc2 = PCA(n_components=8, random_state=100)
new_data = pc2.fit_transform(dup_df)

print(new_data)

new_data = pd.DataFrame(new_data, columns=["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8"])

print(new_data)