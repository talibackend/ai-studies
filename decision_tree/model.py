
import numpy as np
from matplotlib import pyplot
import sklearn
import seaborn as sns
import pandas as pd
import helper
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold, GridSearchCV


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

test_data_y = test_data["class"]
test_data_x = test_data.drop("class", axis=1)

predictions_2 = model_1.predict(test_data_x)

print(sklearn.metrics.accuracy_score(test_data_y, predictions_2))
print(sklearn.metrics.confusion_matrix(test_data_y, predictions_2))
print(sklearn.metrics.classification_report(test_data_y, predictions_2))

params = { "max_depth" : range(1, 10) }
n_folds = 5

model_2 = GridSearchCV(DecisionTreeClassifier(criterion="gini", random_state=100), params, cv=n_folds, scoring="accuracy", return_train_score=True)
model_2 = model_2.fit(train_data_x, train_data_y)

model_3 = GridSearchCV(DecisionTreeClassifier(criterion='gini', random_state=100), { "min_samples_leaf" : range(1, 40, 3) }, cv=n_folds, return_train_score=True)
model_3 = model_3.fit(train_data_x, train_data_y)

model_4 = GridSearchCV(DecisionTreeClassifier(criterion="gini", random_state=100), { "min_samples_split" : range(1, 40, 3) }, cv=n_folds, return_train_score=True)
model_4 = model_4.fit(train_data_x, train_data_y)

score_df = pd.DataFrame(model_4.cv_results_)

print(score_df.head(10))

# pyplot.figure(figsize=(8, 6))
# ax = sns.lineplot(score_df, x="param_min_samples_split", y="mean_train_score")
# sns.lineplot(score_df, x="param_min_samples_split", y="mean_test_score")
# ax.legend()
# ax.set_xlabel("Min Sample Split")
# ax.set_ylabel("Accuracy")
# pyplot.show()

# Max depth is 4 for best optimization
# Min samples leaf is 22 for best optimization 
# Min samples split is 15 for best optimization

# last_params = { "criterion" : ["entropy", "gini"], "max_depth" : range(1, 10), "min_samples_leaf" : range(1, 40, 3), "min_samples_split" : range(1, 40, 3) }

# model_5 = GridSearchCV(DecisionTreeClassifier(criterion="gini", random_state=100), last_params, cv=n_folds, return_train_score=True)
# model_5 = model_5.fit(train_data_x, train_data_y)

# print(model_5.best_score_)
# print(model_5.best_params_)
# print(model_5.best_index_)
# print(model_5.best_estimator_)

final_model = DecisionTreeClassifier(criterion="entropy", max_depth=2, min_samples_leaf=1, min_samples_split=4, random_state=100)
final_model = final_model.fit(train_data_x, train_data_y)

predictions = final_model.predict(test_data_x)

print(sklearn.metrics.classification_report(test_data_y, predictions))
print(sklearn.metrics.confusion_matrix(test_data_y, predictions))