
import numpy as np
from matplotlib import pyplot
import sklearn
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("datasets/credit-card-default.csv").drop("ID", axis=1)

train_data, test_data = train_test_split(df, test_size=0.3, train_size=0.7, random_state=100)

train_data_y = train_data["defaulted"]
train_data_x = train_data.drop("defaulted", axis=1)

test_data_y = test_data["defaulted"]
test_data_x = test_data.drop("defaulted", axis=1)


# model_1 = RandomForestClassifier()
# model_1 = model_1.fit(train_data_x, train_data_y)

# predictions = model_1.predict(test_data_x)

# print(sklearn.metrics.classification_report(test_data_y, predictions))
# print(sklearn.metrics.confusion_matrix(test_data_y, predictions))

cv_parameters = { 
    "max_depth" : range(2, 10, 1), 
    # "n_estimators" : range(10, 100, 10), 
    "min_samples_leaf" : range(100, 400, 100),
    "min_samples_split" : range(100, 500, 100), 
    "criterion" : ["gini", "entropy"],
    "max_features" : range(2, 10, 1)
}

cv = GridSearchCV(RandomForestClassifier(), cv_parameters, n_jobs=-1, cv=5, return_train_score=True, verbose=True)
cv = cv.fit(train_data_x, train_data_y)

print(cv.best_estimator_)
print(cv.best_score_)

# Can't continue because of minimal resources
