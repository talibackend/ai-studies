import sklearn.metrics
import numpy as np
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import pandas as pd
import helper

df = pd.read_csv("datasets/HR-Employee-Attrition.csv")
print(df.shape)

def transformYesNo(field):
    if field == "Yes":
        return 1
    else:
        return 0



df["Attrition"] = df["Attrition"].apply(lambda x: transformYesNo(x))
df["OverTime"] = df["OverTime"].apply(lambda x: transformYesNo(x))

df = df.drop(["Over18", "EmployeeCount", "StandardHours"], axis=1)

df = helper.normalizeCategoryFields(df, ["BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus"])

print(df.head(20))
print(df.info())

highlyCorrelatedColumns = helper.printCorrelations(df, 0.8)

print(highlyCorrelatedColumns)

# Drop highly correlated fields

df = df.drop(highlyCorrelatedColumns, axis=1)

print(df.shape)

scaler = StandardScaler()
columns_without_attrition = df.columns
string_to_remove = "Attrition"
filtered_columns = [column for column in columns_without_attrition if string_to_remove not in column]

df[filtered_columns] = scaler.fit_transform(df[filtered_columns])

train_data, test_data = train_test_split(df, test_size=0.3, train_size=0.7, random_state=10)

train_data = train_data.astype(float)
train_data_y = train_data["Attrition"]
train_data_x = train_data.drop("Attrition", axis=1)

model = GaussianNB()
model = model.fit(train_data_x, train_data_y)

prediction = model.predict(train_data_x)

cm = sklearn.metrics.confusion_matrix(train_data_y, prediction)

print(cm)
print(sklearn.metrics.accuracy_score(train_data_y, prediction))

prediction_prob = model.predict_proba(train_data_x)
prediction_prob = prediction_prob[:, 1]

prediction_df = pd.DataFrame({ "prob" : prediction_prob, "actual" : train_data_y })

for i in range(10):
    new_i = i / 10
    prediction_df[new_i] = prediction_df["prob"].apply(lambda x: 1 if x > new_i else 0)


print(prediction_df)

threshold_predictions = []

# for i in range(10):
#     each_prediction = []
#     new_i = i / 10
#     each_prediction.append(new_i)
#     each_cm = sklearn.metrics.confusion_matrix(train_data_y, prediction_df[new_i])

#     accuracy = (each_cm[0, 0] + each_cm[1, 1])/(each_cm[1, 0] + each_cm[1, 1] + each_cm[0, 1] + each_cm[0, 0])
#     sensitivity = each_cm[1, 1]/(each_cm[1, 1] + each_cm[1, 0])
#     specificity = each_cm[0, 0]/(each_cm[0, 0] + each_cm[0, 1])

#     each_prediction.append(accuracy)
#     each_prediction.append(sensitivity)
#     each_prediction.append(specificity)

#     threshold_predictions.append(each_prediction)


# stats_df = pd.DataFrame(threshold_predictions, columns=["threshold", "accuracy", "sensitivity", "specificity"])

# print(stats_df)

# pyplot.figure(figsize=(8, 6))
# ax = sns.lineplot(stats_df, x="threshold", y="accuracy")
# sns.lineplot(stats_df, x="threshold", y="sensitivity")
# sns.lineplot(stats_df, x="threshold", y="specificity")
# ax.legend(["Accuracy", "Sensitivity", "Specificity"])
# pyplot.show()

# Threshold gotten from the above analysis is 0.456

# threshold_predictions = []
# for i in range(10):
#     each_prediction = []
#     new_i = i / 10
#     each_prediction.append(new_i)
#     each_prediction.append(sklearn.metrics.recall_score(train_data_y, prediction_df[new_i]))
#     each_prediction.append(sklearn.metrics.precision_score(train_data_y, prediction_df[new_i]))

#     threshold_predictions.append(each_prediction)

# stats_df = pd.DataFrame(threshold_predictions, columns=["threshold", "recall", "precision"])

print(stats_df)

pyplot.figure(figsize=(8, 6))
ax = sns.lineplot(stats_df, x="threshold", y="recall")
sns.lineplot(stats_df, x="threshold", y="precision")
ax.legend(["Recall", "Precision"])
pyplot.show()

# We will avoid tuning for this algorithm and try to use another algorithm to yield better results

test_data = test_data.astype(float)
test_data_y = test_data["Attrition"]
test_data_x = test_data.drop("Attrition", axis=1)

test_prediction = model.predict(test_data_x)

test_cm = sklearn.metrics.confusion_matrix(test_data_y, test_prediction)
print(test_cm)
print(sklearn.metrics.accuracy_score(test_data_y, test_prediction))