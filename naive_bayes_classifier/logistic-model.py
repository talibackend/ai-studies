import sklearn.metrics
import numpy as np
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import helper
import sklearn

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

train_data, test_data = train_test_split(df, test_size=0.3, train_size=0.7, random_state=100)

train_data = train_data.astype(float)
train_data_y = train_data["Attrition"]
train_data_x = train_data.drop("Attrition", axis=1)

print(train_data_x)
print(train_data_y)

# model_1 = helper.buildModel(train_data_x, train_data_y)
# print(model_1)

qualified_columns = ["Age", "DistanceFromHome", "EmployeeNumber", 
                     "EnvironmentSatisfaction", "JobInvolvement", "JobSatisfaction", 
                     "NumCompaniesWorked", "OverTime", "RelationshipSatisfaction", 
                     "TotalWorkingYears", "WorkLifeBalance", "YearsAtCompany", 
                     "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager", 
                     "JobRole_Manufacturing Director", "JobRole_Healthcare Representative", 
                     "JobRole_Manager", "JobRole_Sales Representative", "JobRole_Research Director"]

train_data_x = train_data_x[qualified_columns]

# model_2 = helper.buildModel(train_data_x, train_data_y)
# print(model_2)

model_3 = LogisticRegression()
model_3 = model_3.fit(train_data_x, train_data_y)

prediction = model_3.predict(train_data_x)

accuracy = sklearn.metrics.accuracy_score(train_data_y, prediction)
print(accuracy)

confusion_matrix = sklearn.metrics.confusion_matrix(train_data_y, prediction)
print(confusion_matrix)
print(confusion_matrix[1, 1]/confusion_matrix[1, 1] + confusion_matrix[1, 0])
print(confusion_matrix[0, 0]/confusion_matrix[0, 0] + confusion_matrix[0, 1])

confusion_matrix_df = pd.DataFrame(confusion_matrix, columns=["Predicted:0", "Predicted:1"], index=["Actual:0", "Actual:1"])

# pyplot.figure(figsize=(8, 6))
# ax = sns.heatmap(confusion_matrix_df, fmt="d", annot=True)
# ax.legend()
# pyplot.show()

probs = model_3.predict_proba(train_data_x)
probs = probs[:, 1]

proba_df = pd.DataFrame().from_dict({ "prob" : probs, "actual" : train_data_y })

print(proba_df)

for i in range(10):
    new_i = i / 10
    proba_df[new_i] = proba_df["prob"].apply(lambda x: 1 if x > new_i else 0)

print(proba_df)

# cm_array = []

# for i in range(10):
#     current_array = []
#     new_i = i / 10
#     cm = sklearn.metrics.confusion_matrix(proba_df["actual"], proba_df[new_i])
#     accuracy = (cm[0, 0] + cm[1, 1])/(cm[0, 0] + cm[1, 1] + cm[1, 0] + cm[0, 1])
#     sensitivity = cm[1, 1]/(cm[1, 1] + cm[1, 0])
#     specificity = cm[0, 0]/(cm[0, 0] + cm[0, 1])

#     current_array.append(new_i)
#     current_array.append(accuracy)
#     current_array.append(sensitivity)
#     current_array.append(specificity)

#     cm_array.append(current_array)

# cm_df = pd.DataFrame(cm_array, columns=["prob", "accuracy", "sensitivity", "specificity"])
# cm_df.dropna(inplace=True)

# print(cm_df)

# pyplot.figure(figsize=(8, 6))
# ax = sns.lineplot(cm_df, y="accuracy", x="prob")
# sns.lineplot(cm_df, y="sensitivity", x="prob")
# sns.lineplot(cm_df, y="specificity", x="prob")
# ax.legend(["Accuracy", "Sensitivity", "Specificity"])
# pyplot.show()

# Selected Threshold from accuracy, sensitivity and specificity tradeoff is 0.2
# Classes are not evenly distributed so I jave to use recall vs precision tradeoff

print(sklearn.metrics.recall_score(train_data_y, prediction))
print(sklearn.metrics.precision_score(train_data_y, prediction))

cm_array = []

for i in range(10):
    current_array = []
    new_i = i / 10
    cm = sklearn.metrics.confusion_matrix(proba_df["actual"], proba_df[new_i])
    recall = cm[1, 1]/(cm[1, 1] + cm[1, 0])
    precision = cm[1, 1]/(cm[1, 1] + cm[0, 1])

    current_array.append(new_i)
    current_array.append(recall)
    current_array.append(precision)

    cm_array.append(current_array)

cm_df = pd.DataFrame(cm_array, columns=["prob", "recall", "precision"])
cm_df.dropna(inplace=True)

print(cm_df)

# pyplot.figure(figsize=(8, 6))
# ax = sns.lineplot(cm_df, y="recall", x="prob")
# sns.lineplot(cm_df, y="precision", x="prob")
# ax.legend(["Recall", "Precision"])
# pyplot.show()

# Recall vs Precision gave us a 0.33 threshold.

test_data_x = test_data[qualified_columns]
test_data_y = test_data["Attrition"]

new_prediction = model_3.predict_proba(test_data_x)
new_prediction = new_prediction[:, 1]

prediction_df = pd.DataFrame({ "actual" : test_data_y, "predicted" : new_prediction })
prediction_df["predicted"] = prediction_df["predicted"].apply(lambda x: 1 if x > 0.33 else 0)

new_cm = sklearn.metrics.confusion_matrix(prediction_df["actual"], prediction_df["predicted"])

print(new_cm)

# Logistic regression proved to be bad at this.