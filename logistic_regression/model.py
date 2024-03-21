import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot
import helper
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from statsmodels.tools import add_constant as add_constant

dataset_1 = pd.read_csv('datasets/churn_data.csv')
dataset_2 = pd.read_csv('datasets/customer_data.csv')
dataset_3 = pd.read_csv('datasets/internet_data.csv')

dataset_4 = pd.merge(dataset_1, dataset_2, on='customerID', how='inner')
dataset = pd.merge(dataset_4, dataset_3, on='customerID', how='inner')

# Feature engineering start

def transformYesNo(field):
    if field == "Yes":
        return 1
    else:
        return 0

def parseStringFloat(field):
    try:
        return float(field)
    except:
        return 0

dataset.drop("customerID", axis=1, inplace=True)
dataset["Churn"] = dataset["Churn"].apply(transformYesNo)
dataset["PhoneService"] = dataset["PhoneService"].apply(lambda x : transformYesNo(x))
dataset["PaperlessBilling"] = dataset["PaperlessBilling"].apply(lambda x : transformYesNo(x))
dataset["Partner"] = dataset["Partner"].apply(lambda x : transformYesNo(x))
dataset["Dependents"] = dataset["Dependents"].apply(lambda x : transformYesNo(x))
dataset["TotalCharges"] = dataset["TotalCharges"].apply(lambda x : parseStringFloat(x))

categoryFields = ["PaymentMethod", "Contract", "gender", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
dataset = helper.normalizeCategoryFields(dataset, categoryFields)

dataset = dataset[dataset["TotalCharges"] != 0]

print(dataset.describe())
print(dataset.info())
print(dataset.shape)

# Feature engineering end

# Exploratory Data Analysis

# helper.printCorrelations(dataset)
# pyplot.figure(figsize=(8, 6))
# corr = dataset.corr()
# ax = sn.heatmap(corr)

# pyplot.figure(figsize=(8, 6))
# ax = sn.countplot(x="Churn", data=dataset)
# ax.legend()


# pyplot.figure(figsize=(8, 6))
# ax = sn.boxplot(data=dataset, y="TotalCharges")

# pyplot.figure(figsize=(8, 6))
# ax = sn.barplot(data=dataset, x="TotalCharges", y="Churn")
# ax.legend()

# monthToMonth = pd.DataFrame(dataset["Churn"].where(dataset["Contract_Month-to-month"] == True)).dropna()
# oneYear = pd.DataFrame(dataset["Churn"].where(dataset["Contract_One year"] == True)).dropna()
# twoYear = pd.DataFrame(dataset["Churn"].where(dataset["Contract_Two year"] == True)).dropna()
# paymentMethodElectronic = pd.DataFrame(dataset["Churn"].where(dataset["PaymentMethod_Electronic check"] == True)).dropna()
# paymentMethodMailed = pd.DataFrame(dataset["Churn"].where(dataset["PaymentMethod_Mailed check"] == True)).dropna()
# paymentMethodBank = pd.DataFrame(dataset["Churn"].where(dataset["PaymentMethod_Bank transfer (automatic)"] == True)).dropna()
# paymentMethodCredit = pd.DataFrame(dataset["Churn"].where(dataset["PaymentMethod_Credit card (automatic)"] == True)).dropna()
# multipleLinesNo = pd.DataFrame(dataset["Churn"].where(dataset["MultipleLines_No"] == True)).dropna()
# multipleLinesYes = pd.DataFrame(dataset["Churn"].where(dataset["MultipleLines_Yes"] == True)).dropna()
# multipleLinesNoPhone = pd.DataFrame(dataset["Churn"].where(dataset["MultipleLines_No phone service"] == True)).dropna()

# print(monthToMonth)
# print(oneYear)
# print(twoYear)

# print(paymentMethodElectronic)
# print(paymentMethodMailed)
# print(paymentMethodBank)
# print(paymentMethodCredit)

# print(multipleLinesNo)
# print(multipleLinesYes)
# print(multipleLinesNoPhone)

# pyplot.figure(figsize=(8, 6))
# ax = sn.countplot(data=monthToMonth, x="Churn")
# ax.set_title("Month To Month Plan")
# ax.legend()

# pyplot.figure(figsize=(8, 6))
# ax = sn.countplot(data=oneYear, x="Churn")
# ax.set_title("One Year Plan")
# ax.legend()

# pyplot.figure(figsize=(8, 6))
# ax = sn.countplot(data=twoYear, x="Churn")
# ax.set_title("Two Year Plan")
# ax.legend()


# pyplot.figure(figsize=(8, 6))
# ax = sn.countplot(data=paymentMethodElectronic, x="Churn")
# ax.set_title("Payment Electronic")
# ax.legend()

# pyplot.figure(figsize=(8, 6))
# ax = sn.countplot(data=paymentMethodMailed, x="Churn")
# ax.set_title("Payment Mailed")
# ax.legend()

# pyplot.figure(figsize=(8, 6))
# ax = sn.countplot(data=paymentMethodBank, x="Churn")
# ax.set_title("Payment Bank")
# ax.legend()

# pyplot.figure(figsize=(8, 6))
# ax = sn.countplot(data=paymentMethodCredit, x="Churn")
# ax.set_title("Payment Credit")
# ax.legend()


# pyplot.figure(figsize=(8, 6))
# ax = sn.countplot(data=multipleLinesNo, x="Churn")
# ax.set_title("Multiple Lines(NO)")
# ax.legend()

# pyplot.figure(figsize=(8, 6))
# ax = sn.countplot(data=multipleLinesYes, x="Churn")
# ax.set_title("Multiple Lines(YES)")
# ax.legend()

# pyplot.figure(figsize=(8, 6))
# ax = sn.countplot(data=multipleLinesNoPhone, x="Churn")
# ax.set_title("Multiple Lines(NO PHONE SERVICE)")
# ax.legend()


# pyplot.show()

# End exploratory Data Analysis

# Model

print(dataset.isna().sum())
# helper.printAllStringColumns(dataset, False)

scaler = StandardScaler()
columns = dataset.columns
variable_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
dataset[variable_columns] = scaler.fit_transform(dataset[variable_columns])

print(dataset.corr())
fields = helper.printCorrelations(dataset)
print(fields)

dataset = dataset.drop(list(fields), axis=1)
# dataset = dataset.drop(['TotalCharges', 'InternetService_Fiber optic', 'OnlineSecurity_No internet service', 'OnlineBackup_No internet service', 'StreamingMovies_No internet service', 
#                 'TechSupport_No internet service', 'StreamingTV_No internet service', 'StreamingMovies_No internet service', 'DeviceProtection_No internet service'], axis=1)

train_data, test_data = train_test_split(dataset, test_size=0.3, train_size=0.7, random_state=100)

train_data = train_data.astype(float)

train_data_y = train_data["Churn"]
train_data_x = train_data.drop("Churn", axis=1)

print(train_data_y)
print(train_data_x)

model_1 = helper.buildModel(train_data_x, train_data_y)

qualified_columns = ["PaperlessBilling", "Partner", "SeniorCitizen"]

train_data_x = train_data_x[qualified_columns]

model_2 = helper.buildModel(train_data_x, train_data_y)

model_3 = LogisticRegression()
model_3.fit(train_data_x, train_data_y)

test_data = test_data.astype(float)
test_data_y = test_data["Churn"]
test_data_x = test_data.drop("Churn", axis=1)
test_data_x = test_data_x[qualified_columns]

predictions = model_3.predict(test_data_x)

prediction_df = pd.DataFrame()
prediction_df["Actual"] = test_data_y
prediction_df["Predicted"] = predictions
prediction_df["Correct"] =  prediction_df.apply(lambda row: 1 if row['Actual'] == row['Predicted'] else 0, axis=1)

print(prediction_df)


# pyplot.figure(figsize=(8, 6))
# ax = sn.countplot(data=prediction_df, x="Actual")
# ax.set_title("Actual")

# pyplot.figure(figsize=(8, 6))
# ax = sn.countplot(data=prediction_df, x="Predicted")
# ax.set_title("Predicted")

# pyplot.figure(figsize=(8, 6))
# ax = sn.countplot(data=prediction_df, x="Correct")
# ax.set_title("Correctness")

# pyplot.show()