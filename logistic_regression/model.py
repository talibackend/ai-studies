import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot
import helper
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset_1 = pd.read_csv('datasets/churn_data.csv')
dataset_2 = pd.read_csv('datasets/customer_data.csv')
dataset_3 = pd.read_csv('datasets/internet_data.csv')

dataset_4 = pd.merge(dataset_1, dataset_2, on='customerID', how='inner')
dataset = pd.merge(dataset_4, dataset_3, on='customerID', how='inner')

# Feature engineering start

def transformYesNo(field):
    print(field)
    if field.strip() == "Yes":
        print("It is yes...")
        return 1
    else:
        return 0

def parseStringFloat(field):
    try:
        return float(field)
    except:
        return 0

# dataset.drop("customerID", axis=1, inplace=True)
dataset["Churn"] = dataset["Churn"].apply(transformYesNo)
dataset["PhoneService"] = dataset["PhoneService"].apply(lambda x : transformYesNo(x))
dataset["PaperlessBilling"] = dataset["PaperlessBilling"].apply(lambda x : transformYesNo(x))
dataset["Partner"] = dataset["Partner"].apply(lambda x : transformYesNo(x))
dataset["Dependents"] = dataset["Dependents"].apply(lambda x : transformYesNo(x))
dataset["TotalCharges"] = dataset["TotalCharges"].apply(lambda x : parseStringFloat(x))

categoryFields = ["PaymentMethod", "Contract", "gender", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
# dataset = helper.normalizeCategoryFields(dataset, categoryFields)
# scaler = MinMaxScaler()
# columns = dataset.columns
# dataset[columns] = scaler.fit_transform(dataset)

print(dataset.describe())
print(dataset.info())
print(dataset.shape)
helper.printAllStringColumns(dataset)

print(dataset)

# Feature engineering end

# Exploratory Data Analysis

pyplot.figure(figsize=(8, 6))

# corr = dataset.corr()

# ax = sn.heatmap(corr)
ax = sn.countplot(dataset["Churn"])
ax.legend()

pyplot.show()

# helper.printCorrelations(dataset)
