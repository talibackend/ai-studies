import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import helper

data = pd.read_csv('datasets/CarPrice_Assignment.csv')

company_names_matches = {
    "maxda" : "mazda",
    "Nissan" : "nissan",
    "porcshce" : "porsche",
    "toyouta" : "toyota",
    "vw" : "volkswagen",
    "vokswagen" : "volkswagen"
}

def correct_company_name(name):
    keys = company_names_matches.keys()
    if name in keys:
        return company_names_matches[name]
    else:
        return name

print(data.head())
print(data.shape)
print(data.info())
print(helper.printAllStringColumns(data))
print(data.describe())

data["CompanyName"] = data["CarName"].apply(lambda x: correct_company_name(x.split(' ')[0]))
data.drop(columns=['car_ID', 'CarName'], inplace=True)

print(data.info())
print(helper.printAllStringColumns(data))

# plot.figure(figsize=(20, 7))

# plot.subplot(1, 2, 1)
# sns.distplot(data.price)

# plot.subplot(1, 2, 2)
# sns.boxplot(y=data.price)

# plot.subplot(1, 3, 1)
# ax = data["CompanyName"].value_counts().plot(kind='bar')
# plot.title("Companies histogram")

# plot.subplot(1, 3, 2)
# ax = data["fueltype"].value_counts().plot(kind='bar')
# plot.title("Fuel Type")

# plot.subplot(1, 3, 3)
# ax = data["carbody"].value_counts().plot(kind='bar')
# plot.title('Car Type')

# plot.subplot(1, 2, 1)
# sns.countplot(x=data.symboling)
# plot.title("Symboling Histogram")

# plot.subplot(1, 2, 2)
# sns.boxplot(x=data.symboling, y=data.price)
# plot.title("Symboling Box Plot")

# plot.subplot(1, 2, 1)
# sns.countplot(x=data.enginetype)
# plot.title("Engine Type Distribution")

# plot.subplot(1, 2, 2)
# ax = sns.boxplot(x=data.enginetype, y=data.price)
# plot.title("Engine Type Boxplot")

enginetype_df = pd.DataFrame(data.groupby(['enginetype'])['price'].mean().sort_values(ascending=False))
companyname_df = pd.DataFrame(data.groupby(['CompanyName'])['price'].mean().sort_values(ascending=False))
fueltype_df = pd.DataFrame(data.groupby(['fuelsystem'])['price'].mean().sort_values(ascending=False))
cartype_df = pd.DataFrame(data.groupby(['carbody'])['price'].mean().sort_values(ascending=False))

# print(enginetype_df)
# print(enginetype_df['price'][0])

# plot.subplot(1, 4, 1)
# sns.barplot(x=data.enginetype.unique(), y=enginetype_df.price)
# plot.legend()

# plot.subplot(1, 3, 1)
# sns.barplot(x=data.CompanyName.unique(), y=companyname_df.price)
# plot.legend()

# plot.subplot(1, 3, 2)
# sns.barplot(x=data.fuelsystem.unique(), y=fueltype_df.price)
# plot.legend()

# plot.subplot(1, 3, 3)
# sns.barplot(x=data.carbody.unique(), y=cartype_df.price)
# plot.legend()

# plot.figure(figsize=(8, 7))
# ax = sns.scatterplot(data=data, x="price", y="carlength")
# ax.set_title("Price VS Car Length")
# ax.set_ylabel("Car Length")
# ax.set_xlabel("Car Length")

# plot.show()

data['fueleconomy'] = (0.55 * data['citympg']) + (0.45 * data['highwaympg'])
print(data)

required_fields = [
    'price', 'fueltype', 'aspiration', 'carbody', 'drivewheel', 'wheelbase', 'curbweight',
    'enginetype', 'cylindernumber', 'enginesize', 'boreratio', 'horsepower', 'fueleconomy',
    'carlength', 'carwidth'
]

# clean_data = data[required_fields]
clean_data = data.drop(columns=[
    'symboling', 'carlength', 'compressionratio', 'horsepower', 'citympg', 'highwaympg', 'fueleconomy', 'doornumber',
    'fuelsystem', 'CompanyName', 'fueltype', 'carbody', 'drivewheel', 'wheelbase', 'carheight', 'curbweight', 'enginetype'
])
print(helper.printAllStringColumns(clean_data))

# clean_data = helper.normalizeCategoryFields(clean_data, ['fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginetype', 'cylindernumber'])
clean_data = helper.normalizeCategoryFields(clean_data, ['enginelocation', 'aspiration', 'cylindernumber'])
print(clean_data)
print(clean_data.info())
print(clean_data.shape)
print(helper.printCorrelations(clean_data))


train_data, test_data = train_test_split(clean_data, train_size=0.7, test_size=0.3, random_state=100)

print("Train data count {}".format(train_data.shape[0]))
print("Test data count {}".format(test_data.shape[0]))

columns = clean_data.columns

scaler = MinMaxScaler()
train_data[columns] = scaler.fit_transform(train_data)

print(train_data)

y_train = train_data['price']
train_data.drop('price', axis=1, inplace=True)
x_train = train_data
x_vif = helper.getVIF(x_train)

print(x_vif)

model_after_null_hypothesis = helper.buildModel(x_train, y_train)

# null_hypothesis_slim_fails = ['aspiration_turbo', 'aspiration_std', 'cylindernumber_four', 'cylindernumber_six', 'cylindernumber_five', 'cylindernumber_eight' ]

# x_train.drop(columns=null_hypothesis_slim_fails, axis=1, inplace=True)

# model = helper.buildModel(x_train, y_train)

helper.getVIF(train_data)
drop_via_vif = ['enginesize', 'boreratio']
x_train.drop(drop_via_vif, axis=1, inplace=True)
model_after_vif = helper.buildModel(x_train, y_train)

initial_data = data
initial_data = helper.normalizeCategoryFields(initial_data, ['fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginetype', 'cylindernumber', 'doornumber', 'enginelocation', 'fuelsystem', 'CompanyName'])

train_data, test_data = train_test_split(initial_data, train_size=0.7, test_size=0.3, random_state=100)
columns = train_data.columns
train_data[columns] = scaler.fit_transform(train_data)

y_train = train_data['price']
train_data.drop('price', axis=1, inplace=True)
x_train = train_data

lm = LinearRegression()
lm.fit(x_train, y_train)

model_rfe = RFE(lm)
model_rfe = model_rfe.fit(x_train, y_train)

evaluation = list(zip(x_train.columns, model_rfe.support_, model_rfe.ranking_))

print(len(columns))

for each in evaluation:
    print(each)

print(len(evaluation))

x_train = x_train[x_train.columns[model_rfe.support_]]

model_after_rfe = helper.buildModel(x_train, y_train)

helper.getVIF(x_train)

# This is better


