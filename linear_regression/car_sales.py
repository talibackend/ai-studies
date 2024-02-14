import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
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

plot.figure(figsize=(15, 7))

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

plot.show()
