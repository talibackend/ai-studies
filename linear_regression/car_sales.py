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

clean_data = data[required_fields]
print(helper.printAllStringColumns(clean_data))

normalized_data = helper.normalizeStringFields(clean_data)
print(normalized_data)

# print(clean_data.corr())

# plot.figure(figsize=(10, 5))
# sns.pairplot(clean_data)
# sns.heatmap(clean_data.corr())
# plot.show()
