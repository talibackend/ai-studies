import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns

print("Reading started...")

df = pd.read_csv('datasets/loan_data.csv')
# df = df.tail(200)
df['defaulted'] = df['loan_status'].apply(lambda x: 1 if x == 1 else 0)

print(df)

# print(df.describe())
# print(df.columns)
# print(df.info())

# df.fillna(0, inplace=True)

print(df.info())

df['loan_amnt'] = round(df['loan_amnt'])
df['paid_total'] = round(df['loan_amnt'] - df['revol_bal'])

# ax = plot.subplots(figsize=(10, 6))
# plot.hist([df.loan_amount, df.paid_total], label=["Loan Amount", "Funded Amount"])
# plot.legend()
# plot.show()
# sns.boxplot([df.loan_amount])
# plot.legend()
# plot.show()

total_count = int(round(df.annual_inc.count() * 99/100))
sliced_data = pd.DataFrame({ "defaulted" : df['defaulted'], "annual_inc" : df['annual_inc'] })
sliced_data.sort_values('annual_inc', inplace=True)
sliced_data = sliced_data.head(total_count)
print(sliced_data)

# plot.style.use('seaborn')
# ax = sns.boxplot(x="defaulted", y="annual_inc", data=sliced_data)
# ax.set_xlabel('Defaulted? (0=No/1=Yes)', fontsize=14)
# ax.set_ylabel('Annual Income', fontsize=14)
# plot.legend()
# plot.show()

# plot.figure(figsize=(10, 5))
# sns.set_style('dark')
# ax = sns.countplot(x="loan_status", data=df)
# ax.set_title("Loan Status", fontsize=14)
# ax.set_xlabel("Loan Status", fontsize=14)
# ax.set_ylabel("Loan Application Count")

# s = df['loan_status'].value_counts()
# total = s.sum()

# for i, j in s.reset_index().iterrows():
#     txt = str(j['count']) + " of " + str(total) + " / " + str(round(j['count']/total*100, 1)) + '%'
#     ax.text(i - 0.25, j.loan_status + 200, txt, color="k")

# plot.show()

# f, ax = plot.subplots(figsize=(10, 5))
# sns.heatmap(df.corr())
# plot.show()

# total_count = df.shape[0]
# print(total_count)
# sum_object = df.isna().sum()

# percent_object = {}

# for i in sum_object.keys():
#     count = sum_object[i]
#     percent_object[i] = round((count / total_count) * 100, 2)

# percent_df = pd.DataFrame.from_dict(percent_object, orient="index", columns=["Percentage of null values"])
# print(percent_df)

# string_columns = df.columns[df.dtypes == 'object']
# print(string_columns)

# for i in string_columns:
#     print(df[i].value_counts())
#     print('---------------------------')
    

# print(df.info())

# ct = pd.crosstab(df.int_rate, df.loan_amnt)
# print(ct)

# plot.figure(figsize=(8, 5))
# ct.plot()
# plot.legend()
# plot.show()