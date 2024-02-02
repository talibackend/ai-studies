import pandas as pd
import numpy as np

arr = np.array([1, 2, 4, 4])
ser = pd.Series(arr, index=['a', 'b', 'c', 'd'])
print(ser)

dic = {"name" : "Femi", "age" : 22}
ser = pd.Series(dic)
print(ser)

ser = pd.Series([1, 2, 3, 4, 5, 6], index=['a', 'b', 'c', 'd', 'e', 'f'])
print(ser.array)
print(ser.axes)
print(ser.dtypes)
print(ser.describe())

ser[0] = np.nan
print(ser)
print(ser.dropna())
print(ser.fillna(0))

ser = pd.Series([90, 50, 70, 51, 77, 88, 40, 11, 72], index=['HR', 'IT', 'IT', 'HR', 'HR', 'HR', 'IT', 'IT', 'IT'])
print(ser)
print(ser.groupby(level=0).mean())
print(ser.groupby(level=0).max())
print(ser.groupby(level=0).min())
print(ser.groupby(level=0).std())
print(ser.groupby(level=0).sum())