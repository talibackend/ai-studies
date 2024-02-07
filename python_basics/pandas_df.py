import numpy as np
import pandas as pd

dictionary = {
    "Name" : ["Femi", "Alabi", "Fatokun"],
    "Age" : [18, 24, 22],
}

df = pd.DataFrame(dictionary)

print(df)

arr = np.array([['Ford', 'Toyota'], [1500, 1000]])
dictionary = {
    "Car" : arr[0],
    "Price" : arr[1]
}
df = pd.DataFrame(dictionary)

print(df)

df = pd.DataFrame([['Ford', 1500], ['Toyota', 1000]], columns=['Car', 'Price'])

print(df)

rng = pd.date_range('2020-01-01 00:00:00', periods=30, freq="D")

arr1 = np.random.random(30)
arr2 = np.random.random(30)

df = pd.DataFrame({ 'arr1' : arr1, 'arr2' : arr2 }, index=rng)
print(df)

print(df.describe())
print(df.transpose())
print(df.sort_values("arr1", ascending=False))

# df.to_csv('df.csv')