import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns

data = pd.read_csv('datasets/car_sales.csv')

vehicle_types = {
    "Passenger" : 1,
    "Car" : 2
}
    

print(data)
print(data.info())
print(data.isna())
print(data["Vehicle_type"].unique())

data["Vehicle_type"] = data["Vehicle_type"].apply(lambda x: vehicle_types[x] )


# plot.figure(figsize=(8, 5))
# sns.heatmap(data.corr())
# plot.legend()
# plot.show()

# ax = sns.barplot(data=data, x="Power_perf_factor", y="Price_in_thousands")
# ax.set_ylabel("Price in thousands($)")
# ax.set_xlabel("Power performance factor")
# plot.legend()
# plot.show()

# ax = sns.barplot(data=data, x="Horsepower", y="Price_in_thousands")
# ax.set_ylabel("Price in thousands($)")
# ax.set_xlabel("Horsepower")
# plot.legend()
# plot.show()
