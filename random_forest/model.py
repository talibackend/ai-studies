
import numpy as np
from matplotlib import pyplot
import sklearn
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("datasets/credit-card-default.csv").drop("ID", axis=1)
