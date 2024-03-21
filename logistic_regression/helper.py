from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import statsmodels.api as sm 

def printAllStringColumns(df, dtype="object"):
    if not dtype:
        string_columns = df.columns
    else:
        string_columns = df.columns[df.dtypes == "object"]
    print(string_columns)
    for column in string_columns:
        print("Unique values for " + column)
        print(df[column].unique())
        print("=======================")

def normalizeStringFields(df):
    string_columns = df.columns[df.dtypes == "object"]
    full_map = {}
    for column in string_columns:
        values = df[column].unique()
        current_column_map = {}
        for i in range(len(values)):
            value = values[i]
            current_column_map[value] = i + 1

        df[column].replace(current_column_map, inplace=True)
        full_map[column] = current_column_map
    
    print("Done with normalizing")
    print("==>")
    print(full_map)
    print("==>")
    return df

def normalizeCategoryFields(df, columns):
    for column in columns:
        values = df[column].unique()
        for i in range(len(values)):
            value = values[i]
            column_name = "{}_{}".format(column, value)
            df[column_name] = (df[column] == value)
        
        df.drop(column, axis=1, inplace=True)
    
    return df

def printCorrelations(df, min=0.8):
    cor = df.corr()
    columns = cor.columns
    correlated_field = set()
    for column in columns:
        each_cor = cor[column].to_dict()
        keys = each_cor.keys()
        
        for key in keys:
            value = abs(each_cor[key])
            if key != column and value >= min:
                correlated_field.add(column)
                # correlated_field.add(key)
                print("Correlation between {} and {} is {}".format(column, key, value))
    
    return correlated_field

def dropHighlyCorrelatedFields(df, threshold=0.8):
    high_correlations = printCorrelations(df, threshold)
    cols = df.columns
    cols_to_drop = []

    for each in high_correlations:
        print(each)
    return df


def buildModel(x, y):
    x = sm.add_constant(x)
    logit = sm.Logit(y, x[x.columns])
    result = logit.fit()
    print(result.summary())
    return logit
