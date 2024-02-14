def printAllStringColumns(df):
    string_columns = df.columns[df.dtypes == "object"]
    print(string_columns)
    for column in string_columns:
        print("Unique values for " + column)
        print(df[column].unique())
        print("=======================")