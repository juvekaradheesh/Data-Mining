from sklearn.preprocessing import LabelEncoder
import pandas as pd

def getDataset():
    df = pd.read_csv("processed_adult.csv")

    lb_make = LabelEncoder()
    df["Sex"] = lb_make.fit_transform(df["Sex"])
    df["Race"] = lb_make.fit_transform(df["Race"])
    df["Relationship"] = lb_make.fit_transform(df["Relationship"])
    df["Occupation"] = lb_make.fit_transform(df["Occupation"])
    df["Marital_stat"] = lb_make.fit_transform(df["Marital_stat"])
    df["WorkClass"] = lb_make.fit_transform(df["WorkClass"])
    df["Education"] = lb_make.fit_transform(df["Education"])
    df["Final_wt"] = lb_make.fit_transform(df["Final_wt"])
    df["Hr_per_week"] = lb_make.fit_transform(df["Hr_per_week"])

    df['Salary'] = lb_make.fit_transform(df['Salary'])

    X = df.drop(['Salary'],axis=1)
    Y = df['Salary']
    return X,Y

def getDatasetTest():
    df_test = pd.read_csv("processed_test.csv")
    
    lb_make = LabelEncoder()
    df_test["Sex"] = lb_make.fit_transform(df_test["Sex"])
    df_test["Race"] = lb_make.fit_transform(df_test["Race"])
    df_test["Relationship"] = lb_make.fit_transform(df_test["Relationship"])
    df_test["Occupation"] = lb_make.fit_transform(df_test["Occupation"])
    df_test["Marital_stat"] = lb_make.fit_transform(df_test["Marital_stat"])
    df_test["WorkClass"] = lb_make.fit_transform(df_test["WorkClass"])
    df_test["Education"] = lb_make.fit_transform(df_test["Education"])
    df_test["Final_wt"] = lb_make.fit_transform(df_test["Final_wt"])
    df_test["Hr_per_week"] = lb_make.fit_transform(df_test["Hr_per_week"])

    df_test['Salary'] = lb_make.fit_transform(df_test['Salary'])

    X_test = df_test.drop(['Salary'],axis=1)
    Y_test = df_test['Salary']

    return X_test,Y_test