import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def clean_age(df):
    df['Age'] = df['Age'].str.strip()
    df = df[df['Age'].notna()]
    return df
