import pandas as pd

def load_and_clean_data():
    df1 = pd.read_csv("datasets/Audible_Catlog.csv")
    df2 = pd.read_csv("datasets/Audible_Catlog_Advanced_Features.csv")

    df = pd.merge(df1, df2, on=["Book Name", "Author", "Rating", "Number of Reviews", "Price"], how="outer")
    df.drop_duplicates(subset="Book Name", inplace=True)
    df.dropna(subset=["Description"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
