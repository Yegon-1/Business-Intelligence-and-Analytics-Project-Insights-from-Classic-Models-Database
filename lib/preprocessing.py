import pandas as pd

def clean_data(df):
    """
    Cleans a given dataframe by handling missing values and duplicates.
    Args:
        df (DataFrame): The raw dataframe to clean.
    Returns:
        DataFrame: Cleaned dataframe.
    """
    df = df.drop_duplicates()
    df = df.fillna(method='ffill')  # Forward fill for missing data
    return df
