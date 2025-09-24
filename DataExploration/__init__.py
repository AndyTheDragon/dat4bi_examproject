
import pandas as pd
import matplotlib.pyplot as plt

__all__ = ['descriptive_statistics', 'plots', 'read_data_to_dataframe', 'combine_dataframes', 'remove_duplicates']

from . import descriptive_statistics, plots

def read_data_to_dataframe(file_path, winetype) -> pd.DataFrame:
    """
    Reads an Excel file into a DataFrame, converts numeric columns,
    and adds a 'type' column.

    Args:
        file_path (str): Path to the Excel file.
        winetype (str): Label for the wine type.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df = pd.read_excel(file_path, header=1)
    # TODO2: determine datatype and use correct read function, e.g., pd.read_csv for CSV files
    # TODO1: checks for NaN or missing values?
    wine_df = df.apply(pd.to_numeric, errors='coerce')
    wine_df['type'] = winetype

    return wine_df

def combine_dataframes(dfs) -> pd.DataFrame:
    """
    Combines a list of DataFrames into a single DataFrame with a clean index.

    Args:
        dfs (list[pd.DataFrame]): List of DataFrames.

    Returns:
        pd.DataFrame: Combined DataFrame.
    """
    return pd.concat(dfs, ignore_index=True).reset_index(drop=True)

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate rows from the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    return df.drop_duplicates().reset_index(drop=True)
