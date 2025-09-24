import pandas as pd

def describe_data(df: pd.DataFrame, verbose: bool = True, round_digits: int = 0) -> pd.DataFrame:
    """
    Generates descriptive statistics for the wine DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing wine data.
        verbose (bool): If True, prints the summary to the console.
        round_digits (int): If >0, round numeric values to this many decimals.
                            If 0, leave values as-is. Defaults to 0.

    Returns:
        pd.DataFrame: DataFrame with descriptive statistics.
    """
    summary = df.describe(include='all').transpose()
    summary = summary.drop(
        columns=[c for c in ['unique', 'top', 'freq'] if c in summary.columns],
        errors="ignore"
    )

    if round_digits > 0:
        summary = summary.round(round_digits)

    if 'count' in summary.columns:
        summary['count'] = summary['count'].astype('Int64')

    summary.index.name = "feature"

    if verbose:
        print(f"Summary: {df.shape[0]} rows × {df.shape[1]} columns")
        print(summary.to_string())

    return summary
