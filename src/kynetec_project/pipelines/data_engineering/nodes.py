import pandas as pd
from typing import Any, Dict, Tuple, Callable, List


def load_dataset(df: Any) -> pd.DataFrame:
    """
    Load the paruet object to a csv file.
    Args:
        df: parquet object.

    Returns:
        Data in a csv file format.
    """
    return df


def filter_dataset(df: pd.DataFrame, corn_filters: Dict) -> pd.DataFrame:
    """
    Filters the dataframe columns with the specified filtered values.
    Args:
        df: A pandas dataframe
        corn_filters: Specified filters with given columns and respective values.

    Returns:
        A filtered dataframe.

    """
    filter_values = corn_filters['values']
    filter_exclude_values = corn_filters['values_exclude']
    for column in corn_filters['columns']:
        df = df[df[column].isin(filter_values)]
    for column in corn_filters['columns_exclude']:
        df = df[~df[column].isin(filter_exclude_values)]
    return df.reset_index(drop=True)


def fix_datatypes(df: pd.DataFrame, corn_fix_datatypes: Dict) -> pd.DataFrame:
    """
    Modify columns with specified data types.
    Args:
        df: A pandas dataframe.
        corn_fix_datatypes: Specified columns with new data types.

    Returns:
        A dataframe with modified data types
    """
    for column, datatype in zip(corn_fix_datatypes['columns'], corn_fix_datatypes['datatypes']):
        df[column] = df[column].str.replace(',', '').astype(datatype)
    return df

