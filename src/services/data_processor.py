"""
Data processing module for FPSO equipment monitoring data.

This module provides data cleaning, transformation, and preprocessing functions
for exploratory data analysis and machine learning.
"""

import pandas as pd
from typing import List, Union
import numpy as np


class DataProcessor:
    """
    A class for data processing operations including cleaning, encoding, and feature engineering.
    """
    
    def __init__(self):
        """Initialize the DataProcessor."""
        pass
    
    # ================================
    # Data Cleaning Methods
    # ================================
    
    def parse_comma_decimal_series(self, series: pd.Series) -> pd.Series:
        """Convert object columns with commas to float."""
        return series.str.replace(',', '.').astype('float64')

    def find_comma_decimal_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Returns a list of columns that contain commas in their values.
        """
        return [
            col for col in df.columns
            if df[col].dtype == 'object' and df[col].str.contains(',').any()
        ]

    def convert_comma_decimals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adjusts the data types of columns in the DataFrame.
        Converts object columns with commas to float.
        """
        obj_columns = self.find_comma_decimal_columns(df)
        df_copy = df.copy()
        for col in obj_columns:
            df_copy[col] = self.parse_comma_decimal_series(df_copy[col])
        return df_copy
    
    # ================================
    # Encoding Methods
    # ================================
    
    def map_bool_to_int(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Maps boolean values in the specified column to integers: True → 1, False → 0.

        Args:
            df (pd.DataFrame): The input DataFrame.
            column (str): The name of the column to convert.

        Returns:
            pd.DataFrame: Updated DataFrame with the column converted to integers.
        """
        df_copy = df.copy()
        df_copy[column] = df_copy[column].astype(int)
        return df_copy

    def one_hot_encode_column(self, df: pd.DataFrame, column: str, prefix: str = None, drop_original: bool = True) -> pd.DataFrame:
        """
        Applies one-hot encoding to a single column in the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.
            column (str): The name of the column to encode.
            prefix (str, optional): Prefix for new columns. Defaults to column name.
            drop_original (bool): Whether to drop the original column. Defaults to True.

        Returns:
            pd.DataFrame: DataFrame with one-hot encoded columns.
        """
        df_copy = df.copy()
        if prefix is None:
            prefix = column

        dummies = pd.get_dummies(df_copy[column], prefix=prefix)
        df_encoded = pd.concat([df_copy, dummies], axis=1)

        if drop_original:
            df_encoded = df_encoded.drop(columns=[column])

        return df_encoded

    def encode_dataset(self, df: pd.DataFrame, one_hot_columns: List[str] = None) -> pd.DataFrame:
        """
        Encodes the DataFrame by mapping boolean columns to integers and applying one-hot encoding to specified columns.

        Args:
            df (pd.DataFrame): The input DataFrame.
            one_hot_columns (List[str], optional): List of columns to apply one-hot encoding. Defaults to None.

        Returns:
            pd.DataFrame: Encoded DataFrame.
        """
        df_encoded = df.copy()
        
        if one_hot_columns is not None:
            for col in one_hot_columns:
                df_encoded = self.one_hot_encode_column(df_encoded, col)

        for col in df_encoded.select_dtypes(include=['bool']).columns:
            df_encoded = self.map_bool_to_int(df_encoded, col)

        return df_encoded
    

