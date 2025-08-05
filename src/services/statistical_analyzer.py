"""
Statistical analysis module for FPSO equipment monitoring data.

This module provides statistical analysis functions for exploring patterns,
creating summaries, and analyzing failures.
"""

import pandas as pd
from typing import List, Union, Tuple, Optional
import numpy as np


class StatisticalAnalyzer:
    """
    A class for statistical analysis operations including summary statistics,
    failure analysis, and pattern detection.
    """
    
    def __init__(self):
        """Initialize the StatisticalAnalyzer."""
        pass
    
    # ================================
    # Summary Statistics Methods
    # ================================
    
    def create_stats_summary(self, df: pd.DataFrame, target_col: Union[str, List[str]] = 'Fail_index',
                            agg_funcs: List[str] = ['mean', 'max', 'median']) -> pd.DataFrame:
        """Creates a summary DataFrame with specified aggregation functions for each sensor column.
        
        Args:
            df (pd.DataFrame): DataFrame containing the data.
            target_col (Union[str, List[str]]): Target column name(s) for grouping.
            agg_funcs (List[str]): List of aggregation functions to apply.
            
        Returns:
            pd.DataFrame: DataFrame with aggregated statistics.
        """
        # Get float columns for sensor data
        sensored_cols = df.select_dtypes(include=['float64']).columns.tolist()

        # Create aggregation dictionary
        agg_dict = {col: agg_funcs for col in sensored_cols}
        
        # Handle single or multiple target columns for count
        if isinstance(target_col, str):
            agg_dict[target_col] = 'count'  # duration in operation cycles
        elif isinstance(target_col, list):
            # For multiple columns, use the first one for count
            agg_dict[target_col[0]] = 'count'
        
        # Group by target column(s) and aggregate
        result_df = df.groupby(target_col).agg(agg_dict)
        
        return result_df
    
    def mean_difference_by_target(self, df: pd.DataFrame, columns: List[str], target: str = 'Fail') -> pd.Series:
        """
        Calculates the mean difference for the specified columns between True and False states of the target variable.

        Args:
            df (pd.DataFrame): The dataframe containing the data.
            columns (list): List of column names to calculate mean difference for.
            target (str): The binary target column name.

        Returns:
            pd.Series: Mean difference (True - False) for each column.
        """
        assert target in df.columns, f"Target column '{target}' not found in DataFrame."
        assert all(col in df.columns for col in columns), "One or more specified columns not found in DataFrame."
        
        means_true = df[df[target] == True][columns].mean()
        means_false = df[df[target] == False][columns].mean()

        return means_true - means_false
    
    # ================================
    # Failure Analysis Methods
    # ================================
    
    def get_buffer_around_index(self, df: pd.DataFrame, 
                               start_index: int, 
                               stop_index: int,
                               buffer_before: int = 5,
                               buffer_after: int = 5,
                               exclude_failure_cycles: bool = False) -> pd.DataFrame:
        """
        Extracts cycles before and after a specific index for analysis of patterns around that point.
        
        Args:
            df (pd.DataFrame): DataFrame with the data
            start_index (int): The start index around which to create the buffer
            stop_index (int): The stop index around which to create the buffer
            buffer_before (int): Number of cycles before the start index to include
            buffer_after (int): Number of cycles after the stop index to include
            exclude_failure_cycles (bool): If True, excludes cycles that are already in failure
            
        Returns:
            pd.DataFrame: DataFrame containing only the cycles in the buffer around target index
        """
        buffer_indices = []
        
        # Calculate the range of the buffer
        start_buffer = start_index - buffer_before
        end_buffer = stop_index + buffer_after-1
        
        # Ensure we don't go beyond the dataset boundaries
        start_buffer = max(start_buffer, df.index.min())
        end_buffer = min(end_buffer, df.index.max())
        
        # Add indices to buffer if they exist in the dataset
        for idx in range(start_buffer, end_buffer + 1):
            if idx in df.index:
                # If exclude_failure_cycles=True, only add if not in failure
                if exclude_failure_cycles:
                    if not df.loc[idx, 'Fail']:
                        buffer_indices.append(idx)
                else:
                    buffer_indices.append(idx)
        
        # Remove duplicates and sort
        buffer_indices = sorted(list(set(buffer_indices)))
        
        return df.loc[buffer_indices]
    
        
    # ================================
    # Correlation and Relationship Analysis
    # ================================
    
    def analyze_correlations(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate correlation matrix for specified columns or all numeric columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns (Optional[List[str]]): Specific columns to analyze. If None, uses all numeric columns.
            
        Returns:
            pd.DataFrame: Correlation matrix
        """
        if columns is None:
            numeric_data = df.select_dtypes(include=[np.number])
        else:
            numeric_data = df[columns]
        
        return numeric_data.corr()
    
    def find_high_correlations(self, df: pd.DataFrame, threshold: float = 0.7, 
                              columns: Optional[List[str]] = None) -> List[Tuple[str, str, float]]:
        """
        Find pairs of variables with high correlation.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            threshold (float): Correlation threshold for "high" correlation
            columns (Optional[List[str]]): Specific columns to analyze
            
        Returns:
            List[Tuple[str, str, float]]: List of (var1, var2, correlation) tuples
        """
        corr_matrix = self.analyze_correlations(df, columns)
        
        # Find high correlations (excluding self-correlations)
        high_corrs = []
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:  # Avoid duplicates and self-correlations
                    corr_val = corr_matrix.loc[col1, col2]
                    if abs(corr_val) >= threshold:
                        high_corrs.append((col1, col2, corr_val))
        
        # Sort by absolute correlation value
        high_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
        return high_corrs