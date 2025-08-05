"""
Feature engineering module for FPSO equipment monitoring data.

This module provides feature engineering functions for creating new features
from existing sensor data and operational parameters.
"""

import pandas as pd
from typing import List, Optional, Dict, Any
import numpy as np


class FeatureEngineer:
    """
    A class for feature engineering operations including rolling windows,
    threshold alerts, and domain-specific feature creation.
    """
    
    def __init__(self):
        """Initialize the FeatureEngineer."""
        pass
    
    # ================================
    # Rolling Window Features
    # ================================
    
    def add_sensor_rolling_features(self, df: pd.DataFrame, sensor_cols: List[str], 
                                   windows: List[int] = [3, 5]) -> pd.DataFrame:
        """
        Adds rolling window features (mean, std, percent change) for specified sensor columns and window sizes.

        For each column in sensor_cols and each window in windows, the following features are added:
            - Rolling mean: {col}_rolling_mean_{window}
            - Rolling std: {col}_rolling_std_{window}
            - Percent change over window: {col}_pct_change_{window}

        Args:
            df (pd.DataFrame): The input DataFrame.
            sensor_cols (List[str]): List of sensor column names to compute features for.
            windows (List[int], optional): List of window sizes for rolling calculations. Defaults to [3, 5].

        Returns:
            pd.DataFrame: DataFrame with new rolling window features added.
        """
        window_features = df.copy()
        
        for window in windows:
            for col in sensor_cols:
                # Rolling mean
                window_features[f'{col}_rolling_mean_{window}'] = window_features[col].rolling(window).mean()
                # Rolling standard deviation
                window_features[f'{col}_rolling_std_{window}'] = window_features[col].rolling(window).std()
                # Percent change over the window
                window_features[f'{col}_pct_change_{window}'] = window_features[col].pct_change(periods=window)
                
        return window_features
    
    # ================================
    # Threshold and Alert Features
    # ================================
    
    def add_sensor_threshold_features(self, df: pd.DataFrame, sensor_cols: List[str],
                                     thresholds: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Add threshold-based alert features for sensors.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            sensor_cols (List[str]): List of sensor columns.
            thresholds (Optional[Dict[str, float]]): Custom thresholds. 
                                                   If None, uses 90th and 95th percentiles.
        
        Returns:
            pd.DataFrame: DataFrame with threshold alert features.
        """
        df_threshold = df.copy()

        for col in sensor_cols:
            if thresholds and col in thresholds:
                # Use custom threshold
                threshold_high = thresholds[col]
                df_threshold[f'{col}_high_alert'] = (df_threshold[col] > threshold_high).astype(int)
            else:
                # Use percentile-based thresholds
                threshold_high = df[col].quantile(0.9)
                threshold_extreme = df[col].quantile(0.95)
                
                df_threshold[f'{col}_high_alert'] = (df_threshold[col] > threshold_high).astype(int)
                df_threshold[f'{col}_extreme_alert'] = (df_threshold[col] > threshold_extreme).astype(int)

        return df_threshold
    
    # ================================
    # Domain-Specific Features
    # ================================
    
    def add_high_risk_preset_features(self, df: pd.DataFrame, 
                                     risk_combinations: Optional[List[tuple]] = None) -> pd.DataFrame:
        """
        Add binary features for high-risk preset combinations.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            risk_combinations (Optional[List[tuple]]): List of (Preset_1, Preset_2) risk combinations.
                                                      If None, uses default risk combinations.
        
        Returns:
            pd.DataFrame: DataFrame with high-risk preset features.
        """
        df_risk = df.copy()
        
        # Default high-risk combinations from EDA findings
        if risk_combinations is None:
            risk_combinations = [
                (1, 5),  # Preset_1=1, Preset_2=5
                (3, 5),  # Preset_1=3, Preset_2=5
                (1, 2),  # Preset_1=1, Preset_2=2
            ]
        
        # Create features for each risk combination
        for i, (preset_1, preset_2) in enumerate(risk_combinations):
            feature_name = f'high_risk_combo_{preset_1}_{preset_2}'
            df_risk[feature_name] = ((df_risk['Preset_1'] == preset_1) & 
                                   (df_risk['Preset_2'] == preset_2)).astype(int)
        
        return df_risk
    
    # ================================
    # Complete Feature Engineering Pipeline
    # ================================
    
    def complete_feature_engineering(self, df: pd.DataFrame, sensor_cols: List[str],
                                     windows: List[int] = [3, 5]) -> pd.DataFrame:
        """
        Apply complete feature engineering pipeline.
        """
        df = self.add_sensor_rolling_features(df, sensor_cols, windows)
        df = self.add_sensor_threshold_features(df, sensor_cols)
        df = self.add_high_risk_preset_features(df)

        return df
