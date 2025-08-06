import pandas as pd
from typing import List, Optional
from ..services.data_processor import DataProcessor
from ..services.feature_engineer import FeatureEngineer


class FeatureProcessingRoutes:
    """
    Orchestrates the feature engineering and data processing pipeline.

    This class combines data cleaning, feature engineering, and encoding steps
    to produce a fully processed DataFrame ready for modeling.

    Args:
        data_processor (DataProcessor, optional): DataProcessor instance.
        feature_engineer (FeatureEngineer, optional): FeatureEngineer instance.
    """
    
    def __init__(self, data_processor: Optional[DataProcessor] = None, 
                 feature_engineer: Optional[FeatureEngineer] = None):
        """
        Initialize the FeatureProcessingRoutes.

        Args:
            data_processor (Optional[DataProcessor]): DataProcessor instance. 
            feature_engineer (Optional[FeatureEngineer]): FeatureEngineer instance.
        """
        self.data_processor = data_processor or DataProcessor()
        self.feature_engineer = feature_engineer or FeatureEngineer()
    
    def run(self, df: pd.DataFrame, categorical_columns: List[str], 
                            sensor_columns: List[str], windows: List[int] = [3, 5]) -> pd.DataFrame:
        """
        Complete processing pipeline for the dataset.

        This method orchestrates the complete data processing workflow by calling
        individual DataProcessor and FeatureEngineer methods in the correct sequence.

        Args:
            df (pd.DataFrame): Raw input DataFrame.
            categorical_columns (List[str]): Columns to one-hot encode.
            sensor_columns (List[str]): Sensor columns for feature engineering.
            windows (List[int]): Rolling window sizes.

        Returns:
            pd.DataFrame: Fully processed DataFrame.
        """
        # Step 1: Clean data types
        df_clean = self.data_processor.convert_comma_decimals(df)
        
        # Step 2: Feature engineering pipeline
        df_engineered = self.feature_engineer.complete_feature_engineering(
            df_clean, 
            sensor_columns, 
            windows=windows
        )
        df_engineered.fillna(0, inplace=True)

        # Step 3: Encode categorical variables
        df_final = self.data_processor.encode_dataset(df_engineered, one_hot_columns=categorical_columns)

        return df_final

