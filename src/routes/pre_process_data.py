from typing import Optional
import pandas as pd
from src.config import ApplicationConfig
from src.services.data_processor import DataProcessor
from src.services.failure_analyzer import FailureAnalyzer

class PreProcessDataRoute:
    def __init__(self, config: Optional[ApplicationConfig] = None,
                 data_processor: Optional[DataProcessor] = None,
                 failure_analyzer: Optional[FailureAnalyzer] = None):
        """
        """
        self.config = config or ApplicationConfig()
        self.data_processor = data_processor or DataProcessor()
        self.failure_analyzer = failure_analyzer or FailureAnalyzer()

    def run(self):
        print("\n1. LOADING AND PROCESSING DATA...")
        raw_df = pd.read_csv(self.config.data_path)

        print(f"Loaded {len(raw_df)} records from raw data")
        df = self.data_processor.convert_comma_decimals(raw_df)

        # Create failure index and export data
        processed_df = self.failure_analyzer.create_failure_index(df)
        print(f"Processed data shape: {processed_df.shape}")
        print(f"Features created: {processed_df.shape[1] - raw_df.shape[1]} new features")
        processed_df.to_csv(self.config.processed_data_path, index=False)
        print(f"Processed data saved to: {self.config.processed_data_path}")
        
        return {
            "processed_df": processed_df,
            "processed_shape": processed_df.shape
        }