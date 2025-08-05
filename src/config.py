from pathlib import Path

class ApplicationConfig:
    """
    Configuration class for the application.
    """
    def __init__(self):
        # Get the root project directory (parent of src folder)
        project_root = Path(__file__).parent.parent
        
        self.data_path = project_root / 'data' / 'raw' / 'Test O_G_Equipment_Data.xlsx - O&G Equipment Data.csv'
        assert self.data_path.exists(), f"Data file not found at {self.data_path}"

        self.processed_data_path = Path(project_root / 'data' / 'processed' / 'df_fpso_failure_monitoring_processed.csv')
        self.processed_data_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

        self.sensor_columns = ['Temperature','Pressure', 'VibrationX', 'VibrationY','VibrationZ', 'Frequency']
        self.categorical_columns = ['Preset_1', 'Preset_2']