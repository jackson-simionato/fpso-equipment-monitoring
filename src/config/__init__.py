from pathlib import Path

class ApplicationConfig:
    """
    Configuration class for the application.
    """
    def __init__(self):
        # Get the root project directory (parent of src folder)
        project_root = Path(__file__).parent.parent.parent
        
        self.data_path = project_root / 'data' / 'raw' / 'Test O_G_Equipment_Data.xlsx - O&G Equipment Data.csv'
        assert self.data_path.exists(), f"Data file not found at {self.data_path}"

        self.processed_data_path = Path(project_root / 'data' / 'processed' / 'df_fpso_failure_monitoring_processed.csv')
        self.processed_data_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

        # Model output path
        self.model_output_path = project_root / 'models'
        self.model_output_path.mkdir(parents=True, exist_ok=True)

        # Reports output path
        self.reports_path = project_root / 'reports'
        self.reports_path.mkdir(parents=True, exist_ok=True)

        # Column definitions
        self.sensor_columns = ['Temperature','Pressure', 'VibrationX', 'VibrationY','VibrationZ', 'Frequency']
        self.categorical_columns = ['Preset_1', 'Preset_2']
        self.target_column = 'Fail'
        
        # Analysis parameters
        self.rolling_windows = [3, 5]
        self.test_size = 0.2
        self.random_state = 42
        self.cv_folds = 5
        
        # High-risk preset combinations (from EDA findings)
        self.high_risk_presets = [
            (1, 5),  # Preset_1=1, Preset_2=5
            (3, 5),  # Preset_1=3, Preset_2=5
            (1, 2),  # Preset_1=1, Preset_2=2
        ]