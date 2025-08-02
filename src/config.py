from pathlib import Path

class ApplicationConfig:
    """
    Configuration class for the application.
    """
    def __init__(self):
        # Get the root project directory (parent of src folder)
        project_root = Path(__file__).parent.parent
        self.data_path = project_root / 'data' / 'raw' / 'Test O_G_Equipment_Data.xlsx - O&G Equipment Data.csv'