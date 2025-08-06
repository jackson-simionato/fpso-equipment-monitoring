import pandas as pd
from typing import Optional, Dict
from src.config import ApplicationConfig
from src.services.data_plotter import DataPlotter

class CreatePlotsRoute:
    """
    Route for generating and saving key visualizations for the analysis pipeline.

    This class uses DataPlotter to create and save timeline, boxplot, correlation,
    and feature importance plots.

    Args:
        plotter (DataPlotter, optional): DataPlotter instance.
        config (ApplicationConfig, optional): Application configuration instance.
    """
    def __init__(self, plotter: Optional[DataPlotter], config: Optional[ApplicationConfig]):
        """
        Initializes the CreatePlotsRoute with a DataPlotter and ApplicationConfig.

        Args:
            plotter (DataPlotter, optional): DataPlotter instance.
            config (ApplicationConfig, optional): Application configuration instance.
        """
        self.plotter = plotter or DataPlotter()
        self.config = config or ApplicationConfig()

    def run(self, processed_df: pd.DataFrame, feature_importance: pd.DataFrame) -> Dict[str, str]:
        """
        Generate and save all key visualizations.

        Args:
            processed_df (pd.DataFrame): The processed data for plotting.
            feature_importance (pd.DataFrame): DataFrame of feature importances.

        Returns:
            Dict[str, str]: Dictionary with paths to saved plot images.
        """
        print("\n5. GENERATING KEY VISUALIZATIONS...")
        fig_timeline = self.plotter.plot_failure_timeline(processed_df)
        timeline_path = self.config.reports_path / 'failure_timeline.png'
        fig_timeline.savefig(timeline_path, dpi=300, bbox_inches='tight')
        print(f"Failure timeline saved to: {timeline_path}")

        fig_boxplots = self.plotter.plot_multiple_boxplots(processed_df, self.config.sensor_columns, 'Fail')
        boxplot_path = self.config.reports_path / 'sensor_boxplots_by_failure.png'
        fig_boxplots.savefig(boxplot_path, dpi=300, bbox_inches='tight')
        print(f"Sensor boxplots saved to: {boxplot_path}")

        fig_correlation = self.plotter.plot_correlation_heatmap(processed_df, self.config.sensor_columns)
        correlation_path = self.config.reports_path / 'sensor_correlation_heatmap.png'
        fig_correlation.savefig(correlation_path, dpi=300, bbox_inches='tight')
        print(f"Correlation heatmap saved to: {correlation_path}")

        fig_importance = self.plotter.plot_bar(
            feature_importance.head(15), y='feature', x='importance',
            title='Top 15 Feature Importances'
        )
        importance_path = self.config.reports_path / 'feature_importance_plot.png'
        fig_importance.savefig(importance_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to: {importance_path}")

        return {
            "timeline_path": str(timeline_path),
            "boxplot_path": str(boxplot_path),
            "correlation_path": str(correlation_path),
            "importance_path": str(importance_path)
        }