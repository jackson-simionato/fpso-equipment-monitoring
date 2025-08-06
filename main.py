"""
Main pipeline script for FPSO Equipment Monitoring Analysis.
"""

import pandas as pd
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from src.config import ApplicationConfig
from src.services.data_processor import DataProcessor
from src.services.feature_engineer import FeatureEngineer
from src.routes.create_features import FeatureProcessingRoutes
from src.routes.run_models import RunModelsRoutes  # <-- Add this import
from src.routes.pre_process_data import PreProcessDataRoute
from src.routes.analyze_statistics import AnalyzeStatisticsRoute
from src.services.statistical_analyzer import StatisticalAnalyzer
from src.services.failure_analyzer import FailureAnalyzer
from src.services.data_plotter import DataPlotter
from src.services.model_pipeline import ModelPipeline
from src.routes.create_plots import CreatePlotsRoute

def main():
    print("=" * 60)
    print("FPSO EQUIPMENT MONITORING - COMPLETE ANALYSIS PIPELINE")
    print("=" * 60)
    
    # Initialize configuration and services
    config = ApplicationConfig()
    data_processor = DataProcessor()
    feature_engineer = FeatureEngineer()
    data_routes = PreProcessDataRoute(data_processor, feature_engineer)
    stat_analyzer = StatisticalAnalyzer()
    failure_analyzer = FailureAnalyzer()
    plotter = DataPlotter()
    model_pipeline = ModelPipeline(random_state=config.random_state)
    modeling_routes = RunModelsRoutes(model_pipeline, config)  # <-- Instantiate

    # 1. DATA LOADING AND PROCESSING
    pre_process_route = PreProcessDataRoute(config)
    pre_process_info = pre_process_route.run()
    processed_df = pre_process_info["processed_df"]

    # 2. STATISTICAL & FAILURE ANALYSIS (via new route)
    analyze_route = AnalyzeStatisticsRoute(stat_analyzer, failure_analyzer, config)
    stats_info = analyze_route.run(processed_df)
    summary_report = stats_info["summary_report"]
    mean_diff = stats_info["mean_diff"]
    failure_summary = stats_info["failure_summary"]
    failure_duration_stats = stats_info["failure_duration_stats"]

    print(summary_report)
    # Example: mean difference by target
    print("\nTop sensor difference by target:", mean_diff.idxmax())
    
    # 3. FAILURE ANALYSIS
    print("\n3. FAILURE-SPECIFIC ANALYSIS...")
    print(f"\nFailure Events Summary:\n{failure_summary.head()}")
    print(f"\nFailure Duration Analysis:\n{failure_duration_stats}")

    # 4. MACHINE LEARNING PIPELINE
    print("\n4. MACHINE LEARNING ANALYSIS...")
    processed_df = pre_process_route.run()['processed_df']
    modeling_results = modeling_routes.run(processed_df)
    comparison_results = modeling_results["comparison_results"]
    model_report = modeling_results["model_report"]
    best_pipeline = modeling_results["best_pipeline"]
    feature_importance = modeling_results["feature_importance"]

    # 5. VISUALIZATION SUMMARY
    create_plots_route = CreatePlotsRoute(plotter, config)
    plot_paths = create_plots_route.run(processed_df, feature_importance)
    # Optionally print or use plot_paths if needed
    
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nGenerated outputs:")
    print(f"- Processed data: {config.processed_data_path}")
    print(f"- Reports: {config.reports_path}")
    print(f"- Model pipeline: Saved in memory (best_pipeline)")
    print("\nNext steps:")
    print("- Review generated reports for insights")
    print("- Use saved model pipeline for predictions")
    print("- Implement monitoring based on feature importance")

if __name__ == "__main__":
    main()