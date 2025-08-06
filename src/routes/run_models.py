import pandas as pd
from typing import Optional
from src.services.model_pipeline import ModelPipeline
from src.config import ApplicationConfig

class RunModelsRoutes:
    def __init__(self, model_pipeline: Optional[ModelPipeline], config: Optional[ApplicationConfig]):
        """
        Initialize the RunModelsRoutes.

        Args:
            model_pipeline (ModelPipeline): ModelPipeline instance.
            config (ApplicationConfig): Application configuration instance.
        """
        self.model_pipeline = model_pipeline or ModelPipeline()
        self.config = config or ApplicationConfig()

    def run(self, processed_df: pd.DataFrame) -> dict:
        """
        Complete modeling pipeline for the dataset.

        Args:
            processed_df (pd.DataFrame): Fully processed DataFrame after data_processing
        """
        # Step 1: Create train-test split
        X_train, X_test, y_train, y_test = self.model_pipeline.create_train_test_split(
            processed_df, 
            target_col=self.config.target_column, 
            test_size=self.config.test_size
        )

        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Features: {X_train.shape[1]}")

        
        comparison_results = self.model_pipeline.compare_models(
            X_train, y_train, X_test, y_test, cv_folds=self.config.cv_folds
        )

        model_report = self.model_pipeline.generate_model_report(comparison_results)
        print(model_report)
        model_report_path = self.config.reports_path / 'model_comparison_report.txt'

        with open(model_report_path, 'w') as f:
            f.write(model_report)
        print(f"\nModel report saved to: {model_report_path}")

        best_pipeline = self.model_pipeline.save_best_pipeline(comparison_results, metric='recall')
        
        feature_importance = self.model_pipeline.get_feature_importance(best_pipeline, X_train, y_train, top_n=20)
        print("\nTop 10 Most Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"{idx+1:2d}. {row['feature']:30s} {row['importance']:.4f}")
        feature_importance_path = self.config.reports_path / 'feature_importance.csv'
        feature_importance.to_csv(feature_importance_path, index=False)
        print(f"\nFeature importance saved to: {feature_importance_path}")

        return {
            "comparison_results": comparison_results,
            "model_report": model_report,
            "best_pipeline": best_pipeline,
            "feature_importance": feature_importance
        }