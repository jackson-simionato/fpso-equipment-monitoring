import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, recall_score, precision_score, f1_score, accuracy_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.base import TransformerMixin

class ModelPipeline:
    """
    A class for machine learning pipeline operations including model training,
    evaluation, and feature importance analysis.
    """
    
    def __init__(self, random_state: int = 42, **kwargs):
        """
        Initialize the ModelPipeline.
        
        Args:
            random_state (int): Random state for reproducibility
            **kwargs: Optional model hyperparameters
        """
        self.random_state = random_state
        self.models = self._get_default_models(**kwargs)
        self.best_pipeline = None
        self.feature_importance = None
        
    def _get_default_models(self, **kwargs) -> Dict[str, Any]:
        """Get default model configurations, allowing overrides via kwargs.
        
        Args:
            **kwargs: Optional model hyperparameters to override defaults.

        Returns:
            Dict[str, Any]: Dictionary of model names and their configurations.
        """
        rf_params = {
            'random_state': self.random_state,
            'n_estimators': 150,
            'min_samples_leaf': 5,
            'max_features': 'sqrt',
        }
        rf_params.update(kwargs.get('random_forest', {}))

        gb_params = {
            'random_state': self.random_state,
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 3,
            'subsample': 0.8,
            'min_samples_leaf': 3
        }
        gb_params.update(kwargs.get('gradient_boosting', {}))

        xgb_params = {
            'random_state': self.random_state,
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 4,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
        xgb_params.update(kwargs.get('xgboost', {}))

        models = {
            'Random Forest': RandomForestClassifier(**rf_params),
            'Gradient Boosting': GradientBoostingClassifier(**gb_params),
            'XGBoost': XGBClassifier(**xgb_params)
        }
        
        return models
    
    def create_train_test_split(self, df: pd.DataFrame, target_col: str = 'Fail', 
                               test_size: float = 0.2, stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits the DataFrame into train and test sets.

        Args:
            df (pd.DataFrame): The DataFrame.
            target_col (str): Name of the target column.
            test_size (float): Proportion of test set.
            stratify (bool): Whether to stratify by target.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_test, y_train, y_test
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]
        stratify_y = y if stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=stratify_y
        )
        return X_train, X_test, y_train, y_test

    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                      scaler: Optional[TransformerMixin] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scales train and test features using the provided scaler.

        Args:
            X_train: Training features.
            X_test: Test features.
            scaler: Scaler instance (if None, uses StandardScaler).

        Returns:
            Tuple[np.ndarray, np.ndarray]: X_train_scaled, X_test_scaled
        """
        if scaler is None:
            scaler = StandardScaler()
        
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled

    def create_pipeline(self, model: Any, use_smote: bool = True, 
                       use_scaling: bool = True) -> Pipeline:
        """
        Create a machine learning pipeline with optional SMOTE and scaling.
        
        Args:
            model: The machine learning model
            use_smote (bool): Whether to include SMOTE for oversampling
            use_scaling (bool): Whether to include StandardScaler
            
        Returns:
            Pipeline: Sklearn/imblearn pipeline
        """
        steps = []
        
        # Add scaling step if requested
        if use_scaling:
            steps.append(('scaler', StandardScaler()))
        
        # Add SMOTE oversampling if requested
        if use_smote:
            steps.append(('smote', SMOTE(random_state=self.random_state)))
            
        steps.append(('model', model))
        
        return Pipeline(steps)

    def train_single_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series, 
                          use_smote: bool = True) -> Dict[str, Any]:
        """
        Train a single model and return performance metrics.
        
        Args:
            model_name (str): Name of the model to train
            X_train, y_train: Training data
            X_test, y_test: Test data
            use_smote (bool): Whether to use SMOTE
            
        Returns:
            Dict[str, Any]: Performance metrics and trained pipeline
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available. Choose from: {list(self.models.keys())}")
        
        # Create pipeline
        pipeline = self.create_pipeline(self.models[model_name], use_smote=use_smote)
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.ravel().shape[0] == 4 else (0, 0, 0, 0)
        
        # Calculate miss rate and false alarm rate
        miss_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Generate classification report
        results = {
            'pipeline': pipeline,
            'predictions': y_pred,
            'confusion_matrix': cm,
            'miss_rate': miss_rate,
            'false_alarm_rate': false_alarm_rate,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'recall': recall_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'accuracy': accuracy_score(y_test, y_pred)
        }
        
        return results

    def compare_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_test: pd.DataFrame, y_test: pd.Series,
                      cv_folds: int = 5, use_smote: bool = True) -> Dict[str, Any]:
        """
        Compare multiple models using cross-validation and test set evaluation.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            cv_folds (int): Number of cross-validation folds
            use_smote (bool): Whether to use SMOTE
            
        Returns:
            Dict[str, Any]: Comparison results for all models
        """

        # Define scoring metrics
        scoring = {
            'recall': make_scorer(recall_score),
            'precision': make_scorer(precision_score),
            'f1': make_scorer(f1_score),
            'accuracy': make_scorer(accuracy_score)
        }
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        results = {}
        
        # Iterate over each model
        for model_name, model in self.models.items():
            print(f"Testing {model_name}...")
            
            # Create pipeline
            pipeline = self.create_pipeline(model, use_smote=use_smote)
            
            # Cross-validation
            cv_scores = cross_validate(
                pipeline, X_train, y_train, 
                cv=cv, scoring=scoring, 
                return_train_score=False
            )
            
            # Calculate CV statistics
            cv_results = {}
            for metric in ['test_recall', 'test_precision', 'test_f1', 'test_accuracy']:
                cv_results[metric.replace('test_', '')] = {
                    'mean': cv_scores[metric].mean(),
                    'std': cv_scores[metric].std()
                }
            
            # Train on full training set and evaluate on test set
            test_results = self.train_single_model(model_name, X_train, y_train, X_test, y_test, use_smote)
            
            results[model_name] = {
                'cv_results': cv_results,
                'test_results': test_results
            }
        
        return results

    def get_feature_importance(self, pipeline: Pipeline, X_train: pd.DataFrame, 
                              y_train: pd.Series, top_n: int = 20) -> pd.DataFrame:
        """
        Extract and return feature importances from a trained Random Forest pipeline.

        Args:
            pipeline: Trained sklearn Pipeline with RandomForestClassifier
            X_train: Training features (DataFrame to preserve column names)
            y_train: Training target
            top_n (int): Number of top features to return

        Returns:
            pd.DataFrame: Feature importance dataframe sorted by importance
        """
        # Ensure pipeline is fitted
        if not hasattr(pipeline, 'named_steps'):
            pipeline.fit(X_train, y_train)

        # Access the model (try different common naming conventions)
        model = pipeline.named_steps.get('model') or pipeline.named_steps.get('classifier')
        
        if model is None or not hasattr(model, 'feature_importances_'):
            raise ValueError("Pipeline does not contain a model with feature_importances_ attribute")

        # Get feature names
        if hasattr(X_train, 'columns'):
            feature_names = X_train.columns
        else:
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values(by='importance', ascending=False).reset_index(drop=True)

        self.feature_importance = importance_df
        return importance_df.head(top_n)

    def find_best_model(self, comparison_results: Dict[str, Any], 
                       metric: str = 'recall') -> Tuple[str, Any]:
        """
        Find the best performing model based on a specific metric.
        
        Args:
            comparison_results: Results from compare_models()
            metric (str): Metric to optimize for ('recall', 'precision', 'f1', 'accuracy')
            
        Returns:
            Tuple[str, Any]: Best model name and its results
        """
        best_score = -1
        best_model = None
        best_results = None
        
        # Iterate through results to find the best model
        for model_name, results in comparison_results.items():
            score = results['test_results'][metric]
            if score > best_score:
                best_score = score
                best_model = model_name
                best_results = results
        
        return best_model, best_results

    def generate_model_report(self, comparison_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive model comparison report.
        
        Args:
            comparison_results: Results from compare_models()
            
        Returns:
            str: Formatted model comparison report
        """
        report = []
        report.append("=" * 60)
        report.append("MODEL COMPARISON REPORT")
        report.append("=" * 60)
        
        # Test set results summary
        report.append("\nTEST SET PERFORMANCE:")
        report.append("-" * 40)
        for model_name, results in comparison_results.items():
            test_res = results['test_results']
            report.append(f"\n{model_name}:")
            report.append(f"  Recall: {test_res['recall']:.3f}")
            report.append(f"  Precision: {test_res['precision']:.3f}")
            report.append(f"  F1-Score: {test_res['f1']:.3f}")
            report.append(f"  Accuracy: {test_res['accuracy']:.3f}")
            report.append(f"  Miss Rate: {test_res['miss_rate']:.1%}")
            report.append(f"  False Alarm Rate: {test_res['false_alarm_rate']:.1%}")
        
        # Cross-validation summary
        report.append("\n\nCROSS-VALIDATION SUMMARY:")
        report.append("-" * 40)
        for model_name, results in comparison_results.items():
            cv_res = results['cv_results']
            report.append(f"\n{model_name} (mean ± std):")
            report.append(f"  Recall: {cv_res['recall']['mean']:.3f} ± {cv_res['recall']['std']:.3f}")
            report.append(f"  Precision: {cv_res['precision']['mean']:.3f} ± {cv_res['precision']['std']:.3f}")
            report.append(f"  F1-Score: {cv_res['f1']['mean']:.3f} ± {cv_res['f1']['std']:.3f}")
        
        # Best performers
        report.append("\n\nBEST PERFORMERS:")
        report.append("-" * 40)
        for metric in ['recall', 'precision', 'f1', 'accuracy']:
            best_model, _ = self.find_best_model(comparison_results, metric)
            best_score = comparison_results[best_model]['test_results'][metric]
            report.append(f"Best {metric.title()}: {best_model} ({best_score:.3f})")
        
        report.append("\n" + "=" * 60)
        return "\n".join(report)

    def save_best_pipeline(self, comparison_results: Dict[str, Any], 
                          metric: str = 'recall') -> Pipeline:
        """
        Save the best performing pipeline for future use.
        
        Args:
            comparison_results: Results from compare_models()
            metric (str): Metric to optimize for
            
        Returns:
            Pipeline: Best performing pipeline
        """
        # Find the best model based on the specified metric
        best_model_name, best_results = self.find_best_model(comparison_results, metric)
        self.best_pipeline = best_results['test_results']['pipeline']
        
        print(f"Best model ({best_model_name}) saved based on {metric}: {best_results['test_results'][metric]:.3f}")
        return self.best_pipeline
