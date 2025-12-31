"""
Model Training Module
Huấn luyện và đánh giá các mô hình Machine Learning.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
import joblib
from pathlib import Path
from datetime import datetime

from .config import (
    RANDOM_STATE, TEST_SIZE, TARGET_COLUMN, MODELS_DIR
)


class ModelTrainer:
    """Class to handle model training and evaluation."""
    
    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def get_base_models(self) -> Dict[str, Any]:
        """Get dictionary of base models to train.
        
        Hyperparameters are consistent with notebooks/02_Training.ipynb
        """
        return {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=10.0, random_state=self.random_state),
            'Lasso': Lasso(alpha=0.001, random_state=self.random_state, max_iter=10000),
            'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=self.random_state, max_iter=10000),
            'RandomForest': RandomForestRegressor(
                n_estimators=100, 
                random_state=self.random_state,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100,
                random_state=self.random_state
            )
        }
    
    def train_model(
        self, 
        model_name: str,
        model: Any,
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Train a single model and return training metrics.
        
        Args:
            model_name: Name of the model
            model: Sklearn model instance
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            
        Returns:
            Dictionary with training results
        """
        print(f"\n{'='*50}")
        print(f"Training {model_name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Training predictions
        y_train_pred = model.predict(X_train)
        
        # Calculate training metrics
        train_metrics = self._calculate_metrics(y_train, y_train_pred)
        
        result = {
            'model': model,
            'train_metrics': train_metrics
        }
        
        # Validation metrics if provided
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            val_metrics = self._calculate_metrics(y_val, y_val_pred)
            result['val_metrics'] = val_metrics
            result['val_predictions'] = y_val_pred
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        result['cv_r2_mean'] = cv_scores.mean()
        result['cv_r2_std'] = cv_scores.std()
        
        self.models[model_name] = model
        self.results[model_name] = result
        
        print(f"  Training R² Score: {train_metrics['r2']:.4f}")
        if 'val_metrics' in result:
            print(f"  Validation R² Score: {result['val_metrics']['r2']:.4f}")
        print(f"  CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        return result
    
    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ) -> Dict[str, Dict]:
        """Train all base models."""
        base_models = self.get_base_models()
        
        for name, model in base_models.items():
            self.train_model(name, model, X_train, y_train, X_val, y_val)
        
        return self.results
    
    def tune_hyperparameters(
        self,
        model_name: str,
        model: Any,
        param_grid: Dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: int = 5,
        scoring: str = 'neg_root_mean_squared_error'
    ) -> Tuple[Any, Dict]:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            model_name: Name of the model
            model: Sklearn model instance
            param_grid: Parameter grid for search
            X_train: Training features
            y_train: Training target
            cv: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Tuple of (best model, best parameters)
        """
        print(f"\nTuning hyperparameters for {model_name}...")
        
        grid_search = GridSearchCV(
            model, 
            param_grid, 
            cv=cv, 
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"  Best Score: {-grid_search.best_score_:.4f}")
        print(f"  Best Parameters: {grid_search.best_params_}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100
        }
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all trained models.
        
        Returns:
            DataFrame with model comparison
        """
        comparison = []
        
        for name, result in self.results.items():
            row = {'Model': name}
            
            # Training metrics
            for metric, value in result['train_metrics'].items():
                row[f'Train_{metric.upper()}'] = value
            
            # Validation metrics
            if 'val_metrics' in result:
                for metric, value in result['val_metrics'].items():
                    row[f'Val_{metric.upper()}'] = value
            
            # CV scores
            row['CV_R2_Mean'] = result['cv_r2_mean']
            row['CV_R2_Std'] = result['cv_r2_std']
            
            comparison.append(row)
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('Val_R2' if 'Val_R2' in df.columns else 'CV_R2_Mean', ascending=False)
        
        return df
    
    def select_best_model(self, metric: str = 'val_r2') -> Tuple[str, Any]:
        """
        Select the best model based on specified metric.
        
        Args:
            metric: Metric to use for selection ('val_r2', 'cv_r2', 'val_rmse')
            
        Returns:
            Tuple of (model name, model instance)
        """
        best_score = -np.inf if 'r2' in metric else np.inf
        best_name = None
        
        for name, result in self.results.items():
            if metric == 'val_r2' and 'val_metrics' in result:
                score = result['val_metrics']['r2']
            elif metric == 'cv_r2':
                score = result['cv_r2_mean']
            elif metric == 'val_rmse' and 'val_metrics' in result:
                score = -result['val_metrics']['rmse']  # Negative for comparison
            else:
                continue
            
            if 'r2' in metric:
                if score > best_score:
                    best_score = score
                    best_name = name
            else:
                if score > best_score:  # Already negated for rmse
                    best_score = score
                    best_name = name
        
        if best_name:
            self.best_model_name = best_name
            self.best_model = self.models[best_name]
            print(f"\nBest Model: {best_name}")
            print(f"Best Score ({metric}): {abs(best_score):.4f}")
        
        return best_name, self.models.get(best_name)
    
    def save_model(self, model_name: str = None, filepath: str = None):
        """
        Save a trained model to disk.
        
        Args:
            model_name: Name of model to save (uses best model if None)
            filepath: Path to save model (auto-generated if None)
        """
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.models.get(model_name)
        
        if model is None:
            raise ValueError(f"Model '{model_name}' not found")
        
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = MODELS_DIR / f"{model_name}_{timestamp}.joblib"
        
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")
        
        return filepath
    
    def save_best_model(self, filepath: str = None):
        """Save the best model."""
        if filepath is None:
            filepath = MODELS_DIR / "best_model.joblib"
        return self.save_model(self.best_model_name, filepath)
    
    @staticmethod
    def load_model(filepath: str) -> Any:
        """Load a model from disk."""
        return joblib.load(filepath)


def train_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> Dict[str, Any]:
    """
    Complete training pipeline.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Proportion of test set
        random_state: Random seed
        
    Returns:
        Dictionary with trained models and results
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Initialize trainer
    trainer = ModelTrainer(random_state=random_state)
    
    # Train all models
    results = trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # Compare models
    comparison = trainer.compare_models()
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(comparison.to_string(index=False))
    
    # Select best model
    best_name, best_model = trainer.select_best_model(metric='val_r2')
    
    return {
        'trainer': trainer,
        'comparison': comparison,
        'best_model_name': best_name,
        'best_model': best_model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    log_transformed: bool = False
) -> Dict[str, float]:
    """
    Evaluate a model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name: Name for display
        log_transformed: If True, convert predictions back to original scale
        
    Returns:
        Dictionary of metrics
    """
    y_pred = model.predict(X_test)
    
    # Handle NaN/Inf in predictions
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
    
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred) * 100
    }
    
    # Calculate metrics on original scale if log transformed
    # Consistent with notebook's evaluate_model function
    if log_transformed:
        y_pred_clipped = np.clip(y_pred, 0, 20)  # log(500M) ≈ 20
        y_test_orig = np.expm1(y_test)
        y_pred_orig = np.expm1(y_pred_clipped)
        metrics['MAE_Original'] = mean_absolute_error(y_test_orig, y_pred_orig)
    
    print(f"\n{model_name} - Test Set Evaluation")
    print("="*50)
    for metric, value in metrics.items():
        if metric == 'MAE_Original':
            print(f"  {metric}: ${value:,.0f}")
        else:
            print(f"  {metric}: {value:.4f}")
    
    return metrics


if __name__ == "__main__":
    # Test with synthetic data
    from sklearn.datasets import make_regression
    
    X, y = make_regression(n_samples=1000, n_features=20, noise=10, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y, name='target')
    
    results = train_pipeline(X, y)
    print("\nTraining complete!")

