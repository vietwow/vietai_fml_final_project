"""
Data Preprocessing & Feature Engineering Module
Các hàm xử lý dữ liệu và tạo đặc trưng mới.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from pathlib import Path

from .config import (
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, QUALITY_MAPPING,
    TARGET_COLUMN, MODELS_DIR
)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for feature engineering."""
    
    def __init__(self):
        self.feature_names_out_ = None
        
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations."""
        X = X.copy()
        
        # 1. Total Square Footage
        if 'TotalBsmtSF' in X.columns and '1stFlrSF' in X.columns and '2ndFlrSF' in X.columns:
            X['TotalSF'] = X['TotalBsmtSF'].fillna(0) + X['1stFlrSF'] + X['2ndFlrSF']
        
        # 2. Total Bathrooms
        bath_cols = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
        if all(col in X.columns for col in bath_cols[:2]):
            X['TotalBath'] = X.get('FullBath', 0) + 0.5 * X.get('HalfBath', 0) + \
                             X.get('BsmtFullBath', 0).fillna(0) + 0.5 * X.get('BsmtHalfBath', 0).fillna(0)
        
        # 3. Total Porch Area
        porch_cols = ['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
        existing_porch = [col for col in porch_cols if col in X.columns]
        if existing_porch:
            X['TotalPorchSF'] = X[existing_porch].fillna(0).sum(axis=1)
        
        # 4. House Age
        if 'YearBuilt' in X.columns and 'YrSold' in X.columns:
            X['HouseAge'] = X['YrSold'] - X['YearBuilt']
        elif 'YearBuilt' in X.columns:
            X['HouseAge'] = 2024 - X['YearBuilt']
        
        # 5. Remodel Age
        if 'YearRemodAdd' in X.columns and 'YrSold' in X.columns:
            X['RemodAge'] = X['YrSold'] - X['YearRemodAdd']
        elif 'YearRemodAdd' in X.columns:
            X['RemodAge'] = 2024 - X['YearRemodAdd']
        
        # 6. Was Remodeled
        if 'YearBuilt' in X.columns and 'YearRemodAdd' in X.columns:
            X['WasRemodeled'] = (X['YearRemodAdd'] != X['YearBuilt']).astype(int)
        
        # 7. Has features flags
        if 'PoolArea' in X.columns:
            X['HasPool'] = (X['PoolArea'] > 0).astype(int)
        if 'GarageArea' in X.columns:
            X['HasGarage'] = (X['GarageArea'].fillna(0) > 0).astype(int)
        if 'Fireplaces' in X.columns:
            X['HasFireplace'] = (X['Fireplaces'] > 0).astype(int)
        if 'TotalBsmtSF' in X.columns:
            X['HasBasement'] = (X['TotalBsmtSF'].fillna(0) > 0).astype(int)
        if '2ndFlrSF' in X.columns:
            X['Has2ndFloor'] = (X['2ndFlrSF'] > 0).astype(int)
        
        # 8. Quality interactions
        if 'OverallQual' in X.columns and 'GrLivArea' in X.columns:
            X['QualityArea'] = X['OverallQual'] * X['GrLivArea']
        
        # 9. Neighborhood quality proxy
        if 'OverallQual' in X.columns and 'OverallCond' in X.columns:
            X['OverallScore'] = X['OverallQual'] * X['OverallCond']
        
        # 10. Garage car per area efficiency
        if 'GarageCars' in X.columns and 'GarageArea' in X.columns:
            X['GarageEfficiency'] = X['GarageCars'] / (X['GarageArea'].replace(0, np.nan) + 1)
        
        self.feature_names_out_ = X.columns.tolist()
        return X
    
    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


class QualityEncoder(BaseEstimator, TransformerMixin):
    """Encode quality-related categorical variables to ordinal."""
    
    QUALITY_COLS = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 
                    'HeatingQC', 'KitchenQual', 'FireplaceQu', 
                    'GarageQual', 'GarageCond', 'PoolQC']
    
    def __init__(self):
        self.quality_mapping = QUALITY_MAPPING
        
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.QUALITY_COLS:
            if col in X.columns:
                X[col] = X[col].map(self.quality_mapping).fillna(0)
        return X


class DataPreprocessor:
    """Complete data preprocessing pipeline."""
    
    def __init__(
        self,
        numerical_strategy: str = 'median',
        categorical_strategy: str = 'most_frequent',
        scaling_method: str = 'standard',
        handle_unknown: str = 'ignore'
    ):
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
        self.scaling_method = scaling_method
        self.handle_unknown = handle_unknown
        
        self.feature_engineer = FeatureEngineer()
        self.quality_encoder = QualityEncoder()
        self.preprocessor = None
        self.feature_names = None
        self.is_fitted = False
        
    def _get_numerical_cols(self, df: pd.DataFrame) -> List[str]:
        """Get list of numerical columns present in dataframe."""
        return [col for col in df.select_dtypes(include=[np.number]).columns 
                if col not in [TARGET_COLUMN, 'Id']]
    
    def _get_categorical_cols(self, df: pd.DataFrame) -> List[str]:
        """Get list of categorical columns present in dataframe."""
        return df.select_dtypes(include=['object']).columns.tolist()
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'DataPreprocessor':
        """Fit the preprocessor on training data."""
        # Apply feature engineering first
        X_fe = self.feature_engineer.fit_transform(X)
        X_fe = self.quality_encoder.fit_transform(X_fe)
        
        # Get column types after feature engineering
        numerical_cols = self._get_numerical_cols(X_fe)
        categorical_cols = self._get_categorical_cols(X_fe)
        
        # Build preprocessing pipelines
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy=self.numerical_strategy)),
            ('scaler', StandardScaler() if self.scaling_method == 'standard' else MinMaxScaler())
        ])
        
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy=self.categorical_strategy, fill_value='Missing')),
            ('encoder', OneHotEncoder(handle_unknown=self.handle_unknown, sparse_output=False))
        ])
        
        # Combine pipelines
        self.preprocessor = ColumnTransformer([
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ], remainder='drop')
        
        self.preprocessor.fit(X_fe)
        
        # Store feature names
        num_features = numerical_cols
        cat_features = []
        if categorical_cols:
            cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['encoder']
            cat_features = cat_encoder.get_feature_names_out(categorical_cols).tolist()
        
        self.feature_names = num_features + cat_features
        self.is_fitted = True
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted preprocessor."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        X_fe = self.feature_engineer.transform(X)
        X_fe = self.quality_encoder.transform(X_fe)
        return self.preprocessor.transform(X_fe)
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> np.ndarray:
        """Fit and transform data."""
        self.fit(X, y)
        return self.transform(X)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names after preprocessing."""
        return self.feature_names
    
    def save(self, filepath: str = None):
        """Save preprocessor to disk."""
        if filepath is None:
            filepath = MODELS_DIR / 'preprocessor.joblib'
        joblib.dump(self, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str = None) -> 'DataPreprocessor':
        """Load preprocessor from disk."""
        if filepath is None:
            filepath = MODELS_DIR / 'preprocessor.joblib'
        return joblib.load(filepath)


def handle_missing_values(df: pd.DataFrame, strategy: Dict[str, str] = None) -> pd.DataFrame:
    """
    Handle missing values with column-specific strategies.
    Consistent with notebooks/02_Training.ipynb handle_missing function.
    
    Args:
        df: DataFrame with missing values
        strategy: Dict mapping column names to strategies ('mean', 'median', 'mode', 'zero', 'drop')
        
    Returns:
        DataFrame with handled missing values
    """
    df = df.copy()
    
    # Columns where NA means 'None' (no feature present) - consistent with notebook
    NONE_FILL_COLS = [
        'Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu',
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'
    ]
    
    # Default strategies based on data type
    if strategy is None:
        strategy = {}
    
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
            
        if col in strategy:
            strat = strategy[col]
        elif col in NONE_FILL_COLS:
            strat = 'None'
        elif df[col].dtype in ['object']:
            strat = 'mode'
        elif col in ['LotFrontage', 'GarageYrBlt']:
            strat = 'median'
        elif col in ['MasVnrArea'] or ('Bsmt' in col and df[col].dtype != 'object'):
            strat = 'zero'
        else:
            strat = 'median'
        
        if strat == 'mean':
            df[col] = df[col].fillna(df[col].mean())
        elif strat == 'median':
            df[col] = df[col].fillna(df[col].median())
        elif strat == 'mode':
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown')
        elif strat == 'zero':
            df[col] = df[col].fillna(0)
        elif strat == 'None':
            df[col] = df[col].fillna('None')
        elif strat == 'drop':
            df = df.dropna(subset=[col])
    
    return df


# =============================================================================
# Utility Functions (available for custom preprocessing pipelines)
# =============================================================================

def encode_categorical(
    df: pd.DataFrame, 
    columns: List[str], 
    method: str = 'onehot',
    label_encoders: Dict[str, LabelEncoder] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Encode categorical columns.
    Note: DataPreprocessor class uses sklearn's OneHotEncoder internally.
    This function is available for custom preprocessing needs.
    
    Args:
        df: DataFrame to encode
        columns: List of columns to encode
        method: 'onehot', 'label', or 'ordinal'
        label_encoders: Existing encoders for transform (optional)
        
    Returns:
        Tuple of (encoded DataFrame, encoder objects)
    """
    df = df.copy()
    encoders = {} if label_encoders is None else label_encoders
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'label':
            if col in encoders:
                encoder = encoders[col]
                # Handle unseen categories
                df[col] = df[col].apply(lambda x: encoder.transform([x])[0] 
                                        if x in encoder.classes_ else -1)
            else:
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
                encoders[col] = encoder
                
        elif method == 'onehot':
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
    
    return df, encoders


def apply_log_transform(df: pd.DataFrame, columns: List[str], add_constant: float = 1) -> pd.DataFrame:
    """
    Apply log transformation to specified columns.
    Note: Target variable log transform is applied in notebook/API separately.
    This function is available for custom feature transformations.
    
    Args:
        df: DataFrame to transform
        columns: List of columns to transform
        add_constant: Constant to add before log (to handle zeros)
        
    Returns:
        DataFrame with log-transformed columns
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = np.log1p(df[col] + add_constant - 1)
    return df


def remove_outliers(
    df: pd.DataFrame, 
    columns: List[str], 
    method: str = 'iqr',
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    Remove outliers from specified columns.
    
    Args:
        df: DataFrame to clean
        columns: List of columns to check for outliers
        method: 'iqr' or 'zscore'
        threshold: Threshold for outlier detection
        
    Returns:
        DataFrame with outliers removed
    """
    df = df.copy()
    mask = pd.Series([True] * len(df))
    
    for col in columns:
        if col not in df.columns or not np.issubdtype(df[col].dtype, np.number):
            continue
            
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            mask &= (df[col] >= lower) & (df[col] <= upper)
            
        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            mask &= z_scores <= threshold
    
    return df[mask]


def create_polynomial_features(
    df: pd.DataFrame, 
    columns: List[str], 
    degree: int = 2
) -> pd.DataFrame:
    """
    Create polynomial features for specified columns.
    Note: This function is available for advanced feature engineering.
    Not used by default in the current pipeline.
    
    Args:
        df: DataFrame to augment
        columns: List of columns for polynomial features
        degree: Polynomial degree
        
    Returns:
        DataFrame with additional polynomial features
    """
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
        for d in range(2, degree + 1):
            df[f'{col}_pow{d}'] = df[col] ** d
    
    # Create interaction terms for pairs of columns
    if degree >= 2 and len(columns) >= 2:
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                if col1 in df.columns and col2 in df.columns:
                    df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
    
    return df


if __name__ == "__main__":
    # Test preprocessing
    import pandas as pd
    
    # Create sample data
    data = pd.DataFrame({
        'OverallQual': [7, 6, 7, 7, 8],
        'GrLivArea': [1710, 1262, 1786, 1717, 2198],
        'YearBuilt': [2003, 1976, 2001, 1915, 2000],
        'YearRemodAdd': [2003, 1976, 2002, 1970, 2000],
        'TotalBsmtSF': [856, 1262, 920, 756, 1145],
        'FullBath': [2, 2, 2, 1, 2],
        'HalfBath': [1, 0, 1, 0, 1],
        'Neighborhood': ['CollgCr', 'Veenker', 'CollgCr', 'Crawfor', 'NoRidge'],
        'ExterQual': ['Gd', 'TA', 'Gd', 'TA', 'Gd'],
        'KitchenQual': ['Gd', 'TA', 'Gd', 'Gd', 'Gd']
    })
    
    preprocessor = DataPreprocessor()
    X_processed = preprocessor.fit_transform(data)
    
    print(f"Original shape: {data.shape}")
    print(f"Processed shape: {X_processed.shape}")
    print(f"Feature names: {preprocessor.get_feature_names()[:10]}...")

