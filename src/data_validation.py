"""
Data Validation Module
Định nghĩa schema và các hàm kiểm tra dữ liệu đầu vào.
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, ValidationError
from dataclasses import dataclass
from enum import Enum


class DataType(Enum):
    """Enum for data types."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"


@dataclass
class ColumnSchema:
    """Schema definition for a single column."""
    name: str
    dtype: DataType
    nullable: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    
    def validate(self, series: pd.Series) -> Tuple[bool, List[str]]:
        """Validate a pandas Series against this schema."""
        errors = []
        
        # Check for null values
        if not self.nullable and series.isna().any():
            null_count = series.isna().sum()
            errors.append(f"Column '{self.name}' has {null_count} null values but nullable=False")
        
        # Validate numeric columns
        if self.dtype == DataType.NUMERIC:
            non_null = series.dropna()
            if self.min_value is not None and (non_null < self.min_value).any():
                min_found = non_null.min()
                errors.append(f"Column '{self.name}' has values below minimum {self.min_value}: found {min_found}")
            if self.max_value is not None and (non_null > self.max_value).any():
                max_found = non_null.max()
                errors.append(f"Column '{self.name}' has values above maximum {self.max_value}: found {max_found}")
        
        # Validate categorical columns
        if self.dtype == DataType.CATEGORICAL and self.allowed_values is not None:
            non_null = series.dropna()
            invalid_values = set(non_null.unique()) - set(self.allowed_values)
            if invalid_values:
                errors.append(f"Column '{self.name}' has invalid values: {invalid_values}")
        
        return len(errors) == 0, errors


class HousePriceSchema:
    """Complete schema for the House Price dataset."""
    
    def __init__(self):
        self.columns = self._define_columns()
    
    def _define_columns(self) -> Dict[str, ColumnSchema]:
        """Define all column schemas."""
        return {
            # Identification
            'Id': ColumnSchema('Id', DataType.NUMERIC, nullable=False, min_value=1),
            
            # Lot features
            'LotFrontage': ColumnSchema('LotFrontage', DataType.NUMERIC, nullable=True, min_value=0),
            'LotArea': ColumnSchema('LotArea', DataType.NUMERIC, nullable=False, min_value=0),
            
            # Quality ratings
            'OverallQual': ColumnSchema('OverallQual', DataType.NUMERIC, nullable=False, min_value=1, max_value=10),
            'OverallCond': ColumnSchema('OverallCond', DataType.NUMERIC, nullable=False, min_value=1, max_value=10),
            
            # Year features
            'YearBuilt': ColumnSchema('YearBuilt', DataType.NUMERIC, nullable=False, min_value=1800, max_value=2025),
            'YearRemodAdd': ColumnSchema('YearRemodAdd', DataType.NUMERIC, nullable=False, min_value=1800, max_value=2025),
            
            # Area features
            'GrLivArea': ColumnSchema('GrLivArea', DataType.NUMERIC, nullable=False, min_value=0),
            'TotalBsmtSF': ColumnSchema('TotalBsmtSF', DataType.NUMERIC, nullable=True, min_value=0),
            '1stFlrSF': ColumnSchema('1stFlrSF', DataType.NUMERIC, nullable=False, min_value=0),
            '2ndFlrSF': ColumnSchema('2ndFlrSF', DataType.NUMERIC, nullable=False, min_value=0),
            
            # Room counts
            'BedroomAbvGr': ColumnSchema('BedroomAbvGr', DataType.NUMERIC, nullable=False, min_value=0, max_value=20),
            'KitchenAbvGr': ColumnSchema('KitchenAbvGr', DataType.NUMERIC, nullable=False, min_value=0, max_value=10),
            'TotRmsAbvGrd': ColumnSchema('TotRmsAbvGrd', DataType.NUMERIC, nullable=False, min_value=0, max_value=30),
            'FullBath': ColumnSchema('FullBath', DataType.NUMERIC, nullable=False, min_value=0, max_value=10),
            'HalfBath': ColumnSchema('HalfBath', DataType.NUMERIC, nullable=False, min_value=0, max_value=10),
            
            # Garage features
            'GarageCars': ColumnSchema('GarageCars', DataType.NUMERIC, nullable=True, min_value=0, max_value=10),
            'GarageArea': ColumnSchema('GarageArea', DataType.NUMERIC, nullable=True, min_value=0),
            
            # Categorical features
            'MSZoning': ColumnSchema('MSZoning', DataType.CATEGORICAL, nullable=True,
                                    allowed_values=['A', 'C', 'FV', 'I', 'RH', 'RL', 'RP', 'RM', 'C (all)']),
            'Street': ColumnSchema('Street', DataType.CATEGORICAL, nullable=False,
                                  allowed_values=['Grvl', 'Pave']),
            'LotShape': ColumnSchema('LotShape', DataType.CATEGORICAL, nullable=False,
                                    allowed_values=['Reg', 'IR1', 'IR2', 'IR3']),
            'Neighborhood': ColumnSchema('Neighborhood', DataType.CATEGORICAL, nullable=False),
            'BldgType': ColumnSchema('BldgType', DataType.CATEGORICAL, nullable=False,
                                    allowed_values=['1Fam', '2fmCon', 'Duplex', 'TwnhsE', 'Twnhs']),
            'HouseStyle': ColumnSchema('HouseStyle', DataType.CATEGORICAL, nullable=False),
            'ExterQual': ColumnSchema('ExterQual', DataType.CATEGORICAL, nullable=False,
                                     allowed_values=['Ex', 'Gd', 'TA', 'Fa', 'Po']),
            'KitchenQual': ColumnSchema('KitchenQual', DataType.CATEGORICAL, nullable=True,
                                       allowed_values=['Ex', 'Gd', 'TA', 'Fa', 'Po']),
            'SaleCondition': ColumnSchema('SaleCondition', DataType.CATEGORICAL, nullable=False,
                                         allowed_values=['Normal', 'Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial']),
            
            # Target
            'SalePrice': ColumnSchema('SalePrice', DataType.NUMERIC, nullable=True, min_value=0),
        }
    
    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Validate a DataFrame against the schema.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, errors_dict)
        """
        all_errors = {}
        is_valid = True
        
        for col_name, schema in self.columns.items():
            if col_name in df.columns:
                col_valid, errors = schema.validate(df[col_name])
                if not col_valid:
                    is_valid = False
                    all_errors[col_name] = errors
        
        return is_valid, all_errors
    
    def get_required_columns(self) -> List[str]:
        """Get list of required (non-nullable) columns."""
        return [name for name, schema in self.columns.items() if not schema.nullable]
    
    def check_missing_columns(self, df: pd.DataFrame) -> List[str]:
        """Check for missing required columns."""
        required = self.get_required_columns()
        return [col for col in required if col not in df.columns]


class PredictionInput(BaseModel):
    """
    Pydantic model for prediction input validation (reference implementation).
    Note: API uses its own HouseFeatures model in api/main.py with more fields.
    This class is kept for validation testing and as a minimal input example.
    """
    
    OverallQual: int = Field(..., ge=1, le=10, description="Overall material and finish quality")
    GrLivArea: float = Field(..., gt=0, description="Above grade living area square feet")
    GarageCars: int = Field(default=0, ge=0, le=10, description="Size of garage in car capacity")
    GarageArea: float = Field(default=0, ge=0, description="Size of garage in square feet")
    TotalBsmtSF: float = Field(default=0, ge=0, description="Total square feet of basement area")
    FullBath: int = Field(..., ge=0, le=10, description="Full bathrooms above grade")
    YearBuilt: int = Field(..., ge=1800, le=2025, description="Original construction date")
    YearRemodAdd: int = Field(..., ge=1800, le=2025, description="Remodel date")
    TotRmsAbvGrd: int = Field(..., ge=1, le=30, description="Total rooms above grade")
    Fireplaces: int = Field(default=0, ge=0, le=10, description="Number of fireplaces")
    
    # Additional common features
    LotArea: float = Field(default=10000, ge=0, description="Lot size in square feet")
    BedroomAbvGr: int = Field(default=3, ge=0, le=20, description="Bedrooms above grade")
    KitchenAbvGr: int = Field(default=1, ge=0, le=10, description="Kitchens above grade")
    
    # Categorical features (as strings)
    Neighborhood: str = Field(default="NAmes", description="Physical locations within Ames city")
    BldgType: str = Field(default="1Fam", description="Type of dwelling")
    HouseStyle: str = Field(default="1Story", description="Style of dwelling")
    ExterQual: str = Field(default="TA", description="Exterior material quality")
    KitchenQual: str = Field(default="TA", description="Kitchen quality")
    
    class Config:
        json_schema_extra = {
            "example": {
                "OverallQual": 7,
                "GrLivArea": 1500,
                "GarageCars": 2,
                "GarageArea": 500,
                "TotalBsmtSF": 1000,
                "FullBath": 2,
                "YearBuilt": 2005,
                "YearRemodAdd": 2005,
                "TotRmsAbvGrd": 7,
                "Fireplaces": 1,
                "LotArea": 10000,
                "BedroomAbvGr": 3,
                "KitchenAbvGr": 1,
                "Neighborhood": "NAmes",
                "BldgType": "1Fam",
                "HouseStyle": "1Story",
                "ExterQual": "Gd",
                "KitchenQual": "Gd"
            }
        }


def validate_prediction_input(data: Dict) -> Tuple[bool, Optional[PredictionInput], Optional[str]]:
    """
    Validate prediction input data.
    
    Args:
        data: Dictionary containing prediction features
        
    Returns:
        Tuple of (is_valid, validated_input, error_message)
    """
    try:
        validated = PredictionInput(**data)
        return True, validated, None
    except ValidationError as e:
        return False, None, str(e)


def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform comprehensive data quality checks.
    
    Args:
        df: DataFrame to check
        
    Returns:
        Dictionary with data quality metrics
    """
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': {},
        'missing_percentage': {},
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.astype(str).to_dict(),
        'numeric_stats': {},
        'categorical_stats': {}
    }
    
    # Missing values analysis
    missing = df.isnull().sum()
    quality_report['missing_values'] = missing[missing > 0].to_dict()
    quality_report['missing_percentage'] = (missing[missing > 0] / len(df) * 100).round(2).to_dict()
    
    # Numeric columns statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        quality_report['numeric_stats'][col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'median': df[col].median()
        }
    
    # Categorical columns statistics
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        quality_report['categorical_stats'][col] = {
            'unique_values': df[col].nunique(),
            'top_values': df[col].value_counts().head(5).to_dict()
        }
    
    return quality_report


def detect_outliers(df: pd.DataFrame, columns: List[str], method: str = 'iqr') -> Dict[str, pd.Series]:
    """
    Detect outliers in specified columns.
    
    Args:
        df: DataFrame to analyze
        columns: List of column names to check
        method: 'iqr' (Interquartile Range) or 'zscore'
        
    Returns:
        Dictionary mapping column names to boolean Series indicating outliers
    """
    outliers = {}
    
    for col in columns:
        if col not in df.columns or not np.issubdtype(df[col].dtype, np.number):
            continue
            
        series = df[col].dropna()
        
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
            
        elif method == 'zscore':
            mean = series.mean()
            std = series.std()
            z_scores = np.abs((df[col] - mean) / std)
            outliers[col] = z_scores > 3
    
    return outliers


if __name__ == "__main__":
    # Test validation
    schema = HousePriceSchema()
    print("Required columns:", schema.get_required_columns())
    
    # Test prediction input
    test_input = {
        "OverallQual": 7,
        "GrLivArea": 1500,
        "FullBath": 2,
        "YearBuilt": 2005,
        "YearRemodAdd": 2005,
        "TotRmsAbvGrd": 7
    }
    is_valid, validated, error = validate_prediction_input(test_input)
    print(f"Input valid: {is_valid}")
    if validated:
        print(f"Validated input: {validated.model_dump()}")

