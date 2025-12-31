"""
FastAPI Backend for House Price Prediction
VietAI - Foundations of Machine Learning Final Project
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Initialize FastAPI app
app = FastAPI(
    title="🏠 House Price Prediction API",
    description="""
    ## VietAI - Foundations of Machine Learning Final Project
    
    API để dự đoán giá nhà dựa trên các đặc điểm của ngôi nhà.
    
    ### Endpoints:
    - `POST /predict` - Dự đoán giá nhà
    - `GET /health` - Kiểm tra trạng thái API
    - `GET /model-info` - Thông tin về mô hình
    - `GET /features` - Danh sách các đặc trưng cần thiết
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
model_artifacts = None
MODEL_PATH = Path(__file__).parent.parent / "models" / "best_model.joblib"


# Pydantic models for request/response
class HouseFeatures(BaseModel):
    """Input features for house price prediction."""
    
    # Required features
    OverallQual: int = Field(..., ge=1, le=10, description="Chất lượng tổng thể (1-10)")
    GrLivArea: float = Field(..., gt=0, description="Diện tích sinh hoạt trên mặt đất (sq ft)")
    YearBuilt: int = Field(..., ge=1800, le=2025, description="Năm xây dựng")
    YearRemodAdd: int = Field(..., ge=1800, le=2025, description="Năm cải tạo")
    FullBath: int = Field(..., ge=0, le=10, description="Số phòng tắm đầy đủ")
    TotRmsAbvGrd: int = Field(..., ge=1, le=30, description="Tổng số phòng trên mặt đất")
    
    # Optional features with defaults
    OverallCond: int = Field(default=5, ge=1, le=10, description="Điều kiện tổng thể (1-10)")
    TotalBsmtSF: float = Field(default=0, ge=0, description="Diện tích tầng hầm (sq ft)")
    GarageCars: int = Field(default=0, ge=0, le=10, description="Sức chứa garage (số xe)")
    GarageArea: float = Field(default=0, ge=0, description="Diện tích garage (sq ft)")
    Fireplaces: int = Field(default=0, ge=0, le=10, description="Số lò sưởi")
    LotArea: float = Field(default=10000, ge=0, description="Diện tích đất (sq ft)")
    BedroomAbvGr: int = Field(default=3, ge=0, le=20, description="Số phòng ngủ trên mặt đất")
    KitchenAbvGr: int = Field(default=1, ge=0, le=10, description="Số nhà bếp trên mặt đất")
    HalfBath: int = Field(default=0, ge=0, le=10, description="Số phòng tắm nửa")
    
    # Area features
    FirstFlrSF: float = Field(default=0, ge=0, alias="1stFlrSF", description="Diện tích tầng 1 (sq ft)")
    SecondFlrSF: float = Field(default=0, ge=0, alias="2ndFlrSF", description="Diện tích tầng 2 (sq ft)")
    
    # Categorical features
    Neighborhood: str = Field(default="NAmes", description="Khu vực")
    BldgType: str = Field(default="1Fam", description="Loại công trình")
    HouseStyle: str = Field(default="1Story", description="Kiểu nhà")
    ExterQual: str = Field(default="TA", description="Chất lượng ngoại thất (Ex/Gd/TA/Fa/Po)")
    KitchenQual: str = Field(default="TA", description="Chất lượng nhà bếp (Ex/Gd/TA/Fa/Po)")
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "OverallQual": 7,
                "GrLivArea": 1500,
                "YearBuilt": 2005,
                "YearRemodAdd": 2005,
                "FullBath": 2,
                "TotRmsAbvGrd": 7,
                "OverallCond": 5,
                "TotalBsmtSF": 1000,
                "GarageCars": 2,
                "GarageArea": 500,
                "Fireplaces": 1,
                "LotArea": 10000,
                "BedroomAbvGr": 3,
                "KitchenAbvGr": 1,
                "HalfBath": 1,
                "1stFlrSF": 1000,
                "2ndFlrSF": 500,
                "Neighborhood": "NAmes",
                "BldgType": "1Fam",
                "HouseStyle": "1Story",
                "ExterQual": "Gd",
                "KitchenQual": "Gd"
            }
        }


class ConfidenceInterval(BaseModel):
    """Confidence interval for predictions."""
    lower: float
    upper: float
    formatted: str


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predicted_price: float = Field(..., description="Giá nhà dự đoán ($)")
    predicted_price_formatted: str = Field(..., description="Giá nhà định dạng")
    confidence_interval: ConfidenceInterval = Field(
        ..., description="Khoảng tin cậy (±15%)"
    )
    model_info: Dict[str, Any] = Field(..., description="Thông tin mô hình")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    model_name: Optional[str] = None


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    model_name: str
    test_r2: float
    test_rmse: float
    n_features: int
    feature_names: List[str]


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global model_artifacts
    
    try:
        if MODEL_PATH.exists():
            model_artifacts = joblib.load(MODEL_PATH)
            print(f"✅ Model loaded: {model_artifacts.get('model_name', 'Unknown')}")
        else:
            print(f"⚠️ Model file not found at {MODEL_PATH}")
            print("Please run the training notebook first to generate the model.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint."""
    return {
        "message": "🏠 House Price Prediction API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Kiểm tra trạng thái API."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_artifacts is not None,
        model_name=model_artifacts.get("model_name") if model_artifacts else None
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Lấy thông tin về mô hình đang sử dụng."""
    if model_artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_name=model_artifacts.get("model_name", "Unknown"),
        test_r2=model_artifacts.get("test_r2", 0.0),
        test_rmse=model_artifacts.get("test_rmse", 0.0),
        n_features=len(model_artifacts.get("feature_names", [])),
        feature_names=model_artifacts.get("feature_names", [])[:20]  # First 20
    )


@app.get("/features", tags=["Model"])
async def get_required_features():
    """Lấy danh sách các đặc trưng cần thiết cho dự đoán."""
    return {
        "required": [
            "OverallQual", "GrLivArea", "YearBuilt", "YearRemodAdd",
            "FullBath", "TotRmsAbvGrd"
        ],
        "optional": [
            "OverallCond", "TotalBsmtSF", "GarageCars", "GarageArea",
            "Fireplaces", "LotArea", "BedroomAbvGr", "KitchenAbvGr",
            "HalfBath", "1stFlrSF", "2ndFlrSF", "Neighborhood",
            "BldgType", "HouseStyle", "ExterQual", "KitchenQual"
        ],
        "quality_values": ["Ex", "Gd", "TA", "Fa", "Po"],
        "neighborhoods": [
            "NAmes", "CollgCr", "OldTown", "Edwards", "Somerst",
            "Gilbert", "NridgHt", "Sawyer", "NWAmes", "SawyerW"
        ]
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_house_price(features: HouseFeatures):
    """
    Dự đoán giá nhà dựa trên các đặc điểm.
    
    - **OverallQual**: Chất lượng tổng thể (1-10)
    - **GrLivArea**: Diện tích sinh hoạt (sq ft)
    - **YearBuilt**: Năm xây dựng
    - **FullBath**: Số phòng tắm đầy đủ
    - **TotRmsAbvGrd**: Tổng số phòng
    
    Trả về giá nhà dự đoán cùng khoảng tin cậy.
    """
    if model_artifacts is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please run training notebook first."
        )
    
    try:
        # Get model and scaler
        model = model_artifacts["model"]
        scaler = model_artifacts["scaler"]
        feature_names = model_artifacts["feature_names"]
        
        print(f"📊 Model expects {len(feature_names)} features")
        
        # Prepare input data
        input_data = prepare_features(features, feature_names)
        
        print(f"📊 Input data shape: {input_data.shape}")
        
        # Scale features
        X_scaled = scaler.transform(input_data)
        
        # Handle NaN/Inf values
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Predict (log scale)
        prediction_log = model.predict(X_scaled)[0]
        
        # Clip prediction to avoid overflow
        prediction_log = np.clip(prediction_log, 0, 20)
        
        # Convert back to original scale
        predicted_price = float(np.expm1(prediction_log))
        
        # Ensure valid price
        predicted_price = max(10000, min(predicted_price, 10000000))
        
        # Calculate confidence interval (±15%)
        lower_bound = predicted_price * 0.85
        upper_bound = predicted_price * 1.15
        
        print(f"✅ Predicted price: ${predicted_price:,.0f}")
        
        return PredictionResponse(
            predicted_price=round(predicted_price, 2),
            predicted_price_formatted=f"${predicted_price:,.0f}",
            confidence_interval=ConfidenceInterval(
                lower=round(lower_bound, 2),
                upper=round(upper_bound, 2),
                formatted=f"${lower_bound:,.0f} - ${upper_bound:,.0f}"
            ),
            model_info={
                "model_name": model_artifacts.get("model_name", "Unknown"),
                "test_r2": model_artifacts.get("test_r2", 0.0)
            }
        )
        
    except Exception as e:
        import traceback
        print(f"❌ Prediction error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


def prepare_features(features: HouseFeatures, feature_names: List[str]) -> pd.DataFrame:
    """
    Prepare input features for prediction.
    Apply feature engineering and encoding.
    Creates a DataFrame with exactly the columns expected by the model.
    """
    # Start with all features as 0
    data = {col: 0 for col in feature_names}
    
    # Current year for age calculations
    current_year = 2024
    
    # Basic features from input
    basic_features = {
        'OverallQual': features.OverallQual,
        'OverallCond': features.OverallCond,
        'GrLivArea': features.GrLivArea,
        'YearBuilt': features.YearBuilt,
        'YearRemodAdd': features.YearRemodAdd,
        'FullBath': features.FullBath,
        'HalfBath': features.HalfBath,
        'TotRmsAbvGrd': features.TotRmsAbvGrd,
        'TotalBsmtSF': features.TotalBsmtSF,
        'GarageCars': features.GarageCars,
        'GarageArea': features.GarageArea,
        'Fireplaces': features.Fireplaces,
        'LotArea': features.LotArea,
        'BedroomAbvGr': features.BedroomAbvGr,
        'KitchenAbvGr': features.KitchenAbvGr,
        '1stFlrSF': features.FirstFlrSF,
        '2ndFlrSF': features.SecondFlrSF,
    }
    
    # Update data with basic features
    for key, value in basic_features.items():
        if key in data:
            data[key] = value
    
    # Feature Engineering
    engineered = {
        'TotalSF': features.TotalBsmtSF + features.FirstFlrSF + features.SecondFlrSF,
        'TotalBath': features.FullBath + 0.5 * features.HalfBath,
        'HouseAge': current_year - features.YearBuilt,
        'RemodAge': current_year - features.YearRemodAdd,
        'WasRemodeled': int(features.YearRemodAdd != features.YearBuilt),
        'HasGarage': int(features.GarageArea > 0),
        'HasFireplace': int(features.Fireplaces > 0),
        'HasBasement': int(features.TotalBsmtSF > 0),
        'Has2ndFloor': int(features.SecondFlrSF > 0),
        'HasPool': 0,
        'QualityArea': features.OverallQual * features.GrLivArea,
        'OverallScore': features.OverallQual * features.OverallCond,
        'TotalPorchSF': 0,
    }
    
    for key, value in engineered.items():
        if key in data:
            data[key] = value
    
    # Quality encoding
    quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
    
    quality_values = {
        'ExterQual': quality_map.get(features.ExterQual, 3),
        'KitchenQual': quality_map.get(features.KitchenQual, 3),
        'ExterCond': 3,
        'BsmtQual': 3 if features.TotalBsmtSF > 0 else 0,
        'BsmtCond': 3 if features.TotalBsmtSF > 0 else 0,
        'HeatingQC': 3,
        'FireplaceQu': 3 if features.Fireplaces > 0 else 0,
        'GarageQual': 3 if features.GarageArea > 0 else 0,
        'GarageCond': 3 if features.GarageArea > 0 else 0,
        'PoolQC': 0,
    }
    
    for key, value in quality_values.items():
        if key in data:
            data[key] = value
    
    # One-hot encoded categorical features
    # Neighborhood
    neighborhood_col = f'Neighborhood_{features.Neighborhood}'
    if neighborhood_col in data:
        data[neighborhood_col] = 1
    
    # Building type
    bldgtype_col = f'BldgType_{features.BldgType}'
    if bldgtype_col in data:
        data[bldgtype_col] = 1
    
    # House style
    housestyle_col = f'HouseStyle_{features.HouseStyle}'
    if housestyle_col in data:
        data[housestyle_col] = 1
    
    # Create DataFrame with exact column order
    df = pd.DataFrame([data])[feature_names]
    
    # Fill any remaining NaN with 0
    df = df.fillna(0)
    
    return df


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

