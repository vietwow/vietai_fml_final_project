# House Price Prediction ML Project
# VietAI - Foundations of Machine Learning

__version__ = "1.0.0"

# Export main modules
from .config import *
from .data_validation import HousePriceSchema, check_data_quality, detect_outliers
from .preprocessing import (
    FeatureEngineer, QualityEncoder, DataPreprocessor,
    handle_missing_values, remove_outliers
)
from .model_training import ModelTrainer, train_pipeline, evaluate_model

# Deep Learning (optional - requires PyTorch)
try:
    from .deep_learning import (
        HousePriceNN, NeuralNetworkTrainer, 
        train_neural_network, check_pytorch_available
    )
except ImportError:
    pass  # PyTorch not installed

