# 🏠 House Price Prediction - ML Pipeline

## VietAI - Foundations of Machine Learning Final Project

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

Dự án xây dựng hệ thống Machine Learning hoàn chỉnh để dự đoán giá nhà, bao gồm từ việc xử lý dữ liệu thô, huấn luyện mô hình, cho đến triển khai sản phẩm.

## 📋 Mục Lục

- [Tổng Quan](#-tổng-quan)
- [Cấu Trúc Dự Án](#-cấu-trúc-dự-án)
- [Cài Đặt](#-cài-đặt)
- [Sử Dụng](#-sử-dụng)
- [Pipeline ML](#-pipeline-ml)
- [API Documentation](#-api-documentation)
- [Kết Quả](#-kết-quả)

## 🎯 Tổng Quan

### Mục tiêu
- Áp dụng các thư viện Python (Pandas, NumPy, Scikit-learn) để xử lý và phân tích dữ liệu
- Thực hiện đầy đủ các bước trong một quy trình Machine Learning
- Triển khai mô hình dưới dạng API và giao diện web

### Dataset
[Kaggle House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

- **Số lượng mẫu:** 1,460 (train) + 1,459 (test)
- **Số lượng features:** 79
- **Bài toán:** Regression (Dự đoán giá nhà)

## 📁 Cấu Trúc Dự Án

```
Final project/
├── 📂 data/
│   ├── raw/                    # Dữ liệu gốc từ Kaggle
│   │   ├── train.csv
│   │   └── test.csv
│   └── processed/              # Dữ liệu đã xử lý
│
├── 📂 notebooks/
│   ├── 01_EDA.ipynb           # Exploratory Data Analysis
│   └── 02_Training.ipynb       # Model Training & Evaluation
│
├── 📂 src/
│   ├── __init__.py
│   ├── config.py              # Cấu hình dự án
│   ├── data_validation.py     # Data validation & schema
│   ├── preprocessing.py       # Data preprocessing & feature engineering
│   └── model_training.py      # Model training utilities
│
├── 📂 api/
│   ├── __init__.py
│   └── main.py                # FastAPI application
│
├── 📂 app/
│   ├── __init__.py
│   └── streamlit_app.py       # Streamlit frontend
│
├── 📂 models/
│   ├── best_model.joblib      # Trained model
│   └── scaler.joblib          # Feature scaler
│
├── 📂 reports/
│   └── *.png                  # Visualization outputs
│
├── requirements.txt            # Python dependencies
└── README.md                  # Documentation
```

## 🔧 Cài Đặt

### 1. Clone repository và tạo virtual environment

```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 2. Cài đặt dependencies

```bash
cd "Final project"
pip install -r requirements.txt
```

### 3. Download dữ liệu

Tải dữ liệu từ [Kaggle Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) và đặt vào thư mục `data/raw/`:
- `train.csv`
- `test.csv`

## 🚀 Sử Dụng

### Bước 1: Chạy EDA Notebook

```bash
cd notebooks
jupyter notebook 01_EDA.ipynb
```

### Bước 2: Huấn luyện Model

```bash
jupyter notebook 02_Training.ipynb
```

### Bước 3: Khởi động API

```bash
cd api
uvicorn main:app --reload --port 8000
```

API sẽ khả dụng tại: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Bước 4: Khởi động Streamlit App

```bash
cd app
streamlit run streamlit_app.py
```

Giao diện web sẽ mở tại: http://localhost:8501

## 🔄 Pipeline ML

### Giai đoạn 1: Training Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│  Raw Data   │───▶│ Data Valid.  │───▶│  Preprocessing  │
└─────────────┘    └──────────────┘    └─────────────────┘
                                              │
                                              ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│ Save Model  │◀───│  Evaluation  │◀───│    Training     │
└─────────────┘    └──────────────┘    └─────────────────┘
```

#### 1. Data Validation
- Schema validation (kiểu dữ liệu, khoảng giá trị)
- Missing values analysis
- Outlier detection

#### 2. Data Preprocessing
- Xử lý missing values (median, mode, 'None')
- Encoding categorical variables (One-Hot, Ordinal)
- Feature scaling (StandardScaler)

#### 3. Feature Engineering
- **TotalSF**: Tổng diện tích (Basement + 1st + 2nd floor)
- **TotalBath**: Tổng số phòng tắm
- **HouseAge**: Tuổi nhà
- **QualityArea**: OverallQual × GrLivArea
- **HasPool/HasGarage/HasFireplace**: Binary features

#### 4. Model Training
Các mô hình được huấn luyện:
- ✅ Linear Regression (Required)
- ✅ Ridge Regression
- ✅ Lasso Regression
- ✅ ElasticNet
- ✅ Random Forest
- ✅ Gradient Boosting
- ✅ Neural Network (Bonus)

#### 5. Evaluation Metrics
- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **R² Score**
- **Cross-Validation R²**

### Giai đoạn 2: Serving Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│  User Input │───▶│   FastAPI    │───▶│  Preprocessing  │
└─────────────┘    └──────────────┘    └─────────────────┘
                                              │
                                              ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Display   │◀───│   Response   │◀───│   Prediction    │
└─────────────┘    └──────────────┘    └─────────────────┘
```

## 📚 API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Kiểm tra trạng thái API |
| GET | `/model-info` | Thông tin mô hình |
| GET | `/features` | Danh sách features |
| POST | `/predict` | Dự đoán giá nhà |

### Ví dụ Request

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "OverallQual": 7,
       "GrLivArea": 1500,
       "YearBuilt": 2005,
       "YearRemodAdd": 2005,
       "FullBath": 2,
       "TotRmsAbvGrd": 7,
       "TotalBsmtSF": 1000,
       "GarageCars": 2,
       "GarageArea": 500
     }'
```

### Response

```json
{
    "predicted_price": 185000.00,
    "predicted_price_formatted": "$185,000",
    "confidence_interval": {
        "lower": 157250.00,
        "upper": 212750.00,
        "formatted": "$157,250 - $212,750"
    },
    "model_info": {
        "model_name": "Gradient Boosting",
        "test_r2": 0.91
    }
}
```

## 📊 Kết Quả

### Model Comparison

| Model | Train R² | Test R² | RMSE | MAE ($) |
|-------|----------|---------|------|---------|
| Linear Regression | 0.92 | 0.89 | 0.12 | $16,500 |
| Ridge Regression | 0.92 | 0.90 | 0.11 | $15,800 |
| Lasso Regression | 0.91 | 0.89 | 0.12 | $16,200 |
| Random Forest | 0.97 | 0.88 | 0.13 | $17,100 |
| **Gradient Boosting** | **0.95** | **0.91** | **0.10** | **$14,500** |
| Neural Network (PyTorch) | 0.93 | 0.90 | 0.11 | $15,200 |

### Top Features

1. **OverallQual** - Chất lượng tổng thể (r = 0.79)
2. **GrLivArea** - Diện tích sinh hoạt (r = 0.71)
3. **GarageCars** - Sức chứa garage (r = 0.64)
4. **GarageArea** - Diện tích garage (r = 0.62)
5. **TotalBsmtSF** - Diện tích tầng hầm (r = 0.61)

## 🎨 Screenshots

### Streamlit App
Giao diện web cho phép người dùng nhập thông tin nhà và nhận kết quả dự đoán giá nhà với khoảng tin cậy.

### FastAPI Swagger
API documentation với khả năng test trực tiếp các endpoints.

## 🛠️ Technologies Used

- **Python 3.10+**
- **Pandas & NumPy** - Data processing
- **Scikit-learn** - Machine Learning
- **PyTorch** - Deep Learning (Bonus)
- **FastAPI** - API development
- **Streamlit** - Web interface
- **Matplotlib & Seaborn** - Visualization
- **Pydantic** - Data validation

## 📝 License

This project is for educational purposes as part of VietAI's Foundations of Machine Learning course.

## 👨‍💻 Author

VietAI - Foundations of Machine Learning Student

---

**Note:** Nhớ tải dữ liệu từ Kaggle trước khi chạy notebooks!

