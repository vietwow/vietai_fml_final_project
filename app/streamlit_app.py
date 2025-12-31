"""
Streamlit Frontend for House Price Prediction
VietAI - Foundations of Machine Learning Final Project
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Cards */
    .stCard {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Headers */
    h1 {
        color: #2E86AB;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Prediction box */
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
    }
    
    .prediction-price {
        font-size: 48px;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .confidence-range {
        font-size: 18px;
        opacity: 0.9;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: #f8f9fa;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 30px;
        font-size: 18px;
        font-weight: bold;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Feature cards */
    .feature-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #667eea;
    }
    
    /* Metrics */
    .metric-container {
        background: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #2E86AB;
    }
    
    .metric-label {
        font-size: 14px;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# API URL
API_URL = st.sidebar.text_input(
    "API URL",
    value="http://localhost:8000",
    help="URL của FastAPI server"
)


def check_api_health():
    """Check if API is healthy."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.json()
    except:
        return None


def get_prediction(features: dict):
    """Get prediction from API."""
    try:
        response = requests.post(f"{API_URL}/predict", json=features, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.json().get("detail", "Unknown error")}
    except requests.exceptions.ConnectionError:
        return {"error": "Không thể kết nối đến API. Hãy chắc chắn rằng API đang chạy."}
    except Exception as e:
        return {"error": str(e)}


def main():
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1>🏠 Dự Đoán Giá Nhà</h1>
        <p style='font-size: 18px; color: #666;'>
            VietAI - Foundations of Machine Learning | Final Project
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check API health
    health = check_api_health()
    
    if health and health.get("model_loaded"):
        st.sidebar.success(f"✅ API Connected\n\nModel: {health.get('model_name', 'Unknown')}")
    else:
        st.sidebar.warning("⚠️ API không khả dụng hoặc model chưa được load")
        st.sidebar.info("""
        **Hướng dẫn:**
        1. Chạy training notebook để tạo model
        2. Khởi động API:
        ```bash
        cd api
        uvicorn main:app --reload
        ```
        """)
    
    # Create columns for input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Nhập Thông Tin Ngôi Nhà")
        
        # Basic Information
        st.markdown("#### 🏗️ Thông Tin Cơ Bản")
        basic_col1, basic_col2, basic_col3 = st.columns(3)
        
        with basic_col1:
            overall_qual = st.slider(
                "Chất lượng tổng thể",
                min_value=1,
                max_value=10,
                value=7,
                help="1 = Kém nhất, 10 = Tốt nhất"
            )
            
            year_built = st.number_input(
                "Năm xây dựng",
                min_value=1800,
                max_value=2024,
                value=2005
            )
        
        with basic_col2:
            overall_cond = st.slider(
                "Điều kiện tổng thể",
                min_value=1,
                max_value=10,
                value=5,
                help="1 = Kém nhất, 10 = Tốt nhất"
            )
            
            year_remod = st.number_input(
                "Năm cải tạo",
                min_value=1800,
                max_value=2024,
                value=2005
            )
        
        with basic_col3:
            neighborhood = st.selectbox(
                "Khu vực",
                options=["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst",
                         "Gilbert", "NridgHt", "Sawyer", "NWAmes", "SawyerW",
                         "NoRidge", "Timber", "Veenker", "Crawfor", "Mitchel"],
                index=0
            )
            
            bldg_type = st.selectbox(
                "Loại công trình",
                options=["1Fam", "2fmCon", "Duplex", "TwnhsE", "Twnhs"],
                format_func=lambda x: {
                    "1Fam": "Nhà đơn lẻ",
                    "2fmCon": "Nhà 2 gia đình",
                    "Duplex": "Duplex",
                    "TwnhsE": "Townhouse End",
                    "Twnhs": "Townhouse"
                }.get(x, x)
            )
        
        st.markdown("---")
        
        # Area Information
        st.markdown("#### 📐 Diện Tích")
        area_col1, area_col2, area_col3 = st.columns(3)
        
        with area_col1:
            gr_liv_area = st.number_input(
                "Diện tích sinh hoạt (sq ft)",
                min_value=100,
                max_value=10000,
                value=1500,
                step=50
            )
            
            first_flr_sf = st.number_input(
                "Diện tích tầng 1 (sq ft)",
                min_value=0,
                max_value=5000,
                value=1000,
                step=50
            )
        
        with area_col2:
            lot_area = st.number_input(
                "Diện tích đất (sq ft)",
                min_value=1000,
                max_value=100000,
                value=10000,
                step=500
            )
            
            second_flr_sf = st.number_input(
                "Diện tích tầng 2 (sq ft)",
                min_value=0,
                max_value=5000,
                value=500,
                step=50
            )
        
        with area_col3:
            total_bsmt_sf = st.number_input(
                "Diện tích tầng hầm (sq ft)",
                min_value=0,
                max_value=5000,
                value=1000,
                step=50
            )
            
            garage_area = st.number_input(
                "Diện tích garage (sq ft)",
                min_value=0,
                max_value=2000,
                value=500,
                step=50
            )
        
        st.markdown("---")
        
        # Rooms & Features
        st.markdown("#### 🛁 Phòng & Tiện Ích")
        room_col1, room_col2, room_col3, room_col4 = st.columns(4)
        
        with room_col1:
            tot_rms = st.number_input(
                "Tổng số phòng",
                min_value=1,
                max_value=20,
                value=7
            )
        
        with room_col2:
            bedrooms = st.number_input(
                "Phòng ngủ",
                min_value=0,
                max_value=10,
                value=3
            )
        
        with room_col3:
            full_bath = st.number_input(
                "Phòng tắm đầy đủ",
                min_value=0,
                max_value=5,
                value=2
            )
        
        with room_col4:
            half_bath = st.number_input(
                "Phòng tắm nửa",
                min_value=0,
                max_value=5,
                value=1
            )
        
        feature_col1, feature_col2, feature_col3, feature_col4 = st.columns(4)
        
        with feature_col1:
            garage_cars = st.number_input(
                "Sức chứa garage (xe)",
                min_value=0,
                max_value=5,
                value=2
            )
        
        with feature_col2:
            fireplaces = st.number_input(
                "Số lò sưởi",
                min_value=0,
                max_value=5,
                value=1
            )
        
        with feature_col3:
            kitchen_qual = st.selectbox(
                "Chất lượng bếp",
                options=["Ex", "Gd", "TA", "Fa", "Po"],
                index=2,
                format_func=lambda x: {
                    "Ex": "Xuất sắc",
                    "Gd": "Tốt",
                    "TA": "Trung bình",
                    "Fa": "Khá",
                    "Po": "Kém"
                }.get(x, x)
            )
        
        with feature_col4:
            exter_qual = st.selectbox(
                "Chất lượng ngoại thất",
                options=["Ex", "Gd", "TA", "Fa", "Po"],
                index=2,
                format_func=lambda x: {
                    "Ex": "Xuất sắc",
                    "Gd": "Tốt",
                    "TA": "Trung bình",
                    "Fa": "Khá",
                    "Po": "Kém"
                }.get(x, x)
            )
        
        st.markdown("---")
        
        # Predict button
        predict_button = st.button("🔮 DỰ ĐOÁN GIÁ NHÀ", width="stretch")
    
    with col2:
        st.subheader("📊 Kết Quả Dự Đoán")
        
        if predict_button:
            # Prepare features
            features = {
                "OverallQual": overall_qual,
                "OverallCond": overall_cond,
                "GrLivArea": gr_liv_area,
                "YearBuilt": year_built,
                "YearRemodAdd": year_remod,
                "FullBath": full_bath,
                "HalfBath": half_bath,
                "TotRmsAbvGrd": tot_rms,
                "TotalBsmtSF": total_bsmt_sf,
                "GarageCars": garage_cars,
                "GarageArea": garage_area,
                "Fireplaces": fireplaces,
                "LotArea": lot_area,
                "BedroomAbvGr": bedrooms,
                "KitchenAbvGr": 1,
                "1stFlrSF": first_flr_sf,
                "2ndFlrSF": second_flr_sf,
                "Neighborhood": neighborhood,
                "BldgType": bldg_type,
                "HouseStyle": "1Story" if second_flr_sf == 0 else "2Story",
                "ExterQual": exter_qual,
                "KitchenQual": kitchen_qual
            }
            
            with st.spinner("Đang dự đoán..."):
                result = get_prediction(features)
            
            if "error" in result:
                st.error(f"❌ Lỗi: {result['error']}")
            else:
                # Display prediction
                predicted_price = result["predicted_price"]
                price_formatted = result["predicted_price_formatted"]
                confidence = result["confidence_interval"]
                
                st.markdown(f"""
                <div class='prediction-box'>
                    <div style='font-size: 16px;'>Giá Nhà Dự Đoán</div>
                    <div class='prediction-price'>{price_formatted}</div>
                    <div class='confidence-range'>
                        Khoảng tin cậy: {confidence['formatted']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Price gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=predicted_price,
                    number={'prefix': "$", 'valueformat': ",.0f"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [None, predicted_price * 1.5]},
                        'bar': {'color': "#667eea"},
                        'steps': [
                            {'range': [0, confidence['lower']], 'color': "#e8f4f8"},
                            {'range': [confidence['lower'], confidence['upper']], 'color': "#b3d9e8"},
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': predicted_price
                        }
                    }
                ))
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, width="stretch")
                
                # Model info
                model_info = result.get("model_info", {})
                st.markdown(f"""
                <div class='feature-card'>
                    <strong>📈 Thông tin Model:</strong><br>
                    Model: {model_info.get('model_name', 'N/A')}<br>
                    R² Score: {model_info.get('test_r2', 0):.4f}
                </div>
                """, unsafe_allow_html=True)
                
                # Feature summary
                st.markdown("#### 📋 Tóm Tắt Thông Tin")
                summary_data = {
                    "Đặc điểm": [
                        "Chất lượng", "Diện tích sinh hoạt", "Tuổi nhà",
                        "Phòng tắm", "Garage", "Lò sưởi"
                    ],
                    "Giá trị": [
                        f"{overall_qual}/10",
                        f"{gr_liv_area:,} sq ft",
                        f"{2024 - year_built} năm",
                        f"{full_bath} full + {half_bath} half",
                        f"{garage_cars} xe ({garage_area} sq ft)",
                        f"{fireplaces}"
                    ]
                }
                st.table(pd.DataFrame(summary_data))
        
        else:
            st.info("👆 Nhập thông tin và nhấn nút **Dự Đoán** để xem kết quả")
            
            # Show sample predictions
            st.markdown("#### 📚 Ví Dụ Tham Khảo")
            sample_data = {
                "Loại nhà": ["Nhà cơ bản", "Nhà trung bình", "Nhà cao cấp"],
                "Diện tích": ["1,200 sq ft", "1,800 sq ft", "2,500 sq ft"],
                "Giá ước tính": ["$120,000", "$200,000", "$350,000"]
            }
            st.table(pd.DataFrame(sample_data))
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🎓 VietAI - Foundations of Machine Learning Final Project</p>
        <p>Data source: <a href='https://www.kaggle.com/c/house-prices-advanced-regression-techniques'>
        Kaggle House Prices Competition</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

