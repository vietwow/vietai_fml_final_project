#!/bin/bash

# House Price Prediction - Quick Start Script
# VietAI - Foundations of Machine Learning Final Project

# Get the script directory (project root)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "🏠 House Price Prediction - VietAI ML Final Project"
echo "====================================================="

# Check if virtual environment exists
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv "$SCRIPT_DIR/venv"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source "$SCRIPT_DIR/venv/bin/activate"

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r "$SCRIPT_DIR/requirements.txt" -q

# Create data directories
mkdir -p "$SCRIPT_DIR/data/raw" "$SCRIPT_DIR/data/processed" "$SCRIPT_DIR/models" "$SCRIPT_DIR/reports"

# Check if data exists
if [ ! -f "$SCRIPT_DIR/data/raw/train.csv" ]; then
    echo ""
    echo "⚠️  Data files not found!"
    echo "   Please download data from Kaggle:"
    echo "   https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data"
    echo "   And place train.csv and test.csv in data/raw/ folder"
    echo ""
fi

# Function to show menu
show_menu() {
    echo ""
    echo "What would you like to do?"
    echo "1) Run EDA Notebook"
    echo "2) Run Training Notebook"
    echo "3) Start API Server"
    echo "4) Start Streamlit App"
    echo "5) Start Both API and Streamlit"
    echo "6) Exit"
    echo ""
}

# Function to run options
run_option() {
    case $1 in
        1)
            echo "🔍 Opening EDA Notebook..."
            jupyter notebook "$SCRIPT_DIR/notebooks/01_EDA.ipynb"
            ;;
        2)
            echo "🎓 Opening Training Notebook..."
            jupyter notebook "$SCRIPT_DIR/notebooks/02_Training.ipynb"
            ;;
        3)
            echo "🚀 Starting API Server on port 8000..."
            cd "$SCRIPT_DIR/api" && uvicorn main:app --reload --port 8000
            ;;
        4)
            echo "🎨 Starting Streamlit App on port 8501..."
            cd "$SCRIPT_DIR/app" && streamlit run streamlit_app.py
            ;;
        5)
            echo "🚀 Starting API Server..."
            (cd "$SCRIPT_DIR/api" && uvicorn main:app --reload --port 8000) &
            API_PID=$!
            sleep 3
            echo "🎨 Starting Streamlit App..."
            cd "$SCRIPT_DIR/app" && streamlit run streamlit_app.py
            
            # Cleanup: kill API when Streamlit exits
            kill $API_PID 2>/dev/null
            ;;
        6)
            echo "👋 Goodbye!"
            exit 0
            ;;
        *)
            echo "Invalid option!"
            ;;
    esac
}

# Main loop
while true; do
    show_menu
    read -p "Enter your choice (1-6): " choice
    run_option $choice
done

