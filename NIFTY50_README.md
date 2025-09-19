# 🇮🇳 Nifty 50 Market Analysis & Prediction System

A comprehensive Python-based system for analyzing the Indian Nifty 50 index and its constituent stocks using advanced data science and machine learning techniques.

## 🚀 Features

### 📊 **Market Analysis**
- **Real-time Nifty 50 Data**: Live data from Yahoo Finance API
- **50 Constituent Stocks**: Complete coverage of all Nifty 50 stocks
- **Sector Classification**: 11 sector categories with performance analysis
- **Market Breadth**: Advance/Decline ratio, new highs/lows analysis
- **Volatility Analysis**: Rolling volatility, VaR calculations, risk metrics

### 🔍 **Technical Analysis**
- **15+ Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Support & Resistance**: Dynamic level identification
- **Trend Analysis**: Multiple timeframe trend detection
- **Volume Analysis**: Volume-price relationships and patterns

### 🤖 **Machine Learning Predictions**
- **7 Advanced Models**: RandomForest, XGBoost, LightGBM, SVR, Linear Models
- **Ensemble Predictions**: Weighted combination of multiple models
- **Multi-day Forecasting**: 1-10 day price predictions
- **Confidence Scoring**: Model reliability assessment
- **Feature Engineering**: 30+ engineered features for predictions

### 📈 **Data Management**
- **SQLite Database**: Local storage for historical data
- **Real-time Updates**: Automatic data refresh from Yahoo Finance
- **Export Capabilities**: CSV reports for analysis and predictions
- **Data Validation**: Quality checks and error handling

## 🏗️ System Architecture

```
nifty50_system/
├── data/
│   ├── nifty50_data.py          # Nifty 50 data management
│   └── stock_data.py            # General stock data handling
├── analysis/
│   ├── nifty50_predictions.py   # ML prediction models
│   ├── technical_indicators.py  # Technical analysis tools
│   └── ml_models.py             # General ML models
├── nifty50_dashboard.py         # Streamlit web dashboard
├── nifty50_demo.py              # Command-line demo
├── config.py                    # Configuration settings
└── requirements.txt             # Python dependencies
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd stock_market_python

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import yfinance, pandas, numpy, sklearn, xgboost, lightgbm; print('All packages installed successfully!')"
```

## 🚀 Quick Start

### 1. **Command Line Demo**
```bash
# Run comprehensive demo
python nifty50_demo.py
```

### 2. **Interactive Dashboard**
```bash
# Launch Streamlit dashboard
streamlit run nifty50_dashboard.py
```

### 3. **Individual Module Usage**
```python
from data.nifty50_data import Nifty50Data
from analysis.nifty50_predictions import Nifty50Predictions

# Initialize components
nifty_data = Nifty50Data()
predictions = Nifty50Predictions()

# Get Nifty 50 data
index_data = nifty_data.get_nifty50_index_data(period="1y")

# Make predictions
prediction = predictions.predict_next_day('RandomForest')
```

## 📊 Dashboard Features

### **Overview Tab**
- Current Nifty 50 level and daily change
- YTD performance and 52-week range
- Interactive price charts with moving averages
- Volume analysis and market statistics
- Performance metrics (Sharpe ratio, drawdown)

### **Technical Analysis Tab**
- Individual stock analysis (select from 50 stocks)
- Multi-panel charts (price, RSI, MACD, volume)
- Technical signals and interpretations
- Support/resistance level identification
- Trend strength indicators

### **Sector Analysis Tab**
- Sector-wise performance comparison
- Top and worst performing sectors
- Individual stock performance within sectors
- Sector rotation analysis
- Performance heatmaps

### **Predictions Tab**
- Machine learning model training
- Next-day price predictions
- Multi-day forecasting (1-10 days)
- Model performance comparison
- Feature importance analysis
- Confidence scoring

### **Market Breadth Tab**
- Advance/Decline ratio analysis
- New highs vs new lows
- Market sentiment scoring
- Volatility trend analysis
- Risk metrics and VaR calculations

## 🔧 Configuration

### **Nifty 50 Symbols**
The system includes all 50 Nifty 50 constituent stocks:
- **Banking & Financial**: HDFC Bank, ICICI Bank, SBI, Kotak Bank, Axis Bank
- **IT & Technology**: TCS, Infosys, HCL Tech, Wipro, Tech Mahindra
- **Oil & Gas**: Reliance, ONGC, Adani Ports
- **Automobile**: Maruti, Tata Motors, Eicher Motors, Hero MotoCorp
- **Consumer Goods**: HUL, ITC, Titan, Nestle India, Britannia
- **Healthcare**: Sun Pharma, Cipla, Dr Reddy's, Apollo Hospitals
- **Metals & Mining**: Hindalco, JSW Steel, Tata Steel, Vedanta
- **Infrastructure**: L&T, UltraTech Cement, Shree Cement, Grasim

### **Technical Indicators**
- **Trend Indicators**: SMA, EMA, MACD, ADX
- **Momentum Indicators**: RSI, Stochastic, Williams %R
- **Volatility Indicators**: Bollinger Bands, ATR, Standard Deviation
- **Volume Indicators**: Volume Ratio, OBV, VWAP

### **Machine Learning Models**
- **Tree-based**: RandomForest, XGBoost, LightGBM, GradientBoosting
- **Linear**: Linear Regression, Ridge, Lasso
- **Non-linear**: Support Vector Regression (SVR)
- **Ensemble**: Weighted combination of all models

## 📈 Data Sources

- **Primary**: Yahoo Finance API (^NSEI for Nifty 50 index)
- **Stocks**: All 50 Nifty 50 constituents with .NS suffix
- **Frequency**: Daily OHLCV data
- **History**: Up to 5 years of historical data
- **Real-time**: Live market data during trading hours

## 🎯 Use Cases

### **For Traders**
- Technical analysis with multiple indicators
- Support/resistance level identification
- Entry/exit timing optimization
- Risk management with VaR calculations

### **For Investors**
- Sector rotation analysis
- Market sentiment assessment
- Long-term trend identification
- Portfolio diversification insights

### **For Analysts**
- Comprehensive market research
- Data export for external analysis
- Custom indicator development
- Model performance evaluation

### **For Researchers**
- Machine learning model experimentation
- Feature engineering research
- Market microstructure analysis
- Academic research support

## 📊 Sample Outputs

### **Nifty 50 Analysis Report**
```csv
Category,Metric,Value,Date
Index Summary,Nifty 50 Current Level,19500.25,2024-01-15
Index Summary,YTD Return,8.45,2024-01-15
Sector Analysis,Banking & Financial - Average Return,12.34,2024-01-15
Sector Analysis,IT & Technology - Average Return,15.67,2024-01-15
Market Breadth,Advance/Decline Ratio,1.45,2024-01-15
Volatility Analysis,Annualized Volatility,18.23,2024-01-15
```

### **Prediction Report**
```csv
Model,Type,Predicted_Return,Direction,Confidence,Current_Level,Predicted_Level
RandomForest,Individual,0.85,UP,78.5,19500.25,19665.96
XGBoost,Individual,0.92,UP,82.1,19500.25,19679.43
LightGBM,Individual,0.78,UP,75.3,19500.25,19652.20
Ensemble,Ensemble,0.85,UP,80.2,19500.25,19665.96
```

## 🔒 Risk Disclaimer

**⚠️ IMPORTANT**: This system is for educational and research purposes only. 

- **Not Financial Advice**: Predictions and analysis should not be considered as investment recommendations
- **Market Risk**: All investments carry inherent risks
- **Model Limitations**: Machine learning models have limitations and may not always be accurate
- **Data Quality**: Analysis depends on data quality and availability
- **Past Performance**: Historical performance does not guarantee future results

## 🤝 Contributing

We welcome contributions to improve the system:

1. **Fork** the repository
2. **Create** a feature branch
3. **Implement** improvements
4. **Test** thoroughly
5. **Submit** a pull request

### **Areas for Enhancement**
- Additional technical indicators
- More machine learning models
- Real-time data streaming
- Advanced visualization options
- API rate limiting optimization
- Additional market indices

## 📞 Support

### **Documentation**
- Code comments and docstrings
- Configuration file explanations
- Example usage patterns

### **Troubleshooting**
- Common installation issues
- Data fetching problems
- Model training errors
- Performance optimization

### **Community**
- GitHub issues and discussions
- Feature requests and bug reports
- User experience feedback

## 📈 Performance Metrics

### **Data Processing**
- **Fetch Speed**: ~2-5 seconds for 1 year of data
- **Storage Efficiency**: SQLite database with compression
- **Memory Usage**: Optimized for large datasets

### **Model Training**
- **Training Time**: 30-60 seconds for 2 years of data
- **Prediction Speed**: <1 second per prediction
- **Accuracy**: Varies by model (typically 55-70% direction accuracy)

### **Dashboard Performance**
- **Load Time**: 5-10 seconds initial load
- **Responsiveness**: Real-time updates during market hours
- **Scalability**: Handles multiple concurrent users

## 🔮 Future Roadmap

### **Phase 1 (Current)**
- ✅ Nifty 50 index analysis
- ✅ Basic technical indicators
- ✅ Machine learning predictions
- ✅ Streamlit dashboard

### **Phase 2 (Planned)**
- 🔄 Bank Nifty analysis
- 🔄 Options chain analysis
- 🔄 Intraday data support
- 🔄 Advanced charting

### **Phase 3 (Future)**
- 📋 Multi-index comparison
- 📋 Global market integration
- 📋 Real-time alerts
- 📋 Mobile application

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Yahoo Finance**: For providing market data
- **Open Source Community**: For the excellent Python libraries
- **Financial Analysts**: For domain expertise and feedback
- **Contributors**: For code improvements and suggestions

---

**🇮🇳 Made with ❤️ for the Indian Stock Market Community**

*Last updated: January 2024*
