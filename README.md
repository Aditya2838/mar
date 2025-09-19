# Stock Market Python Code Generation System

A comprehensive Python-based stock market analysis and trading system with code generation capabilities.

## Features

### 📊 Data Analysis & Visualization
- Real-time stock data fetching using yfinance
- Technical indicators calculation (RSI, MACD, Bollinger Bands, etc.)
- Interactive charts with Plotly and Matplotlib
- Historical data analysis and pattern recognition

### 🤖 Machine Learning & AI
- Price prediction models using scikit-learn
- Sentiment analysis for stock news
- Pattern recognition algorithms
- Automated trading strategies

### 🚀 Web Applications
- Streamlit dashboard for real-time monitoring
- FastAPI backend for API endpoints
- Dash interactive dashboards
- Web scraping for financial news

### 📈 Trading Tools
- Backtesting framework for strategy testing
- Risk management tools
- Portfolio optimization
- Real-time alerts and notifications

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

1. **Data Analysis**: Run `python data_analyzer.py`
2. **Web Dashboard**: Run `streamlit run streamlit_app.py`
3. **API Server**: Run `python api_server.py`
4. **Trading Bot**: Run `python trading_bot.py`

## Project Structure

```
stock_market_python/
├── data/
│   ├── stock_data.py          # Data fetching and storage
│   └── database.py            # Database operations
├── analysis/
│   ├── technical_indicators.py # Technical analysis tools
│   ├── ml_models.py           # Machine learning models
│   └── sentiment_analysis.py  # News sentiment analysis
├── trading/
│   ├── strategies.py          # Trading strategies
│   ├── backtesting.py        # Backtesting framework
│   └── risk_management.py    # Risk management tools
├── visualization/
│   ├── charts.py             # Chart generation
│   └── dashboards.py         # Dashboard components
├── web/
│   ├── streamlit_app.py      # Streamlit dashboard
│   ├── api_server.py         # FastAPI server
│   └── dash_app.py           # Dash application
├── utils/
│   ├── config.py             # Configuration settings
│   ├── logger.py             # Logging utilities
│   └── helpers.py            # Helper functions
└── tests/
    └── test_*.py             # Unit tests
```

## Usage Examples

### Basic Stock Analysis
```python
from data.stock_data import StockData
from analysis.technical_indicators import TechnicalAnalysis

# Fetch stock data
stock = StockData("AAPL")
data = stock.get_historical_data()

# Calculate technical indicators
ta = TechnicalAnalysis(data)
rsi = ta.calculate_rsi()
macd = ta.calculate_macd()
```

### Machine Learning Prediction
```python
from analysis.ml_models import PricePredictor

predictor = PricePredictor("AAPL")
prediction = predictor.predict_next_day()
print(f"Predicted price: ${prediction:.2f}")
```

### Trading Strategy
```python
from trading.strategies import MovingAverageStrategy

strategy = MovingAverageStrategy("AAPL")
signals = strategy.generate_signals()
```

## Configuration

Create a `.env` file in the root directory:
```
YAHOO_FINANCE_API_KEY=your_api_key
ALPHA_VANTAGE_API_KEY=your_api_key
NEWS_API_KEY=your_api_key
DATABASE_URL=sqlite:///stock_data.db
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. Trading stocks involves risk, and past performance does not guarantee future results. Always do your own research and consider consulting with a financial advisor before making investment decisions.
