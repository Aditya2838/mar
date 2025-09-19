"""
Configuration settings for the Stock Market Analysis System
"""

import os
from typing import Dict, Any

class Config:
    """Configuration class for the stock market analysis system"""
    
    # Database settings
    DATABASE_PATH = "stock_data.db"
    
    # Model settings
    MODEL_PATH = "models/"
    
    # API settings (for future use)
    YAHOO_FINANCE_API_KEY = os.getenv("YAHOO_FINANCE_API_KEY", "")
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
    
    # Default analysis settings
    DEFAULT_PERIOD = "1y"
    DEFAULT_INTERVAL = "1d"
    
    # Technical analysis settings
    RSI_PERIOD = 14
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    
    MACD_FAST_PERIOD = 12
    MACD_SLOW_PERIOD = 26
    MACD_SIGNAL_PERIOD = 9
    
    BOLLINGER_PERIOD = 20
    BOLLINGER_STD_DEV = 2
    
    # Machine learning settings
    LOOKBACK_DAYS = 30
    TEST_SIZE = 0.2
    
    # Trading strategy settings
    DEFAULT_INITIAL_CAPITAL = 10000
    DEFAULT_COMMISSION = 0.001  # 0.1% commission
    
    # Portfolio optimization settings
    RISK_FREE_RATE = 0.02  # 2% risk-free rate
    
    # Web application settings
    STREAMLIT_PORT = 8501
    STREAMLIT_HOST = "localhost"
    
    # Logging settings
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Popular stock symbols for demonstration
    POPULAR_SYMBOLS = [
        "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "NFLX",
        "^GSPC", "^DJI", "^IXIC", "^RUT"  # Major indices
    ]
    
    # Nifty 50 specific symbols and configuration
    NIFTY_50_SYMBOLS = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
        "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "HCLTECH.NS", "SUNPHARMA.NS",
        "TITAN.NS", "WIPRO.NS", "ULTRACEMCO.NS", "TECHM.NS", "NESTLEIND.NS",
        "POWERGRID.NS", "BAJFINANCE.NS", "NTPC.NS", "HINDALCO.NS", "JSWSTEEL.NS",
        "ONGC.NS", "TATAMOTORS.NS", "ADANIENT.NS", "COALINDIA.NS", "INDUSINDBK.NS",
        "CIPLA.NS", "DRREDDY.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "BRITANNIA.NS",
        "DIVISLAB.NS", "SHREECEM.NS", "APOLLOHOSP.NS", "BAJAJFINSV.NS", "TATACONSUM.NS",
        "HDFC.NS", "SBILIFE.NS", "HDFCLIFE.NS", "ICICIGI.NS", "ADANIPORTS.NS",
        "TATASTEEL.NS", "VEDL.NS", "GRASIM.NS", "M&M.NS", "LT.NS"
    ]
    
    # Nifty 50 index symbol
    NIFTY_50_INDEX = "^NSEI"
    
    # Indian market specific settings
    INDIAN_MARKET_HOURS = {
        "pre_market": "09:00",
        "market_open": "09:15",
        "market_close": "15:30",
        "post_market": "15:45"
    }
    
    # Indian market holidays (major ones)
    INDIAN_MARKET_HOLIDAYS = [
        "2024-01-26", "2024-03-08", "2024-03-25", "2024-04-09", "2024-04-10",
        "2024-04-11", "2024-05-01", "2024-08-15", "2024-10-02", "2024-11-15",
        "2024-12-25"
    ]
    
    # Sector classifications for Nifty 50
    NIFTY_50_SECTORS = {
        "Banking & Financial": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", 
                               "AXISBANK.NS", "INDUSINDBK.NS", "HDFC.NS", "SBILIFE.NS", 
                               "HDFCLIFE.NS", "ICICIGI.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS"],
        "IT & Technology": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"],
        "Oil & Gas": ["RELIANCE.NS", "ONGC.NS", "ADANIPORTS.NS"],
        "Automobile": ["MARUTI.NS", "TATAMOTORS.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "M&M.NS"],
        "Consumer Goods": ["HINDUNILVR.NS", "ITC.NS", "TITAN.NS", "NESTLEIND.NS", 
                          "BRITANNIA.NS", "TATACONSUM.NS"],
        "Healthcare": ["SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "APOLLOHOSP.NS", "DIVISLAB.NS"],
        "Metals & Mining": ["HINDALCO.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "VEDL.NS", "COALINDIA.NS"],
        "Infrastructure": ["LT.NS", "ULTRACEMCO.NS", "SHREECEM.NS", "GRASIM.NS"],
        "Telecom": ["BHARTIARTL.NS"],
        "Power": ["POWERGRID.NS", "NTPC.NS"],
        "Others": ["ASIANPAINT.NS", "ADANIENT.NS"]
    }
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get machine learning model configuration"""
        return {
            'lookback_days': cls.LOOKBACK_DAYS,
            'test_size': cls.TEST_SIZE,
            'feature_columns': [
                'Returns', 'Price_Change', 'High_Low_Ratio', 'Close_Open_Ratio',
                'SMA_5', 'SMA_20', 'EMA_12', 'EMA_26', 'Volatility', 'ATR',
                'Volume_Ratio', 'RSI', 'MACD', 'MACD_Signal', 'BB_Position'
            ]
        }
    
    @classmethod
    def get_technical_config(cls) -> Dict[str, Any]:
        """Get technical analysis configuration"""
        return {
            'rsi_period': cls.RSI_PERIOD,
            'rsi_oversold': cls.RSI_OVERSOLD,
            'rsi_overbought': cls.RSI_OVERBOUGHT,
            'macd_fast': cls.MACD_FAST_PERIOD,
            'macd_slow': cls.MACD_SLOW_PERIOD,
            'macd_signal': cls.MACD_SIGNAL_PERIOD,
            'bollinger_period': cls.BOLLINGER_PERIOD,
            'bollinger_std': cls.BOLLINGER_STD_DEV
        }
    
    @classmethod
    def get_trading_config(cls) -> Dict[str, Any]:
        """Get trading strategy configuration"""
        return {
            'initial_capital': cls.DEFAULT_INITIAL_CAPITAL,
            'commission': cls.DEFAULT_COMMISSION,
            'risk_free_rate': cls.RISK_FREE_RATE
        }
    
    @classmethod
    def get_web_config(cls) -> Dict[str, Any]:
        """Get web application configuration"""
        return {
            'port': cls.STREAMLIT_PORT,
            'host': cls.STREAMLIT_HOST,
            'page_title': "Stock Market Analysis Dashboard",
            'page_icon': "📈"
        }

# Comprehensive Indian Stock Market Configuration
INDIAN_STOCKS = {
    # Nifty 50 (Large Cap)
    "NIFTY_50": Config.NIFTY_50_SYMBOLS,
    
    # Nifty Next 50 (Mid Cap)
    "NIFTY_NEXT_50": [
        "PERSISTENT.NS", "ABBOTINDIA.NS", "ADANIGREEN.NS", "ADANITRANS.NS", "ALKEM.NS",
        "AMBUJACEM.NS", "APOLLOTYRE.NS", "ASHOKLEY.NS", "AUBANK.NS", "BANDHANBNK.NS",
        "BERGEPAINT.NS", "BIOCON.NS", "BOSCHLTD.NS", "CADILAHC.NS", "COLPAL.NS",
        "DLF.NS", "DABUR.NS", "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS",
        "FEDERALBNK.NS", "GAIL.NS", "GODREJCP.NS", "HCLTECH.NS", "HDFCAMC.NS",
        "HINDALCO.NS", "HINDPETRO.NS", "ICICIPRULI.NS", "INDUSINDBK.NS", "INFY.NS",
        "IOC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS",
        "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS",
        "RELIANCE.NS", "SBIN.NS", "SUNPHARMA.NS", "TATAMOTORS.NS", "TATASTEEL.NS",
        "TCS.NS", "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS", "VEDL.NS",
        "WIPRO.NS", "ZEEL.NS"
    ],
    
    # Nifty Midcap 100
    "NIFTY_MIDCAP_100": [
        "ABBOTINDIA.NS", "ADANIGREEN.NS", "ADANITRANS.NS", "ALKEM.NS", "AMBUJACEM.NS",
        "APOLLOTYRE.NS", "ASHOKLEY.NS", "AUBANK.NS", "BANDHANBNK.NS", "BERGEPAINT.NS",
        "BIOCON.NS", "BOSCHLTD.NS", "CADILAHC.NS", "COLPAL.NS", "DLF.NS",
        "DABUR.NS", "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS", "FEDERALBNK.NS",
        "GAIL.NS", "GODREJCP.NS", "HCLTECH.NS", "HDFCAMC.NS", "HINDALCO.NS",
        "HINDPETRO.NS", "ICICIPRULI.NS", "INDUSINDBK.NS", "INFY.NS", "IOC.NS",
        "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS", "MARUTI.NS",
        "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS",
        "SBIN.NS", "SUNPHARMA.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TCS.NS",
        "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS", "VEDL.NS", "WIPRO.NS",
        "ZEEL.NS", "ACC.NS", "ADANIENT.NS", "ADANIPORTS.NS", "ASIANPAINT.NS",
        "AXISBANK.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "BRITANNIA.NS",
        "CIPLA.NS", "COALINDIA.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS",
        "HINDUNILVR.NS", "ICICIBANK.NS", "ITC.NS", "SBILIFE.NS", "SHREECEM.NS",
        "TATACONSUM.NS", "UPL.NS", "WIPRO.NS"
    ],
    
    # Nifty Smallcap 100
    "NIFTY_SMALLCAP_100": [
        "AARTIIND.NS", "ABB.NS", "ABBOTINDIA.NS", "ADANIENT.NS", "ADANIPORTS.NS",
        "ALKEM.NS", "AMBUJACEM.NS", "APOLLOTYRE.NS", "ASHOKLEY.NS", "AUBANK.NS",
        "BANDHANBNK.NS", "BERGEPAINT.NS", "BIOCON.NS", "BOSCHLTD.NS", "CADILAHC.NS",
        "COLPAL.NS", "DLF.NS", "DABUR.NS", "DIVISLAB.NS", "DRREDDY.NS",
        "EICHERMOT.NS", "FEDERALBNK.NS", "GAIL.NS", "GODREJCP.NS", "HCLTECH.NS",
        "HDFCAMC.NS", "HINDALCO.NS", "HINDPETRO.NS", "ICICIPRULI.NS", "INDUSINDBK.NS",
        "INFY.NS", "IOC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS",
        "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
        "POWERGRID.NS", "RELIANCE.NS", "SBIN.NS", "SUNPHARMA.NS", "TATAMOTORS.NS",
        "TATASTEEL.NS", "TCS.NS", "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS",
        "VEDL.NS", "WIPRO.NS", "ZEEL.NS"
    ],
    
    # Bank Nifty Constituents
    "BANK_NIFTY": [
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
        "INDUSINDBK.NS", "BANDHANBNK.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS", "PNB.NS",
        "SBILIFE.NS", "HDFCLIFE.NS", "ICICIGI.NS", "ICICIPRULI.NS", "HDFCAMC.NS",
        "BAJFINANCE.NS", "BAJAJFINSV.NS", "CHOLAFIN.NS", "MUTHOOTFIN.NS", "PEL.NS"
    ],
    
    # IT Sector
    "IT_SECTOR": [
        "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
        "MINDTREE.NS", "LTI.NS", "MPHASIS.NS", "PERSISTENT.NS", "COFORGE.NS",
        "LARSEN.NS", "TATAELXSI.NS", "NIITTECH.NS", "HEXAWARE.NS", "CYIENT.NS"
    ],
    
    # Pharma Sector
    "PHARMA_SECTOR": [
        "SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "DIVISLAB.NS", "APOLLOHOSP.NS",
        "BIOCON.NS", "CADILAHC.NS", "LUPIN.NS", "TORRENTPHARMA.NS", "ALKEM.NS",
        "AUROPHARMA.NS", "GLENMARK.NS", "NATCOPHARMA.NS", "PIIND.NS", "SANOFI.NS"
    ],
    
    # Auto Sector
    "AUTO_SECTOR": [
        "MARUTI.NS", "TATAMOTORS.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "M&M.NS",
        "BAJAJAUTO.NS", "TVSMOTOR.NS", "ASHOKLEY.NS", "MRF.NS", "APOLLOTYRE.NS",
        "CEAT.NS", "JKTYRE.NS", "BALKRISIND.NS", "BHARATFORG.NS", "MOTHERSUMI.NS"
    ],
    
    # FMCG Sector
    "FMCG_SECTOR": [
        "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "TATACONSUM.NS",
        "MARICO.NS", "DABUR.NS", "COLPAL.NS", "GODREJCP.NS", "UBL.NS",
        "EMAMILTD.NS", "VBL.NS", "RADICO.NS", "TATAGLOBAL.NS", "JUBLFOOD.NS"
    ],
    
    # Metal Sector
    "METAL_SECTOR": [
        "TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "VEDL.NS", "COALINDIA.NS",
        "HINDCOPPER.NS", "NATIONALUM.NS", "WELCORP.NS", "RATNAMANI.NS", "APLAPOLLO.NS",
        "MAHSEAMLES.NS", "TATASTEEL.NS", "JINDALSTEL.NS", "SAIL.NS", "MOIL.NS"
    ],
    
    # Energy Sector
    "ENERGY_SECTOR": [
        "RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "ADANIENT.NS",
        "ADANIGREEN.NS", "ADANITRANS.NS", "TATAPOWER.NS", "SJVN.NS", "NHPC.NS",
        "POWERGRID.NS", "NLCINDIA.NS", "COALINDIA.NS", "OIL.NS", "GAIL.NS"
    ]
}

# All Indian stocks for intraday trading (unique list)
ALL_INDIAN_STOCKS = list(set([
    symbol for sector_stocks in INDIAN_STOCKS.values() 
    for symbol in sector_stocks
]))

# Popular intraday trading stocks (high volume, high volatility)
POPULAR_INTRADAY_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "HCLTECH.NS", "SUNPHARMA.NS",
    "TITAN.NS", "WIPRO.NS", "ULTRACEMCO.NS", "TECHM.NS", "NESTLEIND.NS",
    "POWERGRID.NS", "BAJFINANCE.NS", "NTPC.NS", "HINDALCO.NS", "JSWSTEEL.NS",
    "ONGC.NS", "TATAMOTORS.NS", "ADANIENT.NS", "COALINDIA.NS", "INDUSINDBK.NS",
    "CIPLA.NS", "DRREDDY.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "BRITANNIA.NS",
    "DIVISLAB.NS", "SHREECEM.NS", "APOLLOHOSP.NS", "BAJAJFINSV.NS", "TATACONSUM.NS",
    "HDFC.NS", "SBILIFE.NS", "HDFCLIFE.NS", "ICICIGI.NS", "ADANIPORTS.NS",
    "TATASTEEL.NS", "VEDL.NS", "GRASIM.NS", "M&M.NS", "LT.NS"
]

# Intraday trading specific settings
INTRADAY_TRADING_CONFIG = {
    "default_interval": "15m",
    "default_period": "5d",
    "prediction_horizon": 6,
    "stop_loss_percentage": 1.5,
    "target_percentages": [1.0, 2.0, 3.0],
    "max_positions": 5,
    "risk_per_trade": 2.0,  # 2% risk per trade
    "min_volume": 1000000,  # Minimum volume for intraday
    "min_market_cap": 10000,  # Minimum market cap in crores
    "trading_hours": {
        "pre_market": "09:00",
        "market_open": "09:15",
        "market_close": "15:30",
        "post_market": "15:45"
    }
}
