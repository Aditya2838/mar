"""
Nifty 50 Data Management Module
Specialized module for Indian Nifty 50 index and constituent stocks analysis
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import os
from typing import Dict, List, Optional, Tuple, Any
import logging
from config import Config

class Nifty50Data:
    """
    Specialized class for Nifty 50 data operations
    """
    
    def __init__(self, database_path: str = "nifty50_data.db"):
        """
        Initialize Nifty50Data
        
        Args:
            database_path (str): Path to SQLite database for Nifty 50 data
        """
        self.database_path = database_path
        self.nifty_index = yf.Ticker(Config.NIFTY_50_INDEX)
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with Nifty 50 specific tables"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Create nifty50_data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nifty50_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                adj_close REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date)
            )
        ''')
        
        # Create nifty50_technical_indicators table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nifty50_technical_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                rsi REAL,
                macd REAL,
                macd_signal REAL,
                bollinger_upper REAL,
                bollinger_lower REAL,
                sma_20 REAL,
                sma_50 REAL,
                ema_12 REAL,
                ema_26 REAL,
                atr REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date)
            )
        ''')
        
        # Create nifty50_sector_performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nifty50_sector_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sector TEXT NOT NULL,
                date TEXT NOT NULL,
                sector_return REAL,
                sector_volume REAL,
                top_performer TEXT,
                worst_performer TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(sector, date)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_nifty50_index_data(self, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch Nifty 50 index data
        
        Args:
            period (str): Data period
            interval (str): Data interval
        
        Returns:
            pd.DataFrame: Nifty 50 index data
        """
        try:
            data = self.nifty_index.history(period=period, interval=interval)
            if data.empty:
                raise ValueError(f"No data found for Nifty 50 index")
            
            data.reset_index(inplace=True)
            data['Symbol'] = '^NSEI'
            
            # Store in database
            self._store_nifty50_data(data, '^NSEI')
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching Nifty 50 index data: {e}")
            return pd.DataFrame()
    
    def get_nifty50_constituents_data(self, period: str = "1y", 
                                    symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for Nifty 50 constituent stocks
        
        Args:
            period (str): Data period
            symbols (List[str]): List of symbols to fetch (default: all Nifty 50)
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of stock data
        """
        if symbols is None:
            symbols = Config.NIFTY_50_SYMBOLS
        
        constituents_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                
                if not data.empty:
                    data.reset_index(inplace=True)
                    data['Symbol'] = symbol
                    
                    # Store in database
                    self._store_nifty50_data(data, symbol)
                    
                    constituents_data[symbol] = data
                    
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {e}")
                continue
        
        return constituents_data
    
    def get_sector_analysis(self, period: str = "1y") -> Dict[str, Dict]:
        """
        Analyze sector-wise performance
        
        Args:
            period (str): Data period for analysis
        
        Returns:
            Dict[str, Dict]: Sector-wise analysis
        """
        sector_analysis = {}
        
        for sector_name, sector_symbols in Config.NIFTY_50_SECTORS.items():
            sector_data = {}
            sector_returns = []
            
            for symbol in sector_symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period=period)
                    
                    if not data.empty:
                        # Calculate returns
                        returns = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]
                        sector_returns.append(returns)
                        
                        sector_data[symbol] = {
                            'current_price': data['Close'].iloc[-1],
                            'returns': returns,
                            'volume': data['Volume'].iloc[-1],
                            'high_52w': data['High'].max(),
                            'low_52w': data['Low'].min()
                        }
                        
                except Exception as e:
                    self.logger.error(f"Error analyzing {symbol} for sector {sector_name}: {e}")
                    continue
            
            if sector_returns:
                sector_analysis[sector_name] = {
                    'avg_return': np.mean(sector_returns),
                    'total_return': np.sum(sector_returns),
                    'stocks': sector_data,
                    'top_performer': max(sector_data.items(), key=lambda x: x[1]['returns'])[0],
                    'worst_performer': min(sector_data.items(), key=lambda x: x[1]['returns'])[0]
                }
        
        return sector_analysis
    
    def get_market_breadth(self, date: str = None) -> Dict[str, Any]:
        """
        Calculate market breadth indicators
        
        Args:
            date (str): Date for analysis (default: latest)
        
        Returns:
            Dict[str, Any]: Market breadth indicators
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            # Get latest data for all Nifty 50 stocks
            constituents_data = self.get_nifty50_constituents_data(period="5d")
            
            advancing = 0
            declining = 0
            unchanged = 0
            new_highs = 0
            new_lows = 0
            
            for symbol, data in constituents_data.items():
                if len(data) >= 2:
                    current_close = data['Close'].iloc[-1]
                    prev_close = data['Close'].iloc[-2]
                    
                    if current_close > prev_close:
                        advancing += 1
                    elif current_close < prev_close:
                        declining += 1
                    else:
                        unchanged += 1
                    
                    # Check for new highs/lows (52-week)
                    high_52w = data['High'].max()
                    low_52w = data['Low'].min()
                    
                    if current_close >= high_52w:
                        new_highs += 1
                    if current_close <= low_52w:
                        new_lows += 1
            
            total_stocks = len(constituents_data)
            
            market_breadth = {
                'date': date,
                'advancing': advancing,
                'declining': declining,
                'unchanged': unchanged,
                'advance_decline_ratio': advancing / declining if declining > 0 else float('inf'),
                'new_highs': new_highs,
                'new_lows': new_lows,
                'total_stocks': total_stocks,
                'advancing_percentage': (advancing / total_stocks) * 100,
                'declining_percentage': (declining / total_stocks) * 100
            }
            
            return market_breadth
            
        except Exception as e:
            self.logger.error(f"Error calculating market breadth: {e}")
            return {}
    
    def get_volatility_analysis(self, period: str = "1y") -> Dict[str, Any]:
        """
        Analyze volatility patterns
        
        Args:
            period (str): Data period for analysis
        
        Returns:
            Dict[str, Any]: Volatility analysis
        """
        try:
            # Get Nifty 50 index data
            index_data = self.get_nifty50_index_data(period=period)
            
            if index_data.empty:
                return {}
            
            # Calculate daily returns
            index_data['Returns'] = index_data['Close'].pct_change()
            
            # Calculate volatility metrics
            daily_volatility = index_data['Returns'].std()
            annualized_volatility = daily_volatility * np.sqrt(252)  # Assuming 252 trading days
            
            # Calculate rolling volatility (20-day)
            index_data['Rolling_Volatility'] = index_data['Returns'].rolling(window=20).std() * np.sqrt(252)
            
            # Calculate Value at Risk (VaR)
            var_95 = np.percentile(index_data['Returns'].dropna(), 5)
            var_99 = np.percentile(index_data['Returns'].dropna(), 1)
            
            volatility_analysis = {
                'daily_volatility': daily_volatility,
                'annualized_volatility': annualized_volatility,
                'current_rolling_volatility': index_data['Rolling_Volatility'].iloc[-1],
                'var_95': var_95,
                'var_99': var_99,
                'max_daily_return': index_data['Returns'].max(),
                'min_daily_return': index_data['Returns'].min(),
                'volatility_trend': 'increasing' if index_data['Rolling_Volatility'].iloc[-1] > 
                                   index_data['Rolling_Volatility'].iloc[-20] else 'decreasing'
            }
            
            return volatility_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing volatility: {e}")
            return {}
    
    def _store_nifty50_data(self, data: pd.DataFrame, symbol: str):
        """Store Nifty 50 data in database"""
        try:
            conn = sqlite3.connect(self.database_path)
            
            for _, row in data.iterrows():
                conn.execute('''
                    INSERT OR REPLACE INTO nifty50_data 
                    (symbol, date, open, high, low, close, volume, adj_close)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol, row['Date'].strftime('%Y-%m-%d'), 
                    row.get('Open', 0), row.get('High', 0), row.get('Low', 0),
                    row.get('Close', 0), row.get('Volume', 0), row.get('Adj Close', 0)
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing data for {symbol}: {e}")
    
    def export_nifty50_report(self, filename: str = "nifty50_analysis_report.csv"):
        """
        Export comprehensive Nifty 50 analysis report
        
        Args:
            filename (str): Output filename
        """
        try:
            # Get all analysis data
            index_data = self.get_nifty50_index_data(period="1y")
            sector_analysis = self.get_sector_analysis(period="1y")
            market_breadth = self.get_market_breadth()
            volatility_analysis = self.get_volatility_analysis(period="1y")
            
            # Create comprehensive report
            report_data = []
            
            # Add index summary
            if not index_data.empty:
                report_data.append({
                    'Category': 'Index Summary',
                    'Metric': 'Nifty 50 Current Level',
                    'Value': index_data['Close'].iloc[-1],
                    'Date': index_data['Date'].iloc[-1].strftime('%Y-%m-%d')
                })
                
                report_data.append({
                    'Category': 'Index Summary',
                    'Metric': 'YTD Return',
                    'Value': ((index_data['Close'].iloc[-1] - index_data['Close'].iloc[0]) / 
                             index_data['Close'].iloc[0]) * 100,
                    'Date': index_data['Date'].iloc[-1].strftime('%Y-%m-%d')
                })
            
            # Add sector analysis
            for sector, data in sector_analysis.items():
                report_data.append({
                    'Category': 'Sector Analysis',
                    'Metric': f'{sector} - Average Return',
                    'Value': f"{data['avg_return']:.2f}%",
                    'Date': datetime.now().strftime('%Y-%m-%d')
                })
            
            # Add market breadth
            if market_breadth:
                report_data.append({
                    'Category': 'Market Breadth',
                    'Metric': 'Advance/Decline Ratio',
                    'Value': f"{market_breadth['advance_decline_ratio']:.2f}",
                    'Date': market_breadth['date']
                })
            
            # Add volatility analysis
            if volatility_analysis:
                report_data.append({
                    'Category': 'Volatility Analysis',
                    'Metric': 'Annualized Volatility',
                    'Value': f"{volatility_analysis['annualized_volatility']:.2f}%",
                    'Date': datetime.now().strftime('%Y-%m-%d')
                })
            
            # Convert to DataFrame and export
            report_df = pd.DataFrame(report_data)
            report_df.to_csv(filename, index=False)
            
            self.logger.info(f"Nifty 50 analysis report exported to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error exporting report: {e}")
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of available Nifty 50 data"""
        try:
            conn = sqlite3.connect(self.database_path)
            
            # Get data counts
            cursor = conn.execute("SELECT COUNT(DISTINCT symbol) FROM nifty50_data")
            symbols_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM nifty50_data")
            total_records = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT MIN(date), MAX(date) FROM nifty50_data")
            date_range = cursor.fetchone()
            
            conn.close()
            
            return {
                'symbols_count': symbols_count,
                'total_records': total_records,
                'date_range': {
                    'start': date_range[0] if date_range[0] else None,
                    'end': date_range[1] if date_range[1] else None
                },
                'database_path': self.database_path
            }
            
        except Exception as e:
            self.logger.error(f"Error getting data summary: {e}")
            return {}
