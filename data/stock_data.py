"""
Stock Data Management Module
Handles fetching, storing, and managing stock market data
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import os
from typing import Dict, List, Optional, Tuple
import logging

class StockData:
    """
    Main class for handling stock market data operations
    """
    
    def __init__(self, symbol: str, database_path: str = "stock_data.db"):
        """
        Initialize StockData with a stock symbol
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL', 'GOOGL')
            database_path (str): Path to SQLite database
        """
        self.symbol = symbol.upper()
        self.database_path = database_path
        self.ticker = yf.Ticker(symbol)
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Create stock_data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date)
            )
        ''')
        
        # Create technical_indicators table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS technical_indicators (
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_historical_data(self, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical stock data
        
        Args:
            period (str): Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval (str): Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            pd.DataFrame: Historical stock data
        """
        try:
            data = self.ticker.history(period=period, interval=interval)
            if data.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")
            
            # Reset index to make date a column
            data.reset_index(inplace=True)
            data['Symbol'] = self.symbol
            
            # Store in database
            self._store_data(data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {self.symbol}: {str(e)}")
            raise
    
    def get_real_time_data(self) -> Dict:
        """
        Get real-time stock data
        
        Returns:
            Dict: Current stock information
        """
        try:
            info = self.ticker.info
            current_price = self.ticker.history(period="1d")
            
            if not current_price.empty:
                latest = current_price.iloc[-1]
                return {
                    'symbol': self.symbol,
                    'current_price': latest['Close'],
                    'change': latest['Close'] - latest['Open'],
                    'change_percent': ((latest['Close'] - latest['Open']) / latest['Open']) * 100,
                    'volume': latest['Volume'],
                    'high': latest['High'],
                    'low': latest['Low'],
                    'open': latest['Open'],
                    'market_cap': info.get('marketCap', 'N/A'),
                    'pe_ratio': info.get('trailingPE', 'N/A'),
                    'dividend_yield': info.get('dividendYield', 'N/A'),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                raise ValueError(f"No real-time data available for {self.symbol}")
                
        except Exception as e:
            self.logger.error(f"Error fetching real-time data for {self.symbol}: {str(e)}")
            raise
    
    def get_company_info(self) -> Dict:
        """
        Get company information
        
        Returns:
            Dict: Company information
        """
        try:
            info = self.ticker.info
            return {
                'symbol': self.symbol,
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'employees': info.get('fullTimeEmployees', 'N/A'),
                'website': info.get('website', 'N/A'),
                'description': info.get('longBusinessSummary', 'N/A')[:500] + '...' if info.get('longBusinessSummary') else 'N/A'
            }
        except Exception as e:
            self.logger.error(f"Error fetching company info for {self.symbol}: {str(e)}")
            raise
    
    def get_financial_statements(self) -> Dict:
        """
        Get financial statements
        
        Returns:
            Dict: Financial statements data
        """
        try:
            return {
                'balance_sheet': self.ticker.balance_sheet.to_dict() if hasattr(self.ticker, 'balance_sheet') else {},
                'income_statement': self.ticker.income_stmt.to_dict() if hasattr(self.ticker, 'income_stmt') else {},
                'cash_flow': self.ticker.cashflow.to_dict() if hasattr(self.ticker, 'cashflow') else {}
            }
        except Exception as e:
            self.logger.error(f"Error fetching financial statements for {self.symbol}: {str(e)}")
            return {}
    
    def _store_data(self, data: pd.DataFrame):
        """Store stock data in SQLite database"""
        conn = sqlite3.connect(self.database_path)
        
        # Prepare data for insertion
        data_to_insert = []
        for _, row in data.iterrows():
            data_to_insert.append((
                self.symbol,
                row['Date'].strftime('%Y-%m-%d'),
                row.get('Open', None),
                row.get('High', None),
                row.get('Low', None),
                row.get('Close', None),
                row.get('Volume', None)
            ))
        
        # Insert data
        cursor = conn.cursor()
        cursor.executemany('''
            INSERT OR REPLACE INTO stock_data 
            (symbol, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', data_to_insert)
        
        conn.commit()
        conn.close()
    
    def get_stored_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Retrieve stored data from database
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
        
        Returns:
            pd.DataFrame: Stored stock data
        """
        conn = sqlite3.connect(self.database_path)
        
        query = "SELECT * FROM stock_data WHERE symbol = ?"
        params = [self.symbol]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date"
        
        data = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if not data.empty:
            data['Date'] = pd.to_datetime(data['date'])
            data.set_index('Date', inplace=True)
        
        return data
    
    def get_multiple_symbols(self, symbols: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols
        
        Args:
            symbols (List[str]): List of stock symbols
            period (str): Data period
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with symbol as key and data as value
        """
        results = {}
        
        for symbol in symbols:
            try:
                stock_data = StockData(symbol, self.database_path)
                results[symbol] = stock_data.get_historical_data(period)
            except Exception as e:
                self.logger.warning(f"Could not fetch data for {symbol}: {str(e)}")
                continue
        
        return results
    
    def get_market_data(self, symbols: List[str] = None) -> pd.DataFrame:
        """
        Get market data for popular stocks if no symbols provided
        
        Args:
            symbols (List[str]): List of symbols, defaults to major indices
        
        Returns:
            pd.DataFrame: Market data
        """
        if symbols is None:
            symbols = ['^GSPC', '^DJI', '^IXIC', '^RUT']  # S&P 500, Dow Jones, NASDAQ, Russell 2000
        
        market_data = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d")
                if not data.empty:
                    latest = data.iloc[-1]
                    market_data.append({
                        'symbol': symbol,
                        'close': latest['Close'],
                        'change': latest['Close'] - latest['Open'],
                        'change_percent': ((latest['Close'] - latest['Open']) / latest['Open']) * 100,
                        'volume': latest['Volume']
                    })
            except Exception as e:
                self.logger.warning(f"Could not fetch market data for {symbol}: {str(e)}")
                continue
        
        return pd.DataFrame(market_data)
    
    def export_data(self, filename: str, format: str = 'csv'):
        """
        Export stock data to file
        
        Args:
            filename (str): Output filename
            format (str): Export format ('csv', 'excel', 'json')
        """
        data = self.get_historical_data()
        
        if format.lower() == 'csv':
            data.to_csv(filename, index=False)
        elif format.lower() == 'excel':
            data.to_excel(filename, index=False)
        elif format.lower() == 'json':
            data.to_json(filename, orient='records')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Data exported to {filename}")
    
    def get_data_summary(self) -> Dict:
        """
        Get summary statistics for the stock data
        
        Returns:
            Dict: Summary statistics
        """
        data = self.get_historical_data()
        
        if data.empty:
            return {}
        
        summary = {
            'symbol': self.symbol,
            'data_points': len(data),
            'date_range': {
                'start': data['Date'].min().strftime('%Y-%m-%d'),
                'end': data['Date'].max().strftime('%Y-%m-%d')
            },
            'price_stats': {
                'current_price': data['Close'].iloc[-1],
                'highest_price': data['High'].max(),
                'lowest_price': data['Low'].min(),
                'average_price': data['Close'].mean(),
                'price_volatility': data['Close'].std()
            },
            'volume_stats': {
                'total_volume': data['Volume'].sum(),
                'average_volume': data['Volume'].mean(),
                'max_volume': data['Volume'].max()
            },
            'returns': {
                'total_return': ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100,
                'annualized_return': self._calculate_annualized_return(data)
            }
        }
        
        return summary
    
    def _calculate_annualized_return(self, data: pd.DataFrame) -> float:
        """Calculate annualized return"""
        if len(data) < 2:
            return 0.0
        
        total_days = (data['Date'].max() - data['Date'].min()).days
        if total_days == 0:
            return 0.0
        
        total_return = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]
        annualized_return = ((1 + total_return) ** (365 / total_days) - 1) * 100
        
        return annualized_return


# Example usage and testing
if __name__ == "__main__":
    # Test the StockData class
    stock = StockData("AAPL")
    
    # Get historical data
    data = stock.get_historical_data(period="6mo")
    print(f"Fetched {len(data)} data points for AAPL")
    
    # Get real-time data
    real_time = stock.get_real_time_data()
    print(f"Current price: ${real_time['current_price']:.2f}")
    
    # Get company info
    info = stock.get_company_info()
    print(f"Company: {info['name']}")
    
    # Get summary
    summary = stock.get_data_summary()
    print(f"Total return: {summary['returns']['total_return']:.2f}%")
