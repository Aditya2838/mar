"""
Intraday Trading Analysis Module
Provides intraday analysis and trading signals for Nifty 50 stocks
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import ta
from config import Config

class IntradayTradingAnalysis:
    """
    Intraday trading analysis and prediction for Nifty 50 stocks
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.nifty_symbols = Config.NIFTY_50_SYMBOLS
        self.market_hours = Config.INDIAN_MARKET_HOURS
        
    def get_intraday_data(self, symbol: str, period: str = "1d", interval: str = "5m") -> pd.DataFrame:
        """
        Get intraday data for a specific stock
        
        Args:
            symbol (str): Stock symbol
            period (str): Data period (1d, 5d)
            interval (str): Data interval (1m, 5m, 15m, 30m, 1h)
        
        Returns:
            pd.DataFrame: Intraday OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                self.logger.warning(f"No intraday data found for {symbol}")
                return pd.DataFrame()
            
            # Add time-based features
            data['Time'] = data.index.time
            data['Hour'] = data.index.hour
            data['Minute'] = data.index.minute
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching intraday data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def calculate_intraday_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate intraday-specific technical indicators
        
        Args:
            data (pd.DataFrame): Intraday OHLCV data
        
        Returns:
            pd.DataFrame: Data with intraday indicators
        """
        try:
            if data.empty:
                return data
            
            # VWAP (Volume Weighted Average Price)
            data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
            
            # Price relative to VWAP
            data['Price_VWAP_Ratio'] = data['Close'] / data['VWAP']
            
            # Intraday momentum
            data['Intraday_Momentum'] = (data['Close'] - data['Open']) / data['Open'] * 100
            
            # Volume intensity
            data['Volume_Intensity'] = data['Volume'] / data['Volume'].rolling(window=20).mean()
            
            # Price volatility (rolling)
            data['Price_Volatility'] = data['Close'].rolling(window=10).std()
            
            # RSI for intraday
            data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
            
            # MACD for intraday
            macd = ta.trend.MACD(data['Close'])
            data['MACD'] = macd.macd()
            data['MACD_Signal'] = macd.macd_signal()
            data['MACD_Histogram'] = macd.macd_diff()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(data['Close'])
            data['BB_Upper'] = bb.bollinger_hband()
            data['BB_Middle'] = bb.bollinger_mavg()
            data['BB_Lower'] = bb.bollinger_lband()
            data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
            
            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'])
            data['Stoch_K'] = stoch.stoch()
            data['Stoch_D'] = stoch.stoch_signal()
            
            # Williams %R
            data['Williams_R'] = ta.momentum.WilliamsRIndicator(data['High'], data['Low'], data['Close']).williams_r()
            
            # ATR (Average True Range)
            data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
            
            # Support and Resistance levels
            data['Support'] = data['Low'].rolling(window=20).min()
            data['Resistance'] = data['High'].rolling(window=20).max()
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error calculating intraday indicators: {str(e)}")
            return data
    
    def generate_intraday_signals(self, data: pd.DataFrame) -> Dict[str, str]:
        """
        Generate intraday trading signals (BUY/SELL/HOLD)
        
        Args:
            data (pd.DataFrame): Data with technical indicators
        
        Returns:
            Dict[str, str]: Trading signals with confidence
        """
        try:
            if data.empty or len(data) < 20:
                return {'signal': 'HOLD', 'confidence': 0, 'reason': 'Insufficient data'}
            
            current_price = data['Close'].iloc[-1]
            vwap = data['VWAP'].iloc[-1]
            rsi = data['RSI'].iloc[-1]
            macd = data['MACD'].iloc[-1]
            macd_signal = data['MACD_Signal'].iloc[-1]
            bb_position = data['BB_Position'].iloc[-1]
            stoch_k = data['Stoch_K'].iloc[-1]
            williams_r = data['Williams_R'].iloc[-1]
            
            # Initialize signal components
            buy_signals = 0
            sell_signals = 0
            total_signals = 0
            reasons = []
            
            # Price vs VWAP
            if current_price > vwap * 1.01:
                buy_signals += 1
                reasons.append("Price above VWAP")
            elif current_price < vwap * 0.99:
                sell_signals += 1
                reasons.append("Price below VWAP")
            total_signals += 1
            
            # RSI signals
            if rsi < 30:
                buy_signals += 1
                reasons.append("RSI oversold")
            elif rsi > 70:
                sell_signals += 1
                reasons.append("RSI overbought")
            total_signals += 1
            
            # MACD signals
            if macd > macd_signal and macd > 0:
                buy_signals += 1
                reasons.append("MACD bullish crossover")
            elif macd < macd_signal and macd < 0:
                sell_signals += 1
                reasons.append("MACD bearish crossover")
            total_signals += 1
            
            # Bollinger Bands
            if bb_position < 0.2:
                buy_signals += 1
                reasons.append("Price near lower Bollinger Band")
            elif bb_position > 0.8:
                sell_signals += 1
                reasons.append("Price near upper Bollinger Band")
            total_signals += 1
            
            # Stochastic
            if stoch_k < 20:
                buy_signals += 1
                reasons.append("Stochastic oversold")
            elif stoch_k > 80:
                sell_signals += 1
                reasons.append("Stochastic overbought")
            total_signals += 1
            
            # Williams %R
            if williams_r < -80:
                buy_signals += 1
                reasons.append("Williams %R oversold")
            elif williams_r > -20:
                sell_signals += 1
                reasons.append("Williams %R overbought")
            total_signals += 1
            
            # Calculate signal strength
            buy_strength = buy_signals / total_signals
            sell_strength = sell_signals / total_signals
            
            # Generate final signal
            if buy_strength > 0.5:
                signal = 'BUY'
                confidence = int(buy_strength * 100)
            elif sell_strength > 0.5:
                signal = 'SELL'
                confidence = int(sell_strength * 100)
            else:
                signal = 'HOLD'
                confidence = int(max(buy_strength, sell_strength) * 100)
            
            return {
                'signal': signal,
                'confidence': confidence,
                'reason': '; '.join(reasons[:3]),  # Top 3 reasons
                'buy_strength': buy_strength,
                'sell_strength': sell_strength
            }
            
        except Exception as e:
            self.logger.error(f"Error generating intraday signals: {str(e)}")
            return {'signal': 'HOLD', 'confidence': 0, 'reason': f'Error: {str(e)}'}
    
    def analyze_market_timing(self, data: pd.DataFrame) -> Dict[str, str]:
        """
        Analyze optimal market timing for intraday trading
        
        Args:
            data (pd.DataFrame): Intraday data with time information
        
        Returns:
            Dict[str, str]: Market timing analysis
        """
        try:
            if data.empty:
                return {'timing': 'Unknown', 'reason': 'No data available'}
            
            # Group by hour and analyze
            hourly_analysis = data.groupby(data.index.hour).agg({
                'Volume': 'mean',
                'Intraday_Momentum': 'mean',
                'Price_Volatility': 'mean'
            }).round(2)
            
            # Find best trading hours
            best_volume_hour = hourly_analysis['Volume'].idxmax()
            best_momentum_hour = hourly_analysis['Intraday_Momentum'].idxmax()
            best_volatility_hour = hourly_analysis['Price_Volatility'].idxmax()
            
            # Market timing recommendations
            current_hour = datetime.now().hour
            
            if 9 <= current_hour <= 10:
                timing = "Market Open - High Volatility"
                reason = "Best for momentum trading"
            elif 10 <= current_hour <= 14:
                timing = "Mid Session - Stable"
                reason = "Good for trend following"
            elif 14 <= current_hour <= 15:
                timing = "Pre-Close - Volatile"
                reason = "Watch for closing momentum"
            else:
                timing = "Market Closed"
                reason = "No trading activity"
            
            return {
                'timing': timing,
                'reason': reason,
                'best_volume_hour': best_volume_hour,
                'best_momentum_hour': best_momentum_hour,
                'best_volatility_hour': best_volatility_hour,
                'current_hour': current_hour
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market timing: {str(e)}")
            return {'timing': 'Unknown', 'reason': f'Error: {str(e)}'}
    
    def get_intraday_predictions(self, symbol: str, prediction_horizon: int = 4) -> Dict[str, any]:
        """
        Get intraday price predictions for a stock
        
        Args:
            symbol (str): Stock symbol
            prediction_horizon (int): Number of intervals to predict
        
        Returns:
            Dict[str, any]: Price predictions and analysis
        """
        try:
            # Get intraday data
            data = self.get_intraday_data(symbol, period="5d", interval="15m")
            if data.empty:
                return {'error': 'No data available'}
            
            # Calculate indicators
            data = self.calculate_intraday_indicators(data)
            
            # Generate signals
            signals = self.generate_intraday_signals(data)
            
            # Market timing
            timing = self.analyze_market_timing(data)
            
            # Simple price prediction based on momentum
            current_price = data['Close'].iloc[-1]
            momentum = data['Intraday_Momentum'].iloc[-1]
            volatility = data['Price_Volatility'].iloc[-1]
            
            # Predict next few intervals
            predictions = []
            for i in range(1, prediction_horizon + 1):
                predicted_change = momentum * 0.1 * i  # Momentum-based prediction
                predicted_price = current_price * (1 + predicted_change / 100)
                predictions.append({
                    'interval': i,
                    'predicted_price': round(predicted_price, 2),
                    'predicted_change': round(predicted_change, 2)
                })
            
            # Get additional data
            vwap = data['VWAP'].iloc[-1] if 'VWAP' in data.columns else current_price
            volume = data['Volume'].iloc[-1] if 'Volume' in data.columns else 0
            
            return {
                'symbol': symbol,
                'current_price': round(current_price, 2),
                'current_momentum': round(momentum, 2),
                'current_volatility': round(volatility, 4),
                'vwap': round(vwap, 2),
                'volume': volume,
                'signals': signals,
                'timing': timing,
                'predictions': predictions,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            self.logger.error(f"Error getting intraday predictions for {symbol}: {str(e)}")
            return {'error': str(e)}
    
    def get_nifty50_intraday_summary(self) -> Dict[str, any]:
        """
        Get intraday summary for all Nifty 50 stocks
        
        Returns:
            Dict[str, any]: Summary of all stocks
        """
        try:
            summary = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_stocks': len(self.nifty_symbols),
                'stocks_analyzed': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'hold_signals': 0,
                'stock_details': []
            }
            
            for symbol in self.nifty_symbols[:10]:  # Analyze first 10 for performance
                try:
                    stock_analysis = self.get_intraday_predictions(symbol)
                    if 'error' not in stock_analysis:
                        summary['stocks_analyzed'] += 1
                        
                        signal = stock_analysis['signals']['signal']
                        if signal == 'BUY':
                            summary['buy_signals'] += 1
                        elif signal == 'SELL':
                            summary['sell_signals'] += 1
                        else:
                            summary['hold_signals'] += 1
                        
                        summary['stock_details'].append({
                            'symbol': symbol,
                            'current_price': stock_analysis['current_price'],
                            'signal': signal,
                            'confidence': stock_analysis['signals']['confidence']
                        })
                        
                except Exception as e:
                    self.logger.warning(f"Error analyzing {symbol}: {str(e)}")
                    continue
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting Nifty 50 intraday summary: {str(e)}")
            return {'error': str(e)}
