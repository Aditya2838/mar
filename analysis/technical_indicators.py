"""
Technical Analysis Module
Provides various technical indicators and analysis tools for stock market data
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Tuple, Optional
import logging

class TechnicalAnalysis:
    """
    Technical analysis tools for stock market data
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize TechnicalAnalysis with stock data
        
        Args:
            data (pd.DataFrame): Stock data with OHLCV columns
        """
        self.data = data.copy()
        self.logger = logging.getLogger(__name__)
        
        # Ensure required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Set date as index if not already
        if 'Date' in self.data.columns:
            self.data.set_index('Date', inplace=True)
    
    def calculate_rsi(self, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            period (int): RSI period (default: 14)
        
        Returns:
            pd.Series: RSI values
        """
        try:
            rsi = ta.momentum.RSIIndicator(self.data['Close'], window=period)
            return rsi.rsi()
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series(dtype=float)
    
    def calculate_macd(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            signal_period (int): Signal line period
        
        Returns:
            Dict[str, pd.Series]: MACD line, signal line, and histogram
        """
        try:
            macd = ta.trend.MACD(
                self.data['Close'], 
                window_fast=fast_period, 
                window_slow=slow_period, 
                window_sign=signal_period
            )
            
            return {
                'macd_line': macd.macd(),
                'signal_line': macd.macd_signal(),
                'histogram': macd.macd_diff()
            }
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {str(e)}")
            return {'macd_line': pd.Series(), 'signal_line': pd.Series(), 'histogram': pd.Series()}
    
    def calculate_bollinger_bands(self, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Args:
            period (int): Moving average period
            std_dev (float): Standard deviation multiplier
        
        Returns:
            Dict[str, pd.Series]: Upper band, middle band (SMA), lower band
        """
        try:
            bb = ta.volatility.BollingerBands(
                self.data['Close'], 
                window=period, 
                window_dev=std_dev
            )
            
            return {
                'upper_band': bb.bollinger_hband(),
                'middle_band': bb.bollinger_mavg(),
                'lower_band': bb.bollinger_lband()
            }
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            return {'upper_band': pd.Series(), 'middle_band': pd.Series(), 'lower_band': pd.Series()}
    
    def calculate_moving_averages(self, periods: List[int] = [20, 50, 200]) -> Dict[str, pd.Series]:
        """
        Calculate Simple Moving Averages
        
        Args:
            periods (List[int]): List of periods to calculate
        
        Returns:
            Dict[str, pd.Series]: Moving averages for each period
        """
        try:
            moving_averages = {}
            for period in periods:
                sma = ta.trend.SMAIndicator(self.data['Close'], window=period)
                moving_averages[f'SMA_{period}'] = sma.sma_indicator()
            
            return moving_averages
        except Exception as e:
            self.logger.error(f"Error calculating moving averages: {str(e)}")
            return {}
    
    def calculate_exponential_moving_averages(self, periods: List[int] = [12, 26]) -> Dict[str, pd.Series]:
        """
        Calculate Exponential Moving Averages
        
        Args:
            periods (List[int]): List of periods to calculate
        
        Returns:
            Dict[str, pd.Series]: EMAs for each period
        """
        try:
            emas = {}
            for period in periods:
                ema = ta.trend.EMAIndicator(self.data['Close'], window=period)
                emas[f'EMA_{period}'] = ema.ema_indicator()
            
            return emas
        except Exception as e:
            self.logger.error(f"Error calculating EMAs: {str(e)}")
            return {}
    
    def calculate_stochastic(self, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator
        
        Args:
            k_period (int): %K period
            d_period (int): %D period
        
        Returns:
            Dict[str, pd.Series]: %K and %D values
        """
        try:
            stoch = ta.momentum.StochasticOscillator(
                self.data['High'], 
                self.data['Low'], 
                self.data['Close'],
                window=k_period,
                smooth_window=d_period
            )
            
            return {
                'k_percent': stoch.stoch(),
                'd_percent': stoch.stoch_signal()
            }
        except Exception as e:
            self.logger.error(f"Error calculating Stochastic: {str(e)}")
            return {'k_percent': pd.Series(), 'd_percent': pd.Series()}
    
    def calculate_stochastic(self, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator
        
        Args:
            k_period (int): %K period
            d_period (int): %D period
        
        Returns:
            Dict[str, pd.Series]: %K and %D values
        """
        try:
            stoch = ta.momentum.StochasticOscillator(
                self.data['High'], 
                self.data['Low'], 
                self.data['Close'],
                window=k_period,
                smooth_window=d_period
            )
            return {
                '%K': stoch.stoch(),
                '%D': stoch.stoch_signal()
            }
        except Exception as e:
            self.logger.error(f"Error calculating Stochastic: {str(e)}")
            return {'%K': pd.Series(dtype=float), '%D': pd.Series(dtype=float)}
    
    def calculate_williams_r(self, period: int = 14) -> pd.Series:
        """
        Calculate Williams %R
        
        Args:
            period (int): Williams %R period
        
        Returns:
            pd.Series: Williams %R values
        """
        try:
            williams_r = ta.momentum.WilliamsRIndicator(
                self.data['High'], 
                self.data['Low'], 
                self.data['Close'],
                window=period
            )
            return williams_r.williams_r()
        except Exception as e:
            self.logger.error(f"Error calculating Williams %R: {str(e)}")
            return pd.Series(dtype=float)
    
    def calculate_atr(self, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR)
        
        Args:
            period (int): ATR period
        
        Returns:
            pd.Series: ATR values
        """
        try:
            atr = ta.volatility.AverageTrueRange(
                self.data['High'], 
                self.data['Low'], 
                self.data['Close'],
                window=period
            )
            return atr.average_true_range()
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {str(e)}")
            return pd.Series(dtype=float)
    
    def calculate_volume_indicators(self) -> Dict[str, pd.Series]:
        """
        Calculate volume-based indicators
        
        Returns:
            Dict[str, pd.Series]: Volume indicators
        """
        try:
            # On Balance Volume (OBV)
            obv = ta.volume.OnBalanceVolumeIndicator(self.data['Close'], self.data['Volume'])
            
            # Volume Weighted Average Price (VWAP)
            vwap = ta.volume.VolumeWeightedAveragePrice(
                self.data['High'], 
                self.data['Low'], 
                self.data['Close'], 
                self.data['Volume']
            )
            
            # Accumulation/Distribution Line
            adl = ta.volume.AccDistIndexIndicator(
                self.data['High'], 
                self.data['Low'], 
                self.data['Close'], 
                self.data['Volume']
            )
            
            return {
                'obv': obv.on_balance_volume(),
                'vwap': vwap.volume_weighted_average_price(),
                'adl': adl.acc_dist_index()
            }
        except Exception as e:
            self.logger.error(f"Error calculating volume indicators: {str(e)}")
            return {}
    
    def detect_support_resistance(self, window: int = 20, threshold: float = 0.02) -> Dict[str, List[float]]:
        """
        Detect support and resistance levels
        
        Args:
            window (int): Window for local extrema detection
            threshold (float): Threshold for level significance
        
        Returns:
            Dict[str, List[float]]: Support and resistance levels
        """
        try:
            highs = self.data['High'].rolling(window=window, center=True).max()
            lows = self.data['Low'].rolling(window=window, center=True).min()
            
            # Find resistance levels (local highs)
            resistance_levels = []
            for i in range(window, len(self.data) - window):
                if self.data['High'].iloc[i] == highs.iloc[i]:
                    resistance_levels.append(self.data['High'].iloc[i])
            
            # Find support levels (local lows)
            support_levels = []
            for i in range(window, len(self.data) - window):
                if self.data['Low'].iloc[i] == lows.iloc[i]:
                    support_levels.append(self.data['Low'].iloc[i])
            
            # Filter levels by threshold
            resistance_levels = list(set([level for level in resistance_levels 
                                       if level > self.data['Close'].mean() * (1 + threshold)]))
            support_levels = list(set([level for level in support_levels 
                                     if level < self.data['Close'].mean() * (1 - threshold)]))
            
            return {
                'support_levels': sorted(support_levels),
                'resistance_levels': sorted(resistance_levels)
            }
        except Exception as e:
            self.logger.error(f"Error detecting support/resistance: {str(e)}")
            return {'support_levels': [], 'resistance_levels': []}
    
    def calculate_fibonacci_retracements(self, start_date: str = None, end_date: str = None) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement levels
        
        Args:
            start_date (str): Start date for swing high/low calculation
            end_date (str): End date for swing high/low calculation
        
        Returns:
            Dict[str, float]: Fibonacci retracement levels
        """
        try:
            if start_date and end_date:
                data_subset = self.data[start_date:end_date]
            else:
                data_subset = self.data
            
            swing_high = data_subset['High'].max()
            swing_low = data_subset['Low'].min()
            price_range = swing_high - swing_low
            
            return {
                'swing_high': swing_high,
                'swing_low': swing_low,
                'fib_0': swing_low,
                'fib_236': swing_low + 0.236 * price_range,
                'fib_382': swing_low + 0.382 * price_range,
                'fib_500': swing_low + 0.500 * price_range,
                'fib_618': swing_low + 0.618 * price_range,
                'fib_786': swing_low + 0.786 * price_range,
                'fib_100': swing_high
            }
        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci retracements: {str(e)}")
            return {}
    
    def generate_signals(self) -> pd.DataFrame:
        """
        Generate trading signals based on technical indicators
        
        Returns:
            pd.DataFrame: DataFrame with signals
        """
        try:
            signals = self.data.copy()
            
            # Calculate indicators
            rsi = self.calculate_rsi()
            macd_data = self.calculate_macd()
            bb_data = self.calculate_bollinger_bands()
            sma_data = self.calculate_moving_averages()
            
            # Add indicators to signals DataFrame
            signals['RSI'] = rsi
            signals['MACD'] = macd_data['macd_line']
            signals['MACD_Signal'] = macd_data['signal_line']
            signals['BB_Upper'] = bb_data['upper_band']
            signals['BB_Lower'] = bb_data['lower_band']
            signals['SMA_20'] = sma_data.get('SMA_20', pd.Series())
            signals['SMA_50'] = sma_data.get('SMA_50', pd.Series())
            
            # Generate buy/sell signals
            signals['Buy_Signal'] = (
                (signals['RSI'] < 30) &  # Oversold
                (signals['Close'] < signals['BB_Lower']) &  # Below lower Bollinger Band
                (signals['Close'] > signals['SMA_20'])  # Above 20-day SMA
            )
            
            signals['Sell_Signal'] = (
                (signals['RSI'] > 70) &  # Overbought
                (signals['Close'] > signals['BB_Upper']) &  # Above upper Bollinger Band
                (signals['Close'] < signals['SMA_20'])  # Below 20-day SMA
            )
            
            # MACD crossover signals
            signals['MACD_Buy'] = (
                (signals['MACD'] > signals['MACD_Signal']) & 
                (signals['MACD'].shift(1) <= signals['MACD_Signal'].shift(1))
            )
            
            signals['MACD_Sell'] = (
                (signals['MACD'] < signals['MACD_Signal']) & 
                (signals['MACD'].shift(1) >= signals['MACD_Signal'].shift(1))
            )
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return pd.DataFrame()
    
    def calculate_all_indicators(self) -> pd.DataFrame:
        """
        Calculate all technical indicators and add them to the data
        
        Returns:
            pd.DataFrame: Data with all technical indicators added
        """
        try:
            # Calculate RSI
            self.data['RSI'] = self.calculate_rsi()
            
            # Calculate MACD
            macd_data = self.calculate_macd()
            self.data['MACD'] = macd_data['macd_line']
            self.data['MACD_Signal'] = macd_data['signal_line']
            self.data['MACD_Histogram'] = macd_data['histogram']
            
            # Calculate Bollinger Bands
            bb_data = self.calculate_bollinger_bands()
            self.data['BB_Upper'] = bb_data['upper_band']
            self.data['BB_Middle'] = bb_data['middle_band']
            self.data['BB_Lower'] = bb_data['lower_band']
            self.data['BB_Position'] = (self.data['Close'] - bb_data['lower_band']) / (bb_data['upper_band'] - bb_data['lower_band'])
            
            # Calculate Moving Averages
            ma_data = self.calculate_moving_averages()
            for period, values in ma_data.items():
                self.data[period] = values
            
            # Calculate Exponential Moving Averages
            ema_data = self.calculate_exponential_moving_averages()
            for period, values in ema_data.items():
                self.data[period] = values
            
            # Calculate Stochastic
            stoch_data = self.calculate_stochastic()
            self.data['Stoch_K'] = stoch_data['%K']
            self.data['Stoch_D'] = stoch_data['%D']
            
            # Calculate Williams %R
            self.data['Williams_R'] = self.calculate_williams_r()
            
            # Calculate ATR
            self.data['ATR'] = self.calculate_atr()
            
            # Calculate Volume Indicators
            volume_data = self.calculate_volume_indicators()
            for indicator, values in volume_data.items():
                self.data[indicator] = values
            
            # Calculate additional derived indicators
            self.data['Returns'] = self.data['Close'].pct_change()
            self.data['Price_Change'] = self.data['Close'] - self.data['Close'].shift(1)
            self.data['Volatility'] = self.data['Returns'].rolling(window=20).std()
            
            # Calculate support and resistance
            support_resistance = self.detect_support_resistance()
            if not support_resistance.empty:
                self.data['Support'] = support_resistance['Support']
                self.data['Resistance'] = support_resistance['Resistance']
            
            return self.data
            
        except Exception as e:
            self.logger.error(f"Error calculating all indicators: {str(e)}")
            return self.data
    
    def get_technical_summary(self) -> Dict:
        """
        Get a summary of all technical indicators
        
        Returns:
            Dict: Technical analysis summary
        """
        try:
            latest_data = self.data.iloc[-1]
            
            # Calculate current indicators
            rsi = self.calculate_rsi().iloc[-1] if len(self.calculate_rsi()) > 0 else None
            macd_data = self.calculate_macd()
            macd_current = macd_data['macd_line'].iloc[-1] if len(macd_data['macd_line']) > 0 else None
            bb_data = self.calculate_bollinger_bands()
            
            # Determine trend
            sma_data = self.calculate_moving_averages()
            sma_20 = sma_data.get('SMA_20', pd.Series())
            sma_50 = sma_data.get('SMA_50', pd.Series())
            
            if len(sma_20) > 0 and len(sma_50) > 0:
                trend = "Bullish" if sma_20.iloc[-1] > sma_50.iloc[-1] else "Bearish"
            else:
                trend = "Neutral"
            
            # RSI interpretation
            rsi_signal = "Oversold" if rsi and rsi < 30 else "Overbought" if rsi and rsi > 70 else "Neutral"
            
            return {
                'current_price': latest_data['Close'],
                'trend': trend,
                'rsi': rsi,
                'rsi_signal': rsi_signal,
                'macd': macd_current,
                'bollinger_position': self._get_bollinger_position(latest_data, bb_data),
                'volume': latest_data['Volume'],
                'support_resistance': self.detect_support_resistance()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating technical summary: {str(e)}")
            return {}
    
    def generate_signals(self) -> pd.DataFrame:
        """
        Generate buy/sell signals based on technical indicators
        
        Returns:
            pd.DataFrame: Data with buy/sell signals
        """
        try:
            signals = self.data.copy()
            
            # RSI signals
            signals['RSI_Buy'] = (signals['RSI'] < 30) & (signals['RSI'].shift(1) >= 30)
            signals['RSI_Sell'] = (signals['RSI'] > 70) & (signals['RSI'].shift(1) <= 70)
            
            # MACD signals
            signals['MACD_Buy'] = (signals['MACD'] > signals['MACD_Signal']) & (signals['MACD'].shift(1) <= signals['MACD_Signal'].shift(1))
            signals['MACD_Sell'] = (signals['MACD'] < signals['MACD_Signal']) & (signals['MACD'].shift(1) >= signals['MACD_Signal'].shift(1))
            
            # Bollinger Bands signals
            signals['BB_Buy'] = signals['Close'] <= signals['BB_Lower']
            signals['BB_Sell'] = signals['Close'] >= signals['BB_Upper']
            
            # Stochastic signals
            signals['Stoch_Buy'] = (signals['Stoch_K'] < 20) & (signals['Stoch_K'].shift(1) >= 20)
            signals['Stoch_Sell'] = (signals['Stoch_K'] > 80) & (signals['Stoch_K'].shift(1) <= 80)
            
            # Combined signals
            signals['Buy_Signal'] = (
                signals['RSI_Buy'] | 
                signals['MACD_Buy'] | 
                signals['BB_Buy'] | 
                signals['Stoch_Buy']
            )
            
            signals['Sell_Signal'] = (
                signals['RSI_Sell'] | 
                signals['MACD_Sell'] | 
                signals['BB_Sell'] | 
                signals['Stoch_Sell']
            )
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return self.data
    
    def _get_bollinger_position(self, latest_data: pd.Series, bb_data: Dict[str, pd.Series]) -> str:
        """Get position relative to Bollinger Bands"""
        if len(bb_data['upper_band']) > 0 and len(bb_data['lower_band']) > 0:
            current_price = latest_data['Close']
            upper = bb_data['upper_band'].iloc[-1]
            lower = bb_data['lower_band'].iloc[-1]
            
            if current_price > upper:
                return "Above Upper Band"
            elif current_price < lower:
                return "Below Lower Band"
            else:
                return "Within Bands"
        return "Unknown"


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    sample_data = pd.DataFrame({
        'Date': dates,
        'Open': np.random.randn(len(dates)).cumsum() + 100,
        'High': np.random.randn(len(dates)).cumsum() + 102,
        'Low': np.random.randn(len(dates)).cumsum() + 98,
        'Close': np.random.randn(len(dates)).cumsum() + 100,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    # Initialize technical analysis
    ta = TechnicalAnalysis(sample_data)
    
    # Calculate indicators
    rsi = ta.calculate_rsi()
    macd = ta.calculate_macd()
    bb = ta.calculate_bollinger_bands()
    
    print("Technical Analysis Results:")
    print(f"RSI: {rsi.iloc[-1]:.2f}")
    print(f"MACD: {macd['macd_line'].iloc[-1]:.2f}")
    print(f"Bollinger Bands: Upper={bb['upper_band'].iloc[-1]:.2f}, Lower={bb['lower_band'].iloc[-1]:.2f}")
    
    # Generate signals
    signals = ta.generate_signals()
    print(f"Buy signals: {signals['Buy_Signal'].sum()}")
    print(f"Sell signals: {signals['Sell_Signal'].sum()}")
    
    # Get summary
    summary = ta.get_technical_summary()
    print(f"Trend: {summary.get('trend', 'Unknown')}")
    print(f"RSI Signal: {summary.get('rsi_signal', 'Unknown')}")
