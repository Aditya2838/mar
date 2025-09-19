"""
Trading Strategies Module
Provides various algorithmic trading strategies and signal generation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

class TradingStrategy:
    """
    Base class for trading strategies
    """
    
    def __init__(self, symbol: str, initial_capital: float = 10000):
        """
        Initialize trading strategy
        
        Args:
            symbol (str): Stock symbol
            initial_capital (float): Initial capital for backtesting
        """
        self.symbol = symbol.upper()
        self.initial_capital = initial_capital
        self.logger = logging.getLogger(__name__)
        self.positions = []
        self.trades = []
        self.capital = initial_capital
        self.shares = 0
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals
        
        Args:
            data (pd.DataFrame): Stock data
        
        Returns:
            pd.DataFrame: Data with signals
        """
        raise NotImplementedError("Subclasses must implement generate_signals")
    
    def backtest(self, data: pd.DataFrame) -> Dict:
        """
        Backtest the strategy
        
        Args:
            data (pd.DataFrame): Historical stock data
        
        Returns:
            Dict: Backtest results
        """
        try:
            # Generate signals
            signals = self.generate_signals(data)
            
            # Reset portfolio
            self.capital = self.initial_capital
            self.shares = 0
            self.positions = []
            self.trades = []
            
            # Execute trades
            for i in range(1, len(signals)):
                current_price = signals['Close'].iloc[i]
                current_date = signals.index[i]
                
                # Check for buy signal
                if signals['Buy_Signal'].iloc[i] and self.capital > 0:
                    shares_to_buy = int(self.capital / current_price)
                    if shares_to_buy > 0:
                        cost = shares_to_buy * current_price
                        self.capital -= cost
                        self.shares += shares_to_buy
                        
                        self.trades.append({
                            'date': current_date,
                            'action': 'BUY',
                            'shares': shares_to_buy,
                            'price': current_price,
                            'cost': cost,
                            'capital': self.capital
                        })
                
                # Check for sell signal
                elif signals['Sell_Signal'].iloc[i] and self.shares > 0:
                    revenue = self.shares * current_price
                    self.capital += revenue
                    
                    self.trades.append({
                        'date': current_date,
                        'action': 'SELL',
                        'shares': self.shares,
                        'price': current_price,
                        'revenue': revenue,
                        'capital': self.capital
                    })
                    
                    self.shares = 0
            
            # Calculate final portfolio value
            final_value = self.capital + (self.shares * signals['Close'].iloc[-1])
            total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100
            
            # Calculate buy and hold return
            buy_hold_return = ((signals['Close'].iloc[-1] - signals['Close'].iloc[0]) / signals['Close'].iloc[0]) * 100
            
            return {
                'initial_capital': self.initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'buy_hold_return': buy_hold_return,
                'excess_return': total_return - buy_hold_return,
                'trades': self.trades,
                'total_trades': len(self.trades),
                'final_capital': self.capital,
                'final_shares': self.shares
            }
            
        except Exception as e:
            self.logger.error(f"Error in backtest: {str(e)}")
            return {}


class MovingAverageStrategy(TradingStrategy):
    """
    Moving Average Crossover Strategy
    """
    
    def __init__(self, symbol: str, short_period: int = 20, long_period: int = 50, initial_capital: float = 10000):
        """
        Initialize Moving Average Strategy
        
        Args:
            symbol (str): Stock symbol
            short_period (int): Short moving average period
            long_period (int): Long moving average period
            initial_capital (float): Initial capital
        """
        super().__init__(symbol, initial_capital)
        self.short_period = short_period
        self.long_period = long_period
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on moving average crossover
        
        Args:
            data (pd.DataFrame): Stock data
        
        Returns:
            pd.DataFrame: Data with signals
        """
        try:
            signals = data.copy()
            
            # Calculate moving averages
            signals[f'SMA_{self.short_period}'] = signals['Close'].rolling(window=self.short_period).mean()
            signals[f'SMA_{self.long_period}'] = signals['Close'].rolling(window=self.long_period).mean()
            
            # Generate signals
            signals['Buy_Signal'] = (
                (signals[f'SMA_{self.short_period}'] > signals[f'SMA_{self.long_period}']) &
                (signals[f'SMA_{self.short_period}'].shift(1) <= signals[f'SMA_{self.long_period}'].shift(1))
            )
            
            signals['Sell_Signal'] = (
                (signals[f'SMA_{self.short_period}'] < signals[f'SMA_{self.long_period}']) &
                (signals[f'SMA_{self.short_period}'].shift(1) >= signals[f'SMA_{self.long_period}'].shift(1))
            )
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating MA signals: {str(e)}")
            return data


class RSIStrategy(TradingStrategy):
    """
    RSI-based Trading Strategy
    """
    
    def __init__(self, symbol: str, rsi_period: int = 14, oversold: int = 30, overbought: int = 70, initial_capital: float = 10000):
        """
        Initialize RSI Strategy
        
        Args:
            symbol (str): Stock symbol
            rsi_period (int): RSI calculation period
            oversold (int): Oversold threshold
            overbought (int): Overbought threshold
            initial_capital (float): Initial capital
        """
        super().__init__(symbol, initial_capital)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on RSI
        
        Args:
            data (pd.DataFrame): Stock data
        
        Returns:
            pd.DataFrame: Data with signals
        """
        try:
            signals = data.copy()
            
            # Calculate RSI
            delta = signals['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            signals['RSI'] = 100 - (100 / (1 + rs))
            
            # Generate signals
            signals['Buy_Signal'] = signals['RSI'] < self.oversold
            signals['Sell_Signal'] = signals['RSI'] > self.overbought
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating RSI signals: {str(e)}")
            return data


class MACDStrategy(TradingStrategy):
    """
    MACD-based Trading Strategy
    """
    
    def __init__(self, symbol: str, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, initial_capital: float = 10000):
        """
        Initialize MACD Strategy
        
        Args:
            symbol (str): Stock symbol
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            signal_period (int): Signal line period
            initial_capital (float): Initial capital
        """
        super().__init__(symbol, initial_capital)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on MACD
        
        Args:
            data (pd.DataFrame): Stock data
        
        Returns:
            pd.DataFrame: Data with signals
        """
        try:
            signals = data.copy()
            
            # Calculate MACD
            ema_fast = signals['Close'].ewm(span=self.fast_period).mean()
            ema_slow = signals['Close'].ewm(span=self.slow_period).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=self.signal_period).mean()
            
            signals['MACD'] = macd_line
            signals['MACD_Signal'] = signal_line
            signals['MACD_Histogram'] = macd_line - signal_line
            
            # Generate signals
            signals['Buy_Signal'] = (
                (macd_line > signal_line) &
                (macd_line.shift(1) <= signal_line.shift(1))
            )
            
            signals['Sell_Signal'] = (
                (macd_line < signal_line) &
                (macd_line.shift(1) >= signal_line.shift(1))
            )
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating MACD signals: {str(e)}")
            return data


class BollingerBandsStrategy(TradingStrategy):
    """
    Bollinger Bands Trading Strategy
    """
    
    def __init__(self, symbol: str, period: int = 20, std_dev: float = 2, initial_capital: float = 10000):
        """
        Initialize Bollinger Bands Strategy
        
        Args:
            symbol (str): Stock symbol
            period (int): Moving average period
            std_dev (float): Standard deviation multiplier
            initial_capital (float): Initial capital
        """
        super().__init__(symbol, initial_capital)
        self.period = period
        self.std_dev = std_dev
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on Bollinger Bands
        
        Args:
            data (pd.DataFrame): Stock data
        
        Returns:
            pd.DataFrame: Data with signals
        """
        try:
            signals = data.copy()
            
            # Calculate Bollinger Bands
            sma = signals['Close'].rolling(window=self.period).mean()
            std = signals['Close'].rolling(window=self.period).std()
            
            signals['BB_Upper'] = sma + (std * self.std_dev)
            signals['BB_Middle'] = sma
            signals['BB_Lower'] = sma - (std * self.std_dev)
            
            # Generate signals
            signals['Buy_Signal'] = signals['Close'] < signals['BB_Lower']
            signals['Sell_Signal'] = signals['Close'] > signals['BB_Upper']
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating Bollinger Bands signals: {str(e)}")
            return data


class MeanReversionStrategy(TradingStrategy):
    """
    Mean Reversion Trading Strategy
    """
    
    def __init__(self, symbol: str, lookback_period: int = 20, std_threshold: float = 2.0, initial_capital: float = 10000):
        """
        Initialize Mean Reversion Strategy
        
        Args:
            symbol (str): Stock symbol
            lookback_period (int): Period for calculating mean and std
            std_threshold (float): Standard deviation threshold
            initial_capital (float): Initial capital
        """
        super().__init__(symbol, initial_capital)
        self.lookback_period = lookback_period
        self.std_threshold = std_threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals based on mean reversion
        
        Args:
            data (pd.DataFrame): Stock data
        
        Returns:
            pd.DataFrame: Data with signals
        """
        try:
            signals = data.copy()
            
            # Calculate rolling mean and standard deviation
            rolling_mean = signals['Close'].rolling(window=self.lookback_period).mean()
            rolling_std = signals['Close'].rolling(window=self.lookback_period).std()
            
            # Calculate z-score
            signals['Z_Score'] = (signals['Close'] - rolling_mean) / rolling_std
            
            # Generate signals
            signals['Buy_Signal'] = signals['Z_Score'] < -self.std_threshold
            signals['Sell_Signal'] = signals['Z_Score'] > self.std_threshold
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating mean reversion signals: {str(e)}")
            return data


class MultiStrategy(TradingStrategy):
    """
    Multi-Strategy Combination
    """
    
    def __init__(self, symbol: str, strategies: List[TradingStrategy], weights: List[float] = None, initial_capital: float = 10000):
        """
        Initialize Multi-Strategy
        
        Args:
            symbol (str): Stock symbol
            strategies (List[TradingStrategy]): List of strategies to combine
            weights (List[float]): Weights for each strategy (optional)
            initial_capital (float): Initial capital
        """
        super().__init__(symbol, initial_capital)
        self.strategies = strategies
        
        if weights is None:
            self.weights = [1.0 / len(strategies)] * len(strategies)
        else:
            self.weights = weights
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate combined signals from multiple strategies
        
        Args:
            data (pd.DataFrame): Stock data
        
        Returns:
            pd.DataFrame: Data with combined signals
        """
        try:
            combined_signals = data.copy()
            combined_signals['Buy_Signal'] = False
            combined_signals['Sell_Signal'] = False
            
            # Get signals from each strategy
            strategy_signals = []
            for strategy in self.strategies:
                signals = strategy.generate_signals(data)
                strategy_signals.append(signals)
            
            # Combine signals with weights
            for i in range(len(data)):
                buy_score = 0
                sell_score = 0
                
                for j, signals in enumerate(strategy_signals):
                    if signals['Buy_Signal'].iloc[i]:
                        buy_score += self.weights[j]
                    if signals['Sell_Signal'].iloc[i]:
                        sell_score += self.weights[j]
                
                # Set combined signals
                combined_signals['Buy_Signal'].iloc[i] = buy_score > 0.5
                combined_signals['Sell_Signal'].iloc[i] = sell_score > 0.5
            
            return combined_signals
            
        except Exception as e:
            self.logger.error(f"Error generating multi-strategy signals: {str(e)}")
            return data


class PortfolioOptimizer:
    """
    Portfolio optimization and risk management
    """
    
    def __init__(self, symbols: List[str], initial_capital: float = 100000):
        """
        Initialize Portfolio Optimizer
        
        Args:
            symbols (List[str]): List of stock symbols
            initial_capital (float): Initial capital
        """
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.logger = logging.getLogger(__name__)
    
    def calculate_optimal_weights(self, returns_data: pd.DataFrame, method: str = 'equal') -> Dict[str, float]:
        """
        Calculate optimal portfolio weights
        
        Args:
            returns_data (pd.DataFrame): Returns data for all symbols
            method (str): Optimization method ('equal', 'min_variance', 'max_sharpe')
        
        Returns:
            Dict[str, float]: Optimal weights for each symbol
        """
        try:
            if method == 'equal':
                # Equal weight allocation
                n_symbols = len(self.symbols)
                weights = {symbol: 1.0 / n_symbols for symbol in self.symbols}
                
            elif method == 'min_variance':
                # Minimum variance portfolio
                cov_matrix = returns_data.cov()
                inv_cov = np.linalg.inv(cov_matrix.values)
                ones = np.ones(len(self.symbols))
                
                weights_vector = inv_cov @ ones
                weights_vector = weights_vector / np.sum(weights_vector)
                
                weights = {symbol: weight for symbol, weight in zip(self.symbols, weights_vector)}
                
            elif method == 'max_sharpe':
                # Maximum Sharpe ratio portfolio
                mean_returns = returns_data.mean()
                cov_matrix = returns_data.cov()
                
                # Risk-free rate (assume 0 for simplicity)
                rf_rate = 0.0
                
                # Calculate optimal weights
                inv_cov = np.linalg.inv(cov_matrix.values)
                excess_returns = mean_returns - rf_rate
                
                weights_vector = inv_cov @ excess_returns
                weights_vector = weights_vector / np.sum(weights_vector)
                
                weights = {symbol: weight for symbol, weight in zip(self.symbols, weights_vector)}
                
            else:
                raise ValueError(f"Unknown optimization method: {method}")
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal weights: {str(e)}")
            return {symbol: 1.0 / len(self.symbols) for symbol in self.symbols}
    
    def calculate_portfolio_metrics(self, returns_data: pd.DataFrame, weights: Dict[str, float]) -> Dict:
        """
        Calculate portfolio performance metrics
        
        Args:
            returns_data (pd.DataFrame): Returns data
            weights (Dict[str, float]): Portfolio weights
        
        Returns:
            Dict: Portfolio metrics
        """
        try:
            # Calculate weighted returns
            weighted_returns = pd.Series(0.0, index=returns_data.index)
            for symbol in self.symbols:
                weighted_returns += returns_data[symbol] * weights[symbol]
            
            # Calculate metrics
            total_return = (1 + weighted_returns).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(returns_data)) - 1
            volatility = weighted_returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            max_drawdown = self._calculate_max_drawdown(weighted_returns)
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'weights': weights
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return {}
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()
        except Exception:
            return 0.0


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
    sample_data.set_index('Date', inplace=True)
    
    # Test Moving Average Strategy
    ma_strategy = MovingAverageStrategy("AAPL")
    ma_results = ma_strategy.backtest(sample_data)
    
    print("Moving Average Strategy Results:")
    print(f"Total Return: {ma_results.get('total_return', 0):.2f}%")
    print(f"Buy & Hold Return: {ma_results.get('buy_hold_return', 0):.2f}%")
    print(f"Total Trades: {ma_results.get('total_trades', 0)}")
    
    # Test RSI Strategy
    rsi_strategy = RSIStrategy("AAPL")
    rsi_results = rsi_strategy.backtest(sample_data)
    
    print("\nRSI Strategy Results:")
    print(f"Total Return: {rsi_results.get('total_return', 0):.2f}%")
    print(f"Buy & Hold Return: {rsi_results.get('buy_hold_return', 0):.2f}%")
    print(f"Total Trades: {rsi_results.get('total_trades', 0)}")
    
    # Test Portfolio Optimization
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    optimizer = PortfolioOptimizer(symbols)
    
    # Create sample returns data
    returns_data = pd.DataFrame({
        'AAPL': np.random.randn(252) * 0.02,
        'GOOGL': np.random.randn(252) * 0.025,
        'MSFT': np.random.randn(252) * 0.018
    })
    
    weights = optimizer.calculate_optimal_weights(returns_data, method='equal')
    metrics = optimizer.calculate_portfolio_metrics(returns_data, weights)
    
    print("\nPortfolio Optimization Results:")
    print(f"Annual Return: {metrics.get('annual_return', 0):.2%}")
    print(f"Volatility: {metrics.get('volatility', 0):.2%}")
    print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
