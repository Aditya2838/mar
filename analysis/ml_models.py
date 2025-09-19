"""
Machine Learning Models for Stock Market Analysis
Provides various ML models for price prediction and pattern recognition
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PricePredictor:
    """
    Machine learning models for stock price prediction
    """
    
    def __init__(self, symbol: str, model_path: str = "models/"):
        """
        Initialize PricePredictor
        
        Args:
            symbol (str): Stock symbol
            model_path (str): Path to save/load models
        """
        self.symbol = symbol.upper()
        self.model_path = model_path
        self.logger = logging.getLogger(__name__)
        
        # Create models directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
        # Initialize models
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'svr': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        self.scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
        self.is_trained = False
        
    def prepare_features(self, data: pd.DataFrame, lookback_days: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for machine learning models
        
        Args:
            data (pd.DataFrame): Stock data
            lookback_days (int): Number of days to look back for features
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and targets
        """
        try:
            # Create technical indicators
            features = []
            targets = []
            
            # Price-based features
            data['Returns'] = data['Close'].pct_change()
            data['Price_Change'] = data['Close'].diff()
            data['High_Low_Ratio'] = data['High'] / data['Low']
            data['Close_Open_Ratio'] = data['Close'] / data['Open']
            
            # Moving averages
            data['SMA_5'] = data['Close'].rolling(window=5).mean()
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['EMA_12'] = data['Close'].ewm(span=12).mean()
            data['EMA_26'] = data['Close'].ewm(span=26).mean()
            
            # Volatility features
            data['Volatility'] = data['Returns'].rolling(window=20).std()
            data['ATR'] = self._calculate_atr(data)
            
            # Volume features
            data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
            
            # RSI
            data['RSI'] = self._calculate_rsi(data['Close'])
            
            # MACD
            macd_data = self._calculate_macd(data['Close'])
            data['MACD'] = macd_data['macd']
            data['MACD_Signal'] = macd_data['signal']
            
            # Bollinger Bands
            bb_data = self._calculate_bollinger_bands(data['Close'])
            data['BB_Upper'] = bb_data['upper']
            data['BB_Lower'] = bb_data['lower']
            data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
            
            # Remove NaN values
            data = data.dropna()
            
            # Create lagged features
            feature_columns = [
                'Returns', 'Price_Change', 'High_Low_Ratio', 'Close_Open_Ratio',
                'SMA_5', 'SMA_20', 'EMA_12', 'EMA_26', 'Volatility', 'ATR',
                'Volume_Ratio', 'RSI', 'MACD', 'MACD_Signal', 'BB_Position'
            ]
            
            for i in range(lookback_days, len(data)):
                # Features from previous days
                feature_vector = []
                for col in feature_columns:
                    if col in data.columns:
                        feature_vector.extend(data[col].iloc[i-lookback_days:i].values)
                
                # Add current day features
                for col in feature_columns:
                    if col in data.columns:
                        feature_vector.append(data[col].iloc[i])
                
                features.append(feature_vector)
                targets.append(data['Close'].iloc[i])
            
            return np.array(features), np.array(targets)
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            return np.array([]), np.array([])
    
    def train_models(self, data: pd.DataFrame, test_size: float = 0.2) -> Dict[str, float]:
        """
        Train all models on the data
        
        Args:
            data (pd.DataFrame): Stock data
            test_size (float): Proportion of data for testing
        
        Returns:
            Dict[str, float]: Model performance scores
        """
        try:
            # Prepare features
            X, y = self.prepare_features(data)
            
            if len(X) == 0 or len(y) == 0:
                raise ValueError("No valid features generated")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=False
            )
            
            # Scale features
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_test_scaled = self.feature_scaler.transform(X_test)
            
            # Train models and evaluate
            results = {}
            
            for name, model in self.models.items():
                try:
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test_scaled)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    results[name] = {
                        'mse': mse,
                        'mae': mae,
                        'r2': r2,
                        'rmse': np.sqrt(mse)
                    }
                    
                    # Save best model
                    if name == 'gradient_boosting':  # Usually performs well
                        self.best_model = model
                        self.best_model_name = name
                    
                except Exception as e:
                    self.logger.warning(f"Error training {name}: {str(e)}")
                    continue
            
            self.is_trained = True
            return results
            
        except Exception as e:
            self.logger.error(f"Error training models: {str(e)}")
            return {}
    
    def predict_next_day(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict next day's price
        
        Args:
            data (pd.DataFrame): Recent stock data
        
        Returns:
            Dict[str, Any]: Prediction results
        """
        try:
            if not self.is_trained:
                raise ValueError("Models must be trained before making predictions")
            
            # Prepare features for prediction
            X, _ = self.prepare_features(data)
            
            if len(X) == 0:
                raise ValueError("No valid features for prediction")
            
            # Use the most recent feature vector
            latest_features = X[-1:].reshape(1, -1)
            latest_features_scaled = self.feature_scaler.transform(latest_features)
            
            # Make predictions with all models
            predictions = {}
            for name, model in self.models.items():
                try:
                    pred = model.predict(latest_features_scaled)[0]
                    predictions[name] = pred
                except Exception as e:
                    self.logger.warning(f"Error predicting with {name}: {str(e)}")
                    continue
            
            # Ensemble prediction (average of all models)
            if predictions:
                ensemble_pred = np.mean(list(predictions.values()))
                predictions['ensemble'] = ensemble_pred
            
            return {
                'predictions': predictions,
                'current_price': data['Close'].iloc[-1],
                'prediction_date': (data.index[-1] + timedelta(days=1)).strftime('%Y-%m-%d'),
                'confidence': self._calculate_confidence(predictions)
            }
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            return {}
    
    def predict_multiple_days(self, data: pd.DataFrame, days: int = 5) -> Dict[str, Any]:
        """
        Predict prices for multiple days
        
        Args:
            data (pd.DataFrame): Recent stock data
            days (int): Number of days to predict
        
        Returns:
            Dict[str, Any]: Multi-day predictions
        """
        try:
            predictions = []
            current_data = data.copy()
            
            for day in range(days):
                # Predict next day
                pred_result = self.predict_next_day(current_data)
                
                if not pred_result:
                    break
                
                # Add prediction to data for next iteration
                next_date = pd.to_datetime(pred_result['prediction_date'])
                ensemble_pred = pred_result['predictions'].get('ensemble', pred_result['current_price'])
                
                # Create new row with predicted price
                new_row = current_data.iloc[-1].copy()
                new_row['Close'] = ensemble_pred
                new_row['Open'] = ensemble_pred * 0.999  # Approximate
                new_row['High'] = ensemble_pred * 1.005
                new_row['Low'] = ensemble_pred * 0.995
                new_row.name = next_date
                
                current_data = current_data.append(new_row)
                predictions.append(pred_result)
            
            return {
                'predictions': predictions,
                'forecast_dates': [p['prediction_date'] for p in predictions],
                'forecast_prices': [p['predictions'].get('ensemble', p['current_price']) for p in predictions]
            }
            
        except Exception as e:
            self.logger.error(f"Error making multi-day prediction: {str(e)}")
            return {}
    
    def save_model(self, model_name: str = None):
        """Save the best model"""
        try:
            if model_name is None:
                model_name = self.best_model_name
            
            model_file = os.path.join(self.model_path, f"{self.symbol}_{model_name}.joblib")
            scaler_file = os.path.join(self.model_path, f"{self.symbol}_scaler.joblib")
            
            joblib.dump(self.models[model_name], model_file)
            joblib.dump(self.feature_scaler, scaler_file)
            
            self.logger.info(f"Model saved to {model_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
    
    def load_model(self, model_name: str = None):
        """Load a saved model"""
        try:
            if model_name is None:
                model_name = 'gradient_boosting'  # Default
            
            model_file = os.path.join(self.model_path, f"{self.symbol}_{model_name}.joblib")
            scaler_file = os.path.join(self.model_path, f"{self.symbol}_scaler.joblib")
            
            if os.path.exists(model_file) and os.path.exists(scaler_file):
                self.models[model_name] = joblib.load(model_file)
                self.feature_scaler = joblib.load(scaler_file)
                self.best_model = self.models[model_name]
                self.best_model_name = model_name
                self.is_trained = True
                
                self.logger.info(f"Model loaded from {model_file}")
                return True
            else:
                self.logger.warning(f"Model files not found: {model_file}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        try:
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window=period).mean()
            return atr
        except Exception:
            return pd.Series(dtype=float)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception:
            return pd.Series(dtype=float)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            
            return {
                'macd': macd_line,
                'signal': signal_line,
                'histogram': macd_line - signal_line
            }
        except Exception:
            return {'macd': pd.Series(), 'signal': pd.Series(), 'histogram': pd.Series()}
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            
            return {
                'upper': sma + (std * std_dev),
                'middle': sma,
                'lower': sma - (std * std_dev)
            }
        except Exception:
            return {'upper': pd.Series(), 'middle': pd.Series(), 'lower': pd.Series()}
    
    def _calculate_confidence(self, predictions: Dict[str, float]) -> float:
        """Calculate confidence based on prediction consistency"""
        if not predictions:
            return 0.0
        
        values = list(predictions.values())
        if len(values) == 1:
            return 0.5  # Medium confidence for single model
        
        # Calculate coefficient of variation (lower is better)
        mean_pred = np.mean(values)
        std_pred = np.std(values)
        
        if mean_pred == 0:
            return 0.0
        
        cv = std_pred / abs(mean_pred)
        confidence = max(0.0, min(1.0, 1.0 - cv))
        
        return confidence


class SentimentAnalyzer:
    """
    Sentiment analysis for stock market news and social media
    """
    
    def __init__(self):
        """Initialize sentiment analyzer"""
        self.logger = logging.getLogger(__name__)
        
    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text
        
        Args:
            text (str): Text to analyze
        
        Returns:
            Dict[str, float]: Sentiment scores
        """
        try:
            # Simple sentiment analysis (can be enhanced with more sophisticated models)
            positive_words = ['bullish', 'positive', 'growth', 'profit', 'gain', 'up', 'rise', 'strong']
            negative_words = ['bearish', 'negative', 'loss', 'decline', 'down', 'fall', 'weak', 'crash']
            
            text_lower = text.lower()
            
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            total_words = len(text.split())
            if total_words == 0:
                return {'sentiment': 0.0, 'confidence': 0.0}
            
            sentiment_score = (positive_count - negative_count) / total_words
            confidence = min(1.0, (positive_count + negative_count) / total_words)
            
            return {
                'sentiment': sentiment_score,
                'confidence': confidence,
                'positive_words': positive_count,
                'negative_words': negative_count
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {str(e)}")
            return {'sentiment': 0.0, 'confidence': 0.0}
    
    def analyze_news_sentiment(self, news_data: List[Dict]) -> Dict[str, Any]:
        """
        Analyze sentiment of news articles
        
        Args:
            news_data (List[Dict]): List of news articles
        
        Returns:
            Dict[str, Any]: Overall sentiment analysis
        """
        try:
            sentiments = []
            
            for article in news_data:
                title = article.get('title', '')
                content = article.get('content', '')
                text = f"{title} {content}"
                
                sentiment = self.analyze_text_sentiment(text)
                sentiment['date'] = article.get('date', '')
                sentiment['source'] = article.get('source', '')
                
                sentiments.append(sentiment)
            
            if not sentiments:
                return {'overall_sentiment': 0.0, 'confidence': 0.0, 'article_count': 0}
            
            # Calculate weighted average sentiment
            total_confidence = sum(s['confidence'] for s in sentiments)
            if total_confidence == 0:
                return {'overall_sentiment': 0.0, 'confidence': 0.0, 'article_count': len(sentiments)}
            
            weighted_sentiment = sum(s['sentiment'] * s['confidence'] for s in sentiments) / total_confidence
            avg_confidence = total_confidence / len(sentiments)
            
            return {
                'overall_sentiment': weighted_sentiment,
                'confidence': avg_confidence,
                'article_count': len(sentiments),
                'sentiments': sentiments
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing news sentiment: {str(e)}")
            return {'overall_sentiment': 0.0, 'confidence': 0.0, 'article_count': 0}


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
    
    # Test price predictor
    predictor = PricePredictor("AAPL")
    results = predictor.train_models(sample_data)
    
    print("Model Performance:")
    for model, metrics in results.items():
        print(f"{model}: R² = {metrics['r2']:.3f}, RMSE = {metrics['rmse']:.2f}")
    
    # Test prediction
    prediction = predictor.predict_next_day(sample_data)
    print(f"\nNext day prediction: ${prediction.get('predictions', {}).get('ensemble', 0):.2f}")
    
    # Test sentiment analyzer
    analyzer = SentimentAnalyzer()
    sentiment = analyzer.analyze_text_sentiment("Stock shows strong growth and positive earnings")
    print(f"\nSentiment score: {sentiment['sentiment']:.3f}")
