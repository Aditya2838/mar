"""
Nifty 50 Prediction Module
Advanced machine learning models for predicting Nifty 50 movements
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data.nifty50_data import Nifty50Data
from analysis.technical_indicators import TechnicalAnalysis
from config import Config

class Nifty50Predictions:
    """
    Advanced prediction models for Nifty 50 index
    """
    
    def __init__(self):
        """Initialize Nifty50Predictions"""
        self.nifty_data = Nifty50Data()
        self.technical_indicators = None  # Will be initialized when needed
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def prepare_features(self, data: pd.DataFrame, lookback_days: int = 30) -> pd.DataFrame:
        """
        Prepare features for prediction models
        
        Args:
            data (pd.DataFrame): Historical price data
            lookback_days (int): Number of days to look back for features
        
        Returns:
            pd.DataFrame: Feature matrix
        """
        try:
            # Initialize technical indicators with data
            self.technical_indicators = TechnicalAnalysis(data)
            # Calculate technical indicators
            data = self.technical_indicators.calculate_all_indicators()
            
            # Create lagged features
            for i in range(1, lookback_days + 1):
                data[f'Close_Lag_{i}'] = data['Close'].shift(i)
                data[f'Volume_Lag_{i}'] = data['Volume'].shift(i)
                data[f'Returns_Lag_{i}'] = data['Returns'].shift(i)
            
            # Create rolling statistics
            for window in [5, 10, 20, 50]:
                data[f'Close_MA_{window}'] = data['Close'].rolling(window=window).mean()
                data[f'Volume_MA_{window}'] = data['Volume'].rolling(window=window).mean()
                data[f'Volatility_{window}'] = data['Returns'].rolling(window=window).std()
            
            # Create price-based features
            data['High_Low_Ratio'] = data['High'] / data['Low']
            data['Close_Open_Ratio'] = data['Close'] / data['Open']
            data['Price_Range'] = (data['High'] - data['Low']) / data['Close']
            
            # Create momentum features
            data['Price_Momentum_5'] = data['Close'] / data['Close'].shift(5) - 1
            data['Price_Momentum_10'] = data['Close'] / data['Close'].shift(10) - 1
            data['Price_Momentum_20'] = data['Close'] / data['Close'].shift(20) - 1
            
            # Create volatility features
            data['ATR_5'] = self.technical_indicators.calculate_atr(data, period=5)
            data['ATR_10'] = self.technical_indicators.calculate_atr(data, period=10)
            
            # Create support/resistance features
            data['Support_Level'] = data['Low'].rolling(window=20).min()
            data['Resistance_Level'] = data['High'].rolling(window=20).max()
            data['Support_Distance'] = (data['Close'] - data['Support_Level']) / data['Close']
            data['Resistance_Distance'] = (data['Resistance_Level'] - data['Close']) / data['Close']
            
            # Create market regime features
            data['Trend_Strength'] = abs(data['SMA_20'] - data['SMA_50']) / data['SMA_50']
            data['Volume_Price_Trend'] = data['Volume'] * data['Returns']
            
            # Create time-based features
            data['Day_of_Week'] = pd.to_datetime(data['Date']).dt.dayofweek
            data['Month'] = pd.to_datetime(data['Date']).dt.month
            data['Quarter'] = pd.to_datetime(data['Date']).dt.quarter
            
            # Create target variable (next day's return)
            data['Target_Return'] = data['Returns'].shift(-1)
            data['Target_Direction'] = (data['Target_Return'] > 0).astype(int)
            
            # Drop rows with NaN values
            data = data.dropna()
            
            return data
            
        except Exception as e:
            print(f"Error preparing features: {e}")
            return pd.DataFrame()
    
    def get_feature_columns(self) -> list:
        """Get list of feature columns for prediction"""
        return [
            'Returns', 'Price_Change', 'High_Low_Ratio', 'Close_Open_Ratio',
            'SMA_5', 'SMA_20', 'EMA_12', 'EMA_26', 'Volatility', 'ATR',
            'Volume_Ratio', 'RSI', 'MACD', 'MACD_Signal', 'BB_Position',
            'Price_Momentum_5', 'Price_Momentum_10', 'Price_Momentum_20',
            'Support_Distance', 'Resistance_Distance', 'Trend_Strength',
            'Volume_Price_Trend', 'Day_of_Week', 'Month', 'Quarter'
        ]
    
    def train_models(self, data: pd.DataFrame, test_size: float = 0.2):
        """
        Train multiple prediction models
        
        Args:
            data (pd.DataFrame): Feature matrix
            test_size (float): Proportion of data for testing
        """
        try:
            # Prepare features and target
            feature_cols = self.get_feature_columns()
            X = data[feature_cols]
            y_return = data['Target_Return']
            y_direction = data['Target_Direction']
            
            # Split data
            X_train, X_test, y_train_return, y_test_return, y_train_dir, y_test_dir = train_test_split(
                X, y_return, y_direction, test_size=test_size, random_state=42, shuffle=False
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            self.scalers['standard'] = scaler
            
            # Initialize models
            models = {
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
                'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42),
                'LinearRegression': LinearRegression(),
                'Ridge': Ridge(alpha=1.0),
                'Lasso': Lasso(alpha=0.1),
                'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
            }
            
            # Train models
            for name, model in models.items():
                print(f"Training {name}...")
                
                if name in ['SVR']:
                    # Use scaled data for SVR
                    model.fit(X_train_scaled, y_train_return)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train_return)
                    y_pred = model.predict(X_test)
                
                # Store model
                self.models[name] = model
                
                # Calculate metrics
                mse = mean_squared_error(y_test_return, y_pred)
                mae = mean_absolute_error(y_test_return, y_pred)
                r2 = r2_score(y_test_return, y_pred)
                
                print(f"{name} - MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.4f}")
                
                # Store feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = dict(zip(feature_cols, model.feature_importances_))
            
            print("Model training completed!")
            
        except Exception as e:
            print(f"Error training models: {e}")
    
    def predict_next_day(self, model_name: str = 'RandomForest') -> dict:
        """
        Predict next day's Nifty 50 movement
        
        Args:
            model_name (str): Name of the model to use for prediction
        
        Returns:
            dict: Prediction results
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
            
            # Get latest data
            latest_data = self.nifty_data.get_nifty50_index_data(period="3mo")
            
            if latest_data.empty:
                return {"error": "No data available for prediction"}
            
            # Prepare features
            feature_data = self.prepare_features(latest_data)
            
            if feature_data.empty:
                return {"error": "Failed to prepare features"}
            
            # Get latest feature vector
            latest_features = feature_data.iloc[-1][self.get_feature_columns()]
            
            # Make prediction
            model = self.models[model_name]
            
            if model_name in ['SVR']:
                # Scale features for SVR
                latest_features_scaled = self.scalers['standard'].transform([latest_features])
                predicted_return = model.predict(latest_features_scaled)[0]
            else:
                predicted_return = model.predict([latest_features])[0]
            
            # Calculate confidence based on model performance
            confidence = self._calculate_prediction_confidence(model_name, predicted_return)
            
            # Determine prediction direction
            direction = "UP" if predicted_return > 0 else "DOWN"
            
            # Get current Nifty 50 level
            current_level = latest_data['Close'].iloc[-1]
            predicted_level = current_level * (1 + predicted_return)
            
            prediction_result = {
                'model': model_name,
                'predicted_return': predicted_return,
                'predicted_direction': direction,
                'current_level': current_level,
                'predicted_level': predicted_level,
                'confidence': confidence,
                'prediction_date': datetime.now().strftime('%Y-%m-%d'),
                'target_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            }
            
            return prediction_result
            
        except Exception as e:
            return {"error": f"Prediction failed: {e}"}
    
    def predict_multiple_days(self, days: int = 5, model_name: str = 'RandomForest') -> list:
        """
        Predict Nifty 50 movements for multiple days
        
        Args:
            days (int): Number of days to predict
            model_name (str): Name of the model to use
        
        Returns:
            list: List of predictions for each day
        """
        try:
            predictions = []
            current_data = self.nifty_data.get_nifty50_index_data(period="3mo")
            
            if current_data.empty:
                return [{"error": "No data available for prediction"}]
            
            current_level = current_data['Close'].iloc[-1]
            
            for day in range(1, days + 1):
                # Get prediction for this day
                prediction = self.predict_next_day(model_name)
                
                if 'error' in prediction:
                    return [prediction]
                
                # Update prediction with day number
                prediction['day'] = day
                prediction['target_date'] = (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d')
                
                predictions.append(prediction)
                
                # Update current level for next iteration
                current_level = prediction['predicted_level']
            
            return predictions
            
        except Exception as e:
            return [{"error": f"Multi-day prediction failed: {e}"}]
    
    def ensemble_prediction(self, weights: dict = None) -> dict:
        """
        Make ensemble prediction using multiple models
        
        Args:
            weights (dict): Weights for each model (default: equal weights)
        
        Returns:
            dict: Ensemble prediction result
        """
        try:
            if not self.models:
                return {"error": "No models available. Train models first."}
            
            # Default equal weights
            if weights is None:
                weights = {name: 1.0/len(self.models) for name in self.models.keys()}
            
            # Get predictions from all models
            predictions = {}
            for model_name in self.models.keys():
                pred = self.predict_next_day(model_name)
                if 'error' not in pred:
                    predictions[model_name] = pred['predicted_return']
            
            if not predictions:
                return {"error": "No valid predictions from any model"}
            
            # Calculate weighted average
            weighted_return = sum(predictions[model] * weights[model] for model in predictions.keys())
            
            # Calculate ensemble confidence
            ensemble_confidence = self._calculate_ensemble_confidence(predictions, weights)
            
            # Get current level
            current_data = self.nifty_data.get_nifty50_index_data(period="1d")
            current_level = current_data['Close'].iloc[-1] if not current_data.empty else 0
            
            ensemble_result = {
                'method': 'Ensemble',
                'predicted_return': weighted_return,
                'predicted_direction': "UP" if weighted_return > 0 else "DOWN",
                'current_level': current_level,
                'predicted_level': current_level * (1 + weighted_return),
                'confidence': ensemble_confidence,
                'individual_predictions': predictions,
                'weights_used': weights,
                'prediction_date': datetime.now().strftime('%Y-%m-%d'),
                'target_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            }
            
            return ensemble_result
            
        except Exception as e:
            return {"error": f"Ensemble prediction failed: {e}"}
    
    def _calculate_prediction_confidence(self, model_name: str, predicted_return: float) -> float:
        """Calculate confidence level for a prediction"""
        try:
            # Base confidence on historical model performance
            base_confidence = 0.7  # Default confidence
            
            # Adjust based on prediction magnitude
            if abs(predicted_return) > 0.02:  # >2% move
                base_confidence += 0.1
            elif abs(predicted_return) < 0.005:  # <0.5% move
                base_confidence -= 0.1
            
            # Adjust based on market volatility
            volatility_data = self.nifty_data.get_volatility_analysis(period="1mo")
            if volatility_data:
                current_vol = volatility_data.get('current_rolling_volatility', 0.2)
                if current_vol > 0.25:  # High volatility
                    base_confidence -= 0.1
                elif current_vol < 0.15:  # Low volatility
                    base_confidence += 0.1
            
            # Ensure confidence is within bounds
            return max(0.1, min(0.95, base_confidence))
            
        except Exception as e:
            return 0.5  # Default confidence
    
    def _calculate_ensemble_confidence(self, predictions: dict, weights: dict) -> float:
        """Calculate confidence for ensemble prediction"""
        try:
            # Calculate weighted standard deviation of predictions
            weighted_mean = sum(predictions[model] * weights[model] for model in predictions.keys())
            weighted_variance = sum(weights[model] * (predictions[model] - weighted_mean)**2 
                                  for model in predictions.keys())
            weighted_std = np.sqrt(weighted_variance)
            
            # Higher agreement (lower std) means higher confidence
            agreement_factor = 1 / (1 + weighted_std * 100)
            
            # Base confidence
            base_confidence = 0.75
            
            # Adjust based on agreement
            final_confidence = base_confidence * agreement_factor
            
            return max(0.1, min(0.95, final_confidence))
            
        except Exception as e:
            return 0.7
    
    def get_model_performance_summary(self) -> dict:
        """Get summary of all model performances"""
        try:
            if not self.models:
                return {"error": "No models available"}
            
            performance_summary = {}
            
            for model_name in self.models.keys():
                # Get recent predictions
                recent_pred = self.predict_next_day(model_name)
                
                if 'error' not in recent_pred:
                    performance_summary[model_name] = {
                        'last_prediction': recent_pred['predicted_return'],
                        'predicted_direction': recent_pred['predicted_direction'],
                        'confidence': recent_pred['confidence'],
                        'feature_importance': self.feature_importance.get(model_name, {})
                    }
            
            return performance_summary
            
        except Exception as e:
            return {"error": f"Failed to get performance summary: {e}"}
    
    def export_predictions_report(self, filename: str = "nifty50_predictions_report.csv"):
        """Export comprehensive predictions report"""
        try:
            # Get predictions from all models
            all_predictions = []
            
            # Individual model predictions
            for model_name in self.models.keys():
                pred = self.predict_next_day(model_name)
                if 'error' not in pred:
                    all_predictions.append({
                        'Model': model_name,
                        'Type': 'Individual',
                        'Predicted_Return': pred['predicted_return'],
                        'Direction': pred['predicted_direction'],
                        'Confidence': pred['confidence'],
                        'Current_Level': pred['current_level'],
                        'Predicted_Level': pred['predicted_level']
                    })
            
            # Ensemble prediction
            ensemble_pred = self.ensemble_prediction()
            if 'error' not in ensemble_pred:
                all_predictions.append({
                    'Model': 'Ensemble',
                    'Type': 'Ensemble',
                    'Predicted_Return': ensemble_pred['predicted_return'],
                    'Direction': ensemble_pred['predicted_direction'],
                    'Confidence': ensemble_pred['confidence'],
                    'Current_Level': ensemble_pred['current_level'],
                    'Predicted_Level': ensemble_pred['predicted_level']
                })
            
            # Convert to DataFrame and export
            if all_predictions:
                report_df = pd.DataFrame(all_predictions)
                report_df.to_csv(filename, index=False)
                print(f"Predictions report exported to {filename}")
            else:
                print("No predictions available for export")
                
        except Exception as e:
            print(f"Error exporting predictions report: {e}")
