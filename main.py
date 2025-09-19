"""
Main Stock Market Analysis System
Demonstrates the complete stock market analysis and trading system
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.stock_data import StockData
from analysis.technical_indicators import TechnicalAnalysis
from analysis.ml_models import PricePredictor, SentimentAnalyzer
from trading.strategies import (
    MovingAverageStrategy, RSIStrategy, MACDStrategy, 
    BollingerBandsStrategy, MeanReversionStrategy, PortfolioOptimizer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Main function demonstrating the stock market analysis system"""
    
    print("🚀 Stock Market Analysis System")
    print("=" * 50)
    
    # Example symbols to analyze
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    
    for symbol in symbols:
        print(f"\n📊 Analyzing {symbol}...")
        
        try:
            # Initialize stock data
            stock_data = StockData(symbol)
            
            # Get historical data
            data = stock_data.get_historical_data(period="1y")
            data.set_index('Date', inplace=True)
            
            if data.empty:
                print(f"❌ No data found for {symbol}")
                continue
            
            # 1. Basic Stock Information
            print(f"\n📋 Basic Information for {symbol}:")
            try:
                real_time = stock_data.get_real_time_data()
                company_info = stock_data.get_company_info()
                
                print(f"   Current Price: ${real_time['current_price']:.2f}")
                print(f"   Change: {real_time['change_percent']:.2f}%")
                print(f"   Volume: {real_time['volume']:,}")
                print(f"   Company: {company_info['name']}")
                print(f"   Sector: {company_info['sector']}")
                
            except Exception as e:
                print(f"   ⚠️ Could not fetch real-time data: {str(e)}")
            
            # 2. Technical Analysis
            print(f"\n📊 Technical Analysis for {symbol}:")
            try:
                ta = TechnicalAnalysis(data)
                
                # Calculate indicators
                rsi = ta.calculate_rsi()
                macd_data = ta.calculate_macd()
                bb_data = ta.calculate_bollinger_bands()
                
                if len(rsi) > 0:
                    current_rsi = rsi.iloc[-1]
                    print(f"   RSI: {current_rsi:.2f}")
                    if current_rsi < 30:
                        print("   📈 RSI indicates oversold conditions")
                    elif current_rsi > 70:
                        print("   📉 RSI indicates overbought conditions")
                    else:
                        print("   ➡️ RSI is neutral")
                
                if len(macd_data['macd_line']) > 0:
                    current_macd = macd_data['macd_line'].iloc[-1]
                    current_signal = macd_data['signal_line'].iloc[-1]
                    print(f"   MACD: {current_macd:.4f}")
                    print(f"   Signal: {current_signal:.4f}")
                    
                    if current_macd > current_signal:
                        print("   📈 MACD indicates bullish momentum")
                    else:
                        print("   📉 MACD indicates bearish momentum")
                
                # Get technical summary
                summary = ta.get_technical_summary()
                if summary:
                    print(f"   Trend: {summary.get('trend', 'Unknown')}")
                    print(f"   BB Position: {summary.get('bollinger_position', 'Unknown')}")
                
            except Exception as e:
                print(f"   ⚠️ Error in technical analysis: {str(e)}")
            
            # 3. Machine Learning Predictions
            print(f"\n🤖 Machine Learning Predictions for {symbol}:")
            try:
                predictor = PricePredictor(symbol)
                
                # Train models
                results = predictor.train_models(data)
                
                if results:
                    # Find best performing model
                    best_model = max(results.keys(), key=lambda x: results[x]['r2'])
                    best_r2 = results[best_model]['r2']
                    print(f"   Best Model: {best_model} (R² = {best_r2:.3f})")
                    
                    # Make prediction
                    prediction = predictor.predict_next_day(data)
                    
                    if prediction:
                        current_price = prediction['current_price']
                        ensemble_pred = prediction['predictions'].get('ensemble', current_price)
                        confidence = prediction.get('confidence', 0)
                        
                        print(f"   Current Price: ${current_price:.2f}")
                        print(f"   Predicted Price: ${ensemble_pred:.2f}")
                        print(f"   Prediction Confidence: {confidence:.1%}")
                        
                        change = ensemble_pred - current_price
                        change_percent = (change / current_price) * 100
                        
                        if change > 0:
                            print(f"   📈 Predicted Change: +{change_percent:.2f}%")
                        else:
                            print(f"   📉 Predicted Change: {change_percent:.2f}%")
                
            except Exception as e:
                print(f"   ⚠️ Error in predictions: {str(e)}")
            
            # 4. Trading Strategy Backtesting
            print(f"\n📈 Trading Strategy Backtesting for {symbol}:")
            try:
                # Test multiple strategies
                strategies = [
                    ("Moving Average", MovingAverageStrategy(symbol)),
                    ("RSI", RSIStrategy(symbol)),
                    ("MACD", MACDStrategy(symbol)),
                    ("Bollinger Bands", BollingerBandsStrategy(symbol)),
                    ("Mean Reversion", MeanReversionStrategy(symbol))
                ]
                
                strategy_results = {}
                
                for name, strategy in strategies:
                    try:
                        results = strategy.backtest(data)
                        if results:
                            strategy_results[name] = results
                            total_return = results.get('total_return', 0)
                            buy_hold = results.get('buy_hold_return', 0)
                            trades = results.get('total_trades', 0)
                            
                            print(f"   {name}:")
                            print(f"     Total Return: {total_return:.2f}%")
                            print(f"     Buy & Hold: {buy_hold:.2f}%")
                            print(f"     Excess Return: {total_return - buy_hold:.2f}%")
                            print(f"     Total Trades: {trades}")
                            
                    except Exception as e:
                        print(f"     ⚠️ Error testing {name}: {str(e)}")
                
                # Find best strategy
                if strategy_results:
                    best_strategy = max(strategy_results.keys(), 
                                     key=lambda x: strategy_results[x].get('total_return', 0))
                    best_return = strategy_results[best_strategy].get('total_return', 0)
                    print(f"\n   🏆 Best Strategy: {best_strategy} ({best_return:.2f}%)")
                
            except Exception as e:
                print(f"   ⚠️ Error in strategy backtesting: {str(e)}")
            
            # 5. Data Summary
            print(f"\n📊 Data Summary for {symbol}:")
            try:
                summary = stock_data.get_data_summary()
                if summary:
                    print(f"   Data Points: {summary['data_points']}")
                    print(f"   Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
                    print(f"   Current Price: ${summary['price_stats']['current_price']:.2f}")
                    print(f"   Highest Price: ${summary['price_stats']['highest_price']:.2f}")
                    print(f"   Lowest Price: ${summary['price_stats']['lowest_price']:.2f}")
                    print(f"   Total Return: {summary['returns']['total_return']:.2f}%")
                    print(f"   Annualized Return: {summary['returns']['annualized_return']:.2f}%")
                
            except Exception as e:
                print(f"   ⚠️ Error in data summary: {str(e)}")
            
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {str(e)}")
    
    # 6. Portfolio Optimization
    print(f"\n🎯 Portfolio Optimization:")
    try:
        optimizer = PortfolioOptimizer(symbols)
        
        # Create sample returns data (in real application, this would be actual returns)
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        returns_data = pd.DataFrame({
            'AAPL': np.random.randn(len(dates)) * 0.02,
            'GOOGL': np.random.randn(len(dates)) * 0.025,
            'MSFT': np.random.randn(len(dates)) * 0.018,
            'TSLA': np.random.randn(len(dates)) * 0.03
        })
        
        # Test different optimization methods
        methods = ['equal', 'min_variance', 'max_sharpe']
        
        for method in methods:
            try:
                weights = optimizer.calculate_optimal_weights(returns_data, method)
                metrics = optimizer.calculate_portfolio_metrics(returns_data, weights)
                
                print(f"\n   {method.upper()} Portfolio:")
                print(f"     Annual Return: {metrics.get('annual_return', 0):.2%}")
                print(f"     Volatility: {metrics.get('volatility', 0):.2%}")
                print(f"     Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
                print(f"     Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
                print(f"     Weights: {weights}")
                
            except Exception as e:
                print(f"     ⚠️ Error in {method} optimization: {str(e)}")
    
    except Exception as e:
        print(f"❌ Error in portfolio optimization: {str(e)}")
    
    # 7. Sentiment Analysis
    print(f"\n😊 Sentiment Analysis:")
    try:
        analyzer = SentimentAnalyzer()
        
        # Sample news for demonstration
        sample_news = [
            {
                'title': 'Tech stocks show strong growth in Q4',
                'content': 'Technology companies reported positive earnings and strong revenue growth.',
                'date': '2024-01-15',
                'source': 'Financial News'
            },
            {
                'title': 'Market volatility affects tech sector',
                'content': 'Recent market fluctuations have impacted technology stock performance.',
                'date': '2024-01-14',
                'source': 'Market Watch'
            }
        ]
        
        sentiment_result = analyzer.analyze_news_sentiment(sample_news)
        
        if sentiment_result:
            sentiment_score = sentiment_result['overall_sentiment']
            confidence = sentiment_result['confidence']
            article_count = sentiment_result['article_count']
            
            print(f"   Overall Sentiment: {sentiment_score:.3f}")
            print(f"   Confidence: {confidence:.1%}")
            print(f"   Articles Analyzed: {article_count}")
            
            if sentiment_score > 0:
                print("   📈 Sentiment is positive")
            elif sentiment_score < 0:
                print("   📉 Sentiment is negative")
            else:
                print("   ➡️ Sentiment is neutral")
    
    except Exception as e:
        print(f"❌ Error in sentiment analysis: {str(e)}")
    
    print(f"\n✅ Stock Market Analysis Complete!")
    print("=" * 50)

def run_streamlit_app():
    """Run the Streamlit web application"""
    import subprocess
    import sys
    
    print("🌐 Starting Streamlit Web Application...")
    print("The dashboard will open in your browser at http://localhost:8501")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Streamlit app stopped by user")
    except Exception as e:
        print(f"❌ Error starting Streamlit app: {str(e)}")

def export_data_example():
    """Example of data export functionality"""
    print("\n📤 Data Export Example:")
    
    try:
        stock_data = StockData("AAPL")
        data = stock_data.get_historical_data(period="6mo")
        
        # Export to different formats
        formats = ['csv', 'excel', 'json']
        
        for format_type in formats:
            filename = f"aapl_data.{format_type}"
            stock_data.export_data(filename, format_type)
            print(f"   ✅ Exported data to {filename}")
    
    except Exception as e:
        print(f"❌ Error in data export: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stock Market Analysis System")
    parser.add_argument("--mode", choices=["analysis", "web", "export"], 
                       default="analysis", help="Run mode")
    
    args = parser.parse_args()
    
    if args.mode == "analysis":
        main()
    elif args.mode == "web":
        run_streamlit_app()
    elif args.mode == "export":
        export_data_example()
    else:
        print("❌ Invalid mode specified")
        sys.exit(1)
