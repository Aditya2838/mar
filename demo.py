#!/usr/bin/env python3
"""
Comprehensive Demo Script for Stock Market Analysis System
Showcases all major features and capabilities
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.stock_data import StockData
from analysis.technical_indicators import TechnicalAnalysis
from analysis.ml_models import PricePredictor, SentimentAnalyzer
from trading.strategies import (
    MovingAverageStrategy, RSIStrategy, MACDStrategy,
    BollingerBandsStrategy, MeanReversionStrategy, PortfolioOptimizer
)

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"🎯 {title}")
    print("=" * 60)

def demo_data_fetching():
    """Demonstrate data fetching capabilities"""
    print_header("Data Fetching Demo")
    
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    for symbol in symbols:
        print(f"\n📊 Fetching data for {symbol}...")
        
        try:
            stock = StockData(symbol)
            
            # Get historical data
            data = stock.get_historical_data(period="6mo")
            print(f"   ✅ Fetched {len(data)} data points")
            
            # Get real-time data
            real_time = stock.get_real_time_data()
            print(f"   💰 Current Price: ${real_time['current_price']:.2f}")
            print(f"   📈 Change: {real_time['change_percent']:.2f}%")
            
            # Get company info
            company_info = stock.get_company_info()
            print(f"   🏢 Company: {company_info['name']}")
            print(f"   🏭 Sector: {company_info['sector']}")
            
            # Get data summary
            summary = stock.get_data_summary()
            if summary:
                print(f"   📊 Total Return: {summary['returns']['total_return']:.2f}%")
                print(f"   📈 Annualized Return: {summary['returns']['annualized_return']:.2f}%")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")

def demo_technical_analysis():
    """Demonstrate technical analysis capabilities"""
    print_header("Technical Analysis Demo")
    
    try:
        # Get data for analysis
        stock = StockData("AAPL")
        data = stock.get_historical_data(period="1y")
        data.set_index('Date', inplace=True)
        
        # Initialize technical analysis
        ta = TechnicalAnalysis(data)
        
        # Calculate various indicators
        print("\n📊 Calculating Technical Indicators...")
        
        # RSI
        rsi = ta.calculate_rsi()
        current_rsi = rsi.iloc[-1] if len(rsi) > 0 else None
        print(f"   RSI: {current_rsi:.2f}" if current_rsi else "   RSI: N/A")
        
        # MACD
        macd_data = ta.calculate_macd()
        if len(macd_data['macd_line']) > 0:
            current_macd = macd_data['macd_line'].iloc[-1]
            current_signal = macd_data['signal_line'].iloc[-1]
            print(f"   MACD: {current_macd:.4f}")
            print(f"   Signal: {current_signal:.4f}")
        
        # Bollinger Bands
        bb_data = ta.calculate_bollinger_bands()
        if len(bb_data['upper_band']) > 0:
            current_price = data['Close'].iloc[-1]
            upper = bb_data['upper_band'].iloc[-1]
            lower = bb_data['lower_band'].iloc[-1]
            print(f"   BB Upper: ${upper:.2f}")
            print(f"   BB Lower: ${lower:.2f}")
            print(f"   Price vs BB: {'Above' if current_price > upper else 'Below' if current_price < lower else 'Within'}")
        
        # Moving Averages
        sma_data = ta.calculate_moving_averages()
        if 'SMA_20' in sma_data and len(sma_data['SMA_20']) > 0:
            sma_20 = sma_data['SMA_20'].iloc[-1]
            sma_50 = sma_data['SMA_50'].iloc[-1] if 'SMA_50' in sma_data and len(sma_data['SMA_50']) > 0 else None
            print(f"   SMA 20: ${sma_20:.2f}")
            if sma_50:
                print(f"   SMA 50: ${sma_50:.2f}")
                trend = "Bullish" if sma_20 > sma_50 else "Bearish"
                print(f"   Trend: {trend}")
        
        # Support and Resistance
        support_resistance = ta.detect_support_resistance()
        if support_resistance['support_levels']:
            print(f"   Support Levels: {[f'${x:.2f}' for x in support_resistance['support_levels'][:3]]}")
        if support_resistance['resistance_levels']:
            print(f"   Resistance Levels: {[f'${x:.2f}' for x in support_resistance['resistance_levels'][:3]]}")
        
        # Fibonacci Retracements
        fib_data = ta.calculate_fibonacci_retracements()
        if fib_data:
            print(f"   Fibonacci 0.618: ${fib_data.get('fib_618', 0):.2f}")
            print(f"   Fibonacci 0.382: ${fib_data.get('fib_382', 0):.2f}")
        
        # Technical Summary
        summary = ta.get_technical_summary()
        if summary:
            print(f"\n📋 Technical Summary:")
            print(f"   Current Price: ${summary.get('current_price', 0):.2f}")
            print(f"   Trend: {summary.get('trend', 'Unknown')}")
            print(f"   RSI Signal: {summary.get('rsi_signal', 'Unknown')}")
            print(f"   BB Position: {summary.get('bollinger_position', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ Error in technical analysis: {str(e)}")

def demo_machine_learning():
    """Demonstrate machine learning capabilities"""
    print_header("Machine Learning Demo")
    
    try:
        # Get data
        stock = StockData("AAPL")
        data = stock.get_historical_data(period="1y")
        data.set_index('Date', inplace=True)
        
        # Initialize predictor
        predictor = PricePredictor("AAPL")
        
        print("\n🤖 Training Machine Learning Models...")
        
        # Train models
        results = predictor.train_models(data)
        
        if results:
            print("\n📊 Model Performance:")
            for model_name, metrics in results.items():
                r2 = metrics.get('r2', 0)
                rmse = metrics.get('rmse', 0)
                print(f"   {model_name}: R² = {r2:.3f}, RMSE = {rmse:.2f}")
            
            # Find best model
            best_model = max(results.keys(), key=lambda x: results[x]['r2'])
            best_r2 = results[best_model]['r2']
            print(f"\n🏆 Best Model: {best_model} (R² = {best_r2:.3f})")
            
            # Make predictions
            print("\n🔮 Making Predictions...")
            prediction = predictor.predict_next_day(data)
            
            if prediction:
                current_price = prediction['current_price']
                predictions = prediction['predictions']
                confidence = prediction.get('confidence', 0)
                
                print(f"   Current Price: ${current_price:.2f}")
                print(f"   Prediction Confidence: {confidence:.1%}")
                
                print("\n📈 Model Predictions:")
                for model, pred in predictions.items():
                    change = pred - current_price
                    change_percent = (change / current_price) * 100
                    print(f"   {model}: ${pred:.2f} ({change_percent:+.2f}%)")
                
                # Ensemble prediction
                ensemble_pred = predictions.get('ensemble', current_price)
                ensemble_change = ensemble_pred - current_price
                ensemble_change_percent = (ensemble_change / current_price) * 100
                
                print(f"\n🎯 Ensemble Prediction: ${ensemble_pred:.2f} ({ensemble_change_percent:+.2f}%)")
                
                if ensemble_change > 0:
                    print("   📈 Bullish prediction")
                else:
                    print("   📉 Bearish prediction")
        
    except Exception as e:
        print(f"❌ Error in machine learning: {str(e)}")

def demo_trading_strategies():
    """Demonstrate trading strategy capabilities"""
    print_header("Trading Strategies Demo")
    
    try:
        # Get data
        stock = StockData("AAPL")
        data = stock.get_historical_data(period="1y")
        data.set_index('Date', inplace=True)
        
        # Test multiple strategies
        strategies = [
            ("Moving Average", MovingAverageStrategy("AAPL")),
            ("RSI", RSIStrategy("AAPL")),
            ("MACD", MACDStrategy("AAPL")),
            ("Bollinger Bands", BollingerBandsStrategy("AAPL")),
            ("Mean Reversion", MeanReversionStrategy("AAPL"))
        ]
        
        print("\n📈 Backtesting Trading Strategies...")
        
        strategy_results = {}
        
        for name, strategy in strategies:
            try:
                print(f"\n🔄 Testing {name} Strategy...")
                results = strategy.backtest(data)
                
                if results:
                    strategy_results[name] = results
                    
                    total_return = results.get('total_return', 0)
                    buy_hold_return = results.get('buy_hold_return', 0)
                    excess_return = results.get('excess_return', 0)
                    total_trades = results.get('total_trades', 0)
                    final_value = results.get('final_value', 0)
                    
                    print(f"   📊 Total Return: {total_return:.2f}%")
                    print(f"   📈 Buy & Hold: {buy_hold_return:.2f}%")
                    print(f"   🎯 Excess Return: {excess_return:.2f}%")
                    print(f"   🔄 Total Trades: {total_trades}")
                    print(f"   💰 Final Value: ${final_value:.2f}")
                    
                    # Performance assessment
                    if excess_return > 0:
                        print(f"   ✅ Strategy outperformed buy & hold")
                    else:
                        print(f"   ❌ Strategy underperformed buy & hold")
                
            except Exception as e:
                print(f"   ❌ Error testing {name}: {str(e)}")
        
        # Compare strategies
        if strategy_results:
            print(f"\n🏆 Strategy Comparison:")
            
            # Find best performing strategy
            best_strategy = max(strategy_results.keys(), 
                              key=lambda x: strategy_results[x].get('total_return', 0))
            best_return = strategy_results[best_strategy].get('total_return', 0)
            
            print(f"   🥇 Best Strategy: {best_strategy} ({best_return:.2f}%)")
            
            # Sort strategies by return
            sorted_strategies = sorted(strategy_results.items(), 
                                     key=lambda x: x[1].get('total_return', 0), 
                                     reverse=True)
            
            print(f"\n📊 Strategy Rankings:")
            for i, (name, results) in enumerate(sorted_strategies, 1):
                return_val = results.get('total_return', 0)
                trades = results.get('total_trades', 0)
                print(f"   {i}. {name}: {return_val:.2f}% ({trades} trades)")
        
    except Exception as e:
        print(f"❌ Error in trading strategies: {str(e)}")

def demo_portfolio_optimization():
    """Demonstrate portfolio optimization capabilities"""
    print_header("Portfolio Optimization Demo")
    
    try:
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
        optimizer = PortfolioOptimizer(symbols)
        
        print(f"\n🎯 Optimizing Portfolio: {', '.join(symbols)}")
        
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
        
        optimization_results = {}
        
        for method in methods:
            try:
                print(f"\n🔄 {method.upper()} Optimization...")
                
                weights = optimizer.calculate_optimal_weights(returns_data, method)
                metrics = optimizer.calculate_portfolio_metrics(returns_data, weights)
                
                optimization_results[method] = metrics
                
                print(f"   📊 Annual Return: {metrics.get('annual_return', 0):.2%}")
                print(f"   📈 Volatility: {metrics.get('volatility', 0):.2%}")
                print(f"   🎯 Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
                print(f"   📉 Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
                
                print(f"   📋 Weights:")
                for symbol, weight in weights.items():
                    print(f"     {symbol}: {weight:.1%}")
                
            except Exception as e:
                print(f"   ❌ Error in {method} optimization: {str(e)}")
        
        # Compare optimization methods
        if optimization_results:
            print(f"\n🏆 Optimization Method Comparison:")
            
            # Find best Sharpe ratio
            best_sharpe_method = max(optimization_results.keys(), 
                                   key=lambda x: optimization_results[x].get('sharpe_ratio', 0))
            best_sharpe = optimization_results[best_sharpe_method].get('sharpe_ratio', 0)
            
            print(f"   🥇 Best Sharpe Ratio: {best_sharpe_method} ({best_sharpe:.2f})")
            
            # Find lowest volatility
            min_vol_method = min(optimization_results.keys(), 
                               key=lambda x: optimization_results[x].get('volatility', float('inf')))
            min_vol = optimization_results[min_vol_method].get('volatility', 0)
            
            print(f"   🛡️ Lowest Volatility: {min_vol_method} ({min_vol:.2%})")
            
            # Find highest return
            max_return_method = max(optimization_results.keys(), 
                                  key=lambda x: optimization_results[x].get('annual_return', 0))
            max_return = optimization_results[max_return_method].get('annual_return', 0)
            
            print(f"   📈 Highest Return: {max_return_method} ({max_return:.2%})")
        
    except Exception as e:
        print(f"❌ Error in portfolio optimization: {str(e)}")

def demo_sentiment_analysis():
    """Demonstrate sentiment analysis capabilities"""
    print_header("Sentiment Analysis Demo")
    
    try:
        analyzer = SentimentAnalyzer()
        
        # Sample news articles for demonstration
        sample_news = [
            {
                'title': 'Tech stocks surge on strong earnings reports',
                'content': 'Technology companies reported exceptional quarterly earnings, driving market optimism and positive investor sentiment.',
                'date': '2024-01-15',
                'source': 'Financial Times'
            },
            {
                'title': 'Market volatility concerns investors',
                'content': 'Recent market fluctuations and economic uncertainty have created bearish sentiment among traders.',
                'date': '2024-01-14',
                'source': 'Wall Street Journal'
            },
            {
                'title': 'Apple announces innovative new products',
                'content': 'Apple Inc. revealed groundbreaking new technology that analysts predict will boost company growth and stock performance.',
                'date': '2024-01-13',
                'source': 'Tech News'
            },
            {
                'title': 'Federal Reserve policy changes impact markets',
                'content': 'Central bank decisions led to mixed market reactions with some sectors showing decline while others remain stable.',
                'date': '2024-01-12',
                'source': 'Reuters'
            }
        ]
        
        print(f"\n📰 Analyzing {len(sample_news)} news articles...")
        
        # Analyze individual articles
        print(f"\n📋 Individual Article Analysis:")
        for i, article in enumerate(sample_news, 1):
            sentiment = analyzer.analyze_text_sentiment(f"{article['title']} {article['content']}")
            
            print(f"\n   Article {i}: {article['title']}")
            print(f"   Sentiment Score: {sentiment['sentiment']:.3f}")
            print(f"   Confidence: {sentiment['confidence']:.1%}")
            print(f"   Positive Words: {sentiment['positive_words']}")
            print(f"   Negative Words: {sentiment['negative_words']}")
            
            if sentiment['sentiment'] > 0:
                print(f"   📈 Sentiment: Positive")
            elif sentiment['sentiment'] < 0:
                print(f"   📉 Sentiment: Negative")
            else:
                print(f"   ➡️ Sentiment: Neutral")
        
        # Overall sentiment analysis
        print(f"\n🎯 Overall Market Sentiment:")
        sentiment_result = analyzer.analyze_news_sentiment(sample_news)
        
        if sentiment_result:
            overall_sentiment = sentiment_result['overall_sentiment']
            confidence = sentiment_result['confidence']
            article_count = sentiment_result['article_count']
            
            print(f"   Overall Sentiment: {overall_sentiment:.3f}")
            print(f"   Confidence: {confidence:.1%}")
            print(f"   Articles Analyzed: {article_count}")
            
            if overall_sentiment > 0.1:
                print(f"   📈 Market Sentiment: Bullish")
            elif overall_sentiment < -0.1:
                print(f"   📉 Market Sentiment: Bearish")
            else:
                print(f"   ➡️ Market Sentiment: Neutral")
            
            # Sentiment interpretation
            if overall_sentiment > 0:
                print(f"   💡 Interpretation: Positive news outweighs negative news")
            elif overall_sentiment < 0:
                print(f"   💡 Interpretation: Negative news outweighs positive news")
            else:
                print(f"   💡 Interpretation: Balanced news sentiment")
        
    except Exception as e:
        print(f"❌ Error in sentiment analysis: {str(e)}")

def demo_data_export():
    """Demonstrate data export capabilities"""
    print_header("Data Export Demo")
    
    try:
        stock = StockData("AAPL")
        data = stock.get_historical_data(period="3mo")
        
        print(f"\n📤 Exporting AAPL data...")
        
        # Export to different formats
        formats = ['csv', 'excel', 'json']
        
        for format_type in formats:
            try:
                filename = f"aapl_demo_data.{format_type}"
                stock.export_data(filename, format_type)
                print(f"   ✅ Exported to {filename}")
                
                # Show file size
                if os.path.exists(filename):
                    size = os.path.getsize(filename)
                    print(f"   📁 File size: {size:,} bytes")
                
            except Exception as e:
                print(f"   ❌ Error exporting to {format_type}: {str(e)}")
        
        print(f"\n💡 Export files created in current directory")
        
    except Exception as e:
        print(f"❌ Error in data export: {str(e)}")

def main():
    """Main demo function"""
    print("🚀 Stock Market Analysis System - Comprehensive Demo")
    print("=" * 70)
    print("This demo showcases all major features of the system")
    print("=" * 70)
    
    # Run all demos
    demos = [
        ("Data Fetching", demo_data_fetching),
        ("Technical Analysis", demo_technical_analysis),
        ("Machine Learning", demo_machine_learning),
        ("Trading Strategies", demo_trading_strategies),
        ("Portfolio Optimization", demo_portfolio_optimization),
        ("Sentiment Analysis", demo_sentiment_analysis),
        ("Data Export", demo_data_export)
    ]
    
    for name, demo_func in demos:
        try:
            demo_func()
            print(f"\n✅ {name} demo completed successfully")
        except Exception as e:
            print(f"\n❌ {name} demo failed: {str(e)}")
        
        # Add a small delay between demos
        time.sleep(1)
    
    print("\n" + "=" * 70)
    print("🎉 Demo completed! All features have been demonstrated.")
    print("=" * 70)
    print("\n💡 Next steps:")
    print("   - Run 'python main.py --mode web' to start the web dashboard")
    print("   - Run 'python main.py --mode analysis' for detailed analysis")
    print("   - Check the README.md for more usage examples")
    print("   - Modify config.py to customize settings")

if __name__ == "__main__":
    main()
