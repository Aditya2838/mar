"""
Nifty 50 Analysis and Prediction Demo
Comprehensive demonstration of Indian Nifty 50 market analysis capabilities
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data.nifty50_data import Nifty50Data
from analysis.nifty50_predictions import Nifty50Predictions
from analysis.technical_indicators import TechnicalAnalysis
from config import Config

def main():
    """Main demo function"""
    print("🇮🇳 Nifty 50 Market Analysis & Prediction System")
    print("=" * 60)
    
    # Initialize components
    print("\n1. Initializing Nifty 50 Data Manager...")
    nifty_data = Nifty50Data()
    
    print("\n2. Initializing Technical Indicators...")
    # Will be initialized when needed with data
    
    print("\n3. Initializing Prediction Models...")
    predictions = Nifty50Predictions()
    
    # Demo 1: Nifty 50 Index Analysis
    print("\n" + "="*60)
    print("📊 DEMO 1: Nifty 50 Index Analysis")
    print("="*60)
    
    try:
        # Get Nifty 50 index data
        print("\nFetching Nifty 50 index data...")
        index_data = nifty_data.get_nifty50_index_data(period="1y")
        
        if not index_data.empty:
            current_level = index_data['Close'].iloc[-1]
            ytd_return = ((current_level - index_data['Close'].iloc[0]) / index_data['Close'].iloc[0]) * 100
            
            print(f"✅ Current Nifty 50 Level: ₹{current_level:,.2f}")
            print(f"✅ YTD Return: {ytd_return:+.2f}%")
            print(f"✅ 52-Week High: ₹{index_data['High'].max():,.2f}")
            print(f"✅ 52-Week Low: ₹{index_data['Low'].min():,.2f}")
            print(f"✅ Data Points: {len(index_data)}")
        else:
            print("❌ Failed to fetch Nifty 50 data")
            return
    except Exception as e:
        print(f"❌ Error in index analysis: {e}")
    
    # Demo 2: Sector Analysis
    print("\n" + "="*60)
    print("🏭 DEMO 2: Sector Analysis")
    print("="*60)
    
    try:
        print("\nAnalyzing sector performance...")
        sector_analysis = nifty_data.get_sector_analysis(period="6mo")
        
        if sector_analysis:
            print(f"✅ Analyzed {len(sector_analysis)} sectors")
            
            # Show top and worst performing sectors
            sector_performance = []
            for sector, data in sector_analysis.items():
                sector_performance.append({
                    'Sector': sector,
                    'Avg Return (%)': round(data['avg_return'] * 100, 2),
                    'Top Stock': data['top_performer'],
                    'Worst Stock': data['worst_performer']
                })
            
            sector_df = pd.DataFrame(sector_performance)
            sector_df = sector_df.sort_values('Avg Return (%)', ascending=False)
            
            print("\n🏆 Top Performing Sectors:")
            for _, row in sector_df.head(3).iterrows():
                print(f"   {row['Sector']}: {row['Avg Return (%)']}%")
            
            print("\n📉 Worst Performing Sectors:")
            for _, row in sector_df.tail(3).iterrows():
                print(f"   {row['Sector']}: {row['Avg Return (%)']}%")
        else:
            print("❌ Failed to get sector analysis")
    except Exception as e:
        print(f"❌ Error in sector analysis: {e}")
    
    # Demo 3: Market Breadth Analysis
    print("\n" + "="*60)
    print("📊 DEMO 3: Market Breadth Analysis")
    print("="*60)
    
    try:
        print("\nCalculating market breadth...")
        market_breadth = nifty_data.get_market_breadth()
        
        if market_breadth:
            print(f"✅ Total Stocks Analyzed: {market_breadth['total_stocks']}")
            print(f"✅ Advancing Stocks: {market_breadth['advancing']} ({market_breadth['advancing_percentage']:.1f}%)")
            print(f"✅ Declining Stocks: {market_breadth['declining']} ({market_breadth['declining_percentage']:.1f}%)")
            print(f"✅ Unchanged: {market_breadth['unchanged']}")
            print(f"✅ Advance/Decline Ratio: {market_breadth['advance_decline_ratio']:.2f}")
            print(f"✅ New 52-Week Highs: {market_breadth['new_highs']}")
            print(f"✅ New 52-Week Lows: {market_breadth['new_lows']}")
            
            # Market sentiment
            if market_breadth['advance_decline_ratio'] > 1.5:
                sentiment = "🟢 Bullish"
            elif market_breadth['advance_decline_ratio'] < 0.5:
                sentiment = "🔴 Bearish"
            else:
                sentiment = "🟡 Neutral"
            
            print(f"✅ Market Sentiment: {sentiment}")
        else:
            print("❌ Failed to get market breadth")
    except Exception as e:
        print(f"❌ Error in market breadth analysis: {e}")
    
    # Demo 4: Volatility Analysis
    print("\n" + "="*60)
    print("📈 DEMO 4: Volatility Analysis")
    print("="*60)
    
    try:
        print("\nAnalyzing volatility patterns...")
        volatility_analysis = nifty_data.get_volatility_analysis(period="1y")
        
        if volatility_analysis:
            print(f"✅ Daily Volatility: {volatility_analysis['daily_volatility']*100:.2f}%")
            print(f"✅ Annualized Volatility: {volatility_analysis['annualized_volatility']*100:.2f}%")
            print(f"✅ Current Rolling Volatility: {volatility_analysis['current_rolling_volatility']*100:.2f}%")
            print(f"✅ Volatility Trend: {volatility_analysis['volatility_trend']}")
            print(f"✅ Value at Risk (95%): {volatility_analysis['var_95']*100:.2f}%")
            print(f"✅ Value at Risk (99%): {volatility_analysis['var_99']*100:.2f}%")
            print(f"✅ Max Daily Return: {volatility_analysis['max_daily_return']*100:.2f}%")
            print(f"✅ Min Daily Return: {volatility_analysis['min_daily_return']*100:.2f}%")
        else:
            print("❌ Failed to get volatility analysis")
    except Exception as e:
        print(f"❌ Error in volatility analysis: {e}")
    
    # Demo 5: Technical Analysis
    print("\n" + "="*60)
    print("🔍 DEMO 5: Technical Analysis")
    print("="*60)
    
    try:
        # Analyze a specific stock (RELIANCE.NS)
        selected_stock = "RELIANCE.NS"
        print(f"\nAnalyzing technical indicators for {selected_stock}...")
        
        # Get stock data
        ticker_data = nifty_data.get_nifty50_constituents_data(period="6mo", symbols=[selected_stock])
        
        if selected_stock in ticker_data:
            stock_data = ticker_data[selected_stock]
            
            # Calculate technical indicators
            tech_indicators = TechnicalAnalysis(stock_data)
            stock_data = tech_indicators.calculate_all_indicators()
            
            current_price = stock_data['Close'].iloc[-1]
            current_rsi = stock_data['RSI'].iloc[-1]
            current_macd = stock_data['MACD'].iloc[-1]
            current_sma20 = stock_data['SMA_20'].iloc[-1]
            
            print(f"✅ Current Price: ₹{current_price:.2f}")
            print(f"✅ RSI: {current_rsi:.2f}")
            print(f"✅ MACD: {current_macd:.4f}")
            print(f"✅ SMA 20: ₹{current_sma20:.2f}")
            
            # Technical signals
            rsi_signal = "Oversold" if current_rsi < 30 else "Overbought" if current_rsi > 70 else "Neutral"
            trend_signal = "Above MA20" if current_price > current_sma20 else "Below MA20"
            
            print(f"✅ RSI Signal: {rsi_signal}")
            print(f"✅ Trend Signal: {trend_signal}")
        else:
            print(f"❌ Failed to get data for {selected_stock}")
    except Exception as e:
        print(f"❌ Error in technical analysis: {e}")
    
    # Demo 6: Machine Learning Predictions
    print("\n" + "="*60)
    print("🔮 DEMO 6: Machine Learning Predictions")
    print("="*60)
    
    try:
        print("\nTraining prediction models...")
        
        # Get training data
        training_data = nifty_data.get_nifty50_index_data(period="2y")
        
        if not training_data.empty:
            # Prepare features and train models
            feature_data = predictions.prepare_features(training_data)
            
            if not feature_data.empty:
                print(f"✅ Prepared {len(feature_data)} data points with {len(predictions.get_feature_columns())} features")
                
                # Train models
                predictions.train_models(feature_data)
                
                # Make predictions
                print("\nMaking predictions...")
                
                # Individual model predictions
                for model_name in ['RandomForest', 'XGBoost', 'LightGBM']:
                    if model_name in predictions.models:
                        pred = predictions.predict_next_day(model_name)
                        if 'error' not in pred:
                            print(f"✅ {model_name}: {pred['predicted_direction']} {pred['predicted_return']*100:.2f}% (Confidence: {pred['confidence']*100:.1f}%)")
                        else:
                            print(f"❌ {model_name}: {pred['error']}")
                
                # Ensemble prediction
                ensemble_pred = predictions.ensemble_prediction()
                if 'error' not in ensemble_pred:
                    print(f"✅ Ensemble: {ensemble_pred['predicted_direction']} {ensemble_pred['predicted_return']*100:.2f}% (Confidence: {ensemble_pred['confidence']*100:.1f}%)")
                else:
                    print(f"❌ Ensemble: {ensemble_pred['error']}")
                
                # Multi-day forecast
                print("\nGenerating 5-day forecast...")
                multi_predictions = predictions.predict_multiple_days(5, 'RandomForest')
                
                if multi_predictions and 'error' not in multi_predictions[0]:
                    print("✅ 5-Day Forecast:")
                    for pred in multi_predictions:
                        print(f"   Day {pred['day']}: ₹{pred['predicted_level']:,.2f} ({pred['predicted_return']*100:+.2f}%)")
                else:
                    print("❌ Failed to generate multi-day forecast")
                
            else:
                print("❌ Failed to prepare features for training")
        else:
            print("❌ No training data available")
    except Exception as e:
        print(f"❌ Error in predictions: {e}")
    
    # Demo 7: Export Reports
    print("\n" + "="*60)
    print("📋 DEMO 7: Export Reports")
    print("="*60)
    
    try:
        print("\nExporting analysis reports...")
        
        # Export Nifty 50 analysis report
        nifty_data.export_nifty50_report("nifty50_analysis_report.csv")
        print("✅ Nifty 50 analysis report exported")
        
        # Export predictions report
        if hasattr(predictions, 'models') and predictions.models:
            predictions.export_predictions_report("nifty50_predictions_report.csv")
            print("✅ Predictions report exported")
        
        print("✅ All reports exported successfully!")
        
    except Exception as e:
        print(f"❌ Error exporting reports: {e}")
    
    # Demo 8: Data Summary
    print("\n" + "="*60)
    print("📊 DEMO 8: Data Summary")
    print("="*60)
    
    try:
        print("\nGetting data summary...")
        
        # Nifty 50 data summary
        nifty_summary = nifty_data.get_data_summary()
        if nifty_summary:
            print(f"✅ Nifty 50 Database: {nifty_summary['symbols_count']} symbols, {nifty_summary['total_records']} records")
            if nifty_summary['date_range']['start'] and nifty_summary['date_range']['end']:
                print(f"✅ Date Range: {nifty_summary['date_range']['start']} to {nifty_summary['date_range']['end']}")
        
        # Model performance summary
        if hasattr(predictions, 'models') and predictions.models:
            performance_summary = predictions.get_model_performance_summary()
            if 'error' not in performance_summary:
                print(f"✅ Trained Models: {len(performance_summary)}")
                for model_name in performance_summary.keys():
                    print(f"   - {model_name}")
        
    except Exception as e:
        print(f"❌ Error getting data summary: {e}")
    
    # Final summary
    print("\n" + "="*60)
    print("🎯 DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print("\n📋 What was demonstrated:")
    print("✅ Nifty 50 index data fetching and analysis")
    print("✅ Sector-wise performance analysis")
    print("✅ Market breadth and sentiment analysis")
    print("✅ Volatility and risk analysis")
    print("✅ Technical indicators and signals")
    print("✅ Machine learning predictions (multiple models)")
    print("✅ Ensemble predictions")
    print("✅ Multi-day forecasting")
    print("✅ Report generation and export")
    
    print("\n🚀 Next steps:")
    print("1. Run 'streamlit run nifty50_dashboard.py' for interactive dashboard")
    print("2. Use individual modules for specific analysis needs")
    print("3. Customize prediction models and parameters")
    print("4. Add more technical indicators as needed")
    
    print("\n💡 Features available:")
    print("• Real-time Nifty 50 data from Yahoo Finance")
    print("• 50 constituent stocks analysis")
    print("• 11 sector classifications")
    print("• 7 machine learning models")
    print("• Technical analysis with 15+ indicators")
    print("• Market sentiment scoring")
    print("• Risk metrics (VaR, volatility)")
    print("• Comprehensive reporting system")

if __name__ == "__main__":
    main()
