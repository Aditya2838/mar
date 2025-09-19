#!/usr/bin/env python3
"""
Intraday Trading Demo Script
Command-line interface for Nifty 50 intraday analysis and predictions
"""

import sys
import time
from datetime import datetime
import logging

# Add the project root to the path
sys.path.append('.')

from analysis.intraday_trading import IntradayTradingAnalysis
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_header():
    """Print the demo header"""
    print("=" * 80)
    print("📈 NIFTY 50 INTRADAY TRADING ANALYSIS & PREDICTIONS")
    print("=" * 80)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

def print_stock_analysis(analysis):
    """Print detailed stock analysis"""
    if 'error' in analysis:
        print(f"❌ Error: {analysis['error']}")
        return
    
    print(f"\n📊 {analysis['symbol']} - Intraday Analysis")
    print("-" * 60)
    
    # Current status
    print(f"💰 Current Price: ₹{analysis['current_price']}")
    print(f"📈 Momentum: {analysis['current_momentum']:.2f}%")
    print(f"📊 Volatility: {analysis['current_volatility']:.4f}")
    
    # Trading signals
    signals = analysis['signals']
    print(f"\n🎯 Trading Signal: {signals['signal']}")
    print(f"🎯 Confidence: {signals['confidence']}%")
    print(f"🎯 Buy Strength: {signals['buy_strength']:.2f}")
    print(f"🎯 Sell Strength: {signals['sell_strength']:.2f}")
    print(f"🎯 Reasons: {signals['reason']}")
    
    # Market timing
    timing = analysis['timing']
    print(f"\n⏰ Market Timing: {timing['timing']}")
    print(f"⏰ Reason: {timing['reason']}")
    print(f"⏰ Best Volume Hour: {timing['best_volume_hour']}:00")
    print(f"⏰ Best Momentum Hour: {timing['best_momentum_hour']}:00")
    
    # Price predictions
    print(f"\n🔮 Price Predictions:")
    for pred in analysis['predictions']:
        print(f"   {pred['interval']} intervals: ₹{pred['predicted_price']} ({pred['predicted_change']:+.2f}%)")
    
    print(f"\n🕐 Last Updated: {analysis['last_updated']}")

def print_summary(summary):
    """Print Nifty 50 summary"""
    if 'error' in summary:
        print(f"❌ Summary Error: {summary['error']}")
        return
    
    print(f"\n📋 NIFTY 50 INTRADAY SUMMARY")
    print("=" * 60)
    print(f"📊 Total Stocks: {summary['total_stocks']}")
    print(f"📊 Analyzed: {summary['stocks_analyzed']}")
    print(f"🟢 BUY Signals: {summary['buy_signals']}")
    print(f"🔴 SELL Signals: {summary['sell_signals']}")
    print(f"🟡 HOLD Signals: {summary['hold_signals']}")
    
    if summary['stock_details']:
        print(f"\n🏆 Top Signals by Confidence:")
        top_stocks = sorted(
            summary['stock_details'],
            key=lambda x: x['confidence'],
            reverse=True
        )[:10]
        
        for i, stock in enumerate(top_stocks, 1):
            signal_emoji = {
                'BUY': '🟢',
                'SELL': '🔴',
                'HOLD': '🟡'
            }.get(stock['signal'], '⚪')
            
            print(f"   {i:2d}. {signal_emoji} {stock['symbol']:<15} - {stock['signal']:<4} ({stock['confidence']:2d}%) - ₹{stock['current_price']}")
    
    print(f"\n🕐 Summary Time: {summary['timestamp']}")

def main():
    """Main demo function"""
    try:
        print_header()
        
        # Initialize intraday analysis
        print("\n🚀 Initializing Intraday Trading Analysis...")
        intraday_analysis = IntradayTradingAnalysis()
        print("✅ Intraday analysis initialized successfully!")
        
        # Demo individual stock analysis
        print("\n" + "="*80)
        print("📊 INDIVIDUAL STOCK ANALYSIS DEMO")
        print("="*80)
        
        # Analyze a few sample stocks
        sample_stocks = Config.NIFTY_50_SYMBOLS[:5]  # First 5 stocks
        
        for symbol in sample_stocks:
            try:
                print(f"\n🔄 Analyzing {symbol}...")
                analysis = intraday_analysis.get_intraday_predictions(symbol, prediction_horizon=4)
                print_stock_analysis(analysis)
                
                # Small delay between stocks
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Error analyzing {symbol}: {str(e)}")
                continue
        
        # Demo Nifty 50 summary
        print("\n" + "="*80)
        print("📋 NIFTY 50 SUMMARY DEMO")
        print("="*80)
        
        print("🔄 Generating Nifty 50 intraday summary...")
        summary = intraday_analysis.get_nifty50_intraday_summary()
        print_summary(summary)
        
        # Interactive mode
        print("\n" + "="*80)
        print("🎮 INTERACTIVE MODE")
        print("="*80)
        
        while True:
            print("\nOptions:")
            print("1. Analyze specific stock")
            print("2. Get Nifty 50 summary")
            print("3. Exit")
            
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                symbol = input("Enter stock symbol (e.g., RELIANCE.NS): ").strip()
                if symbol:
                    try:
                        print(f"\n🔄 Analyzing {symbol}...")
                        analysis = intraday_analysis.get_intraday_predictions(symbol, prediction_horizon=4)
                        print_stock_analysis(analysis)
                    except Exception as e:
                        print(f"❌ Error: {str(e)}")
                else:
                    print("❌ Please enter a valid symbol")
            
            elif choice == '2':
                try:
                    print("🔄 Generating summary...")
                    summary = intraday_analysis.get_nifty50_intraday_summary()
                    print_summary(summary)
                except Exception as e:
                    print(f"❌ Error: {str(e)}")
            
            elif choice == '3':
                print("\n👋 Thank you for using Nifty 50 Intraday Trading Analysis!")
                break
            
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
        
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Demo error: {str(e)}")
        logger.error(f"Demo error: {str(e)}")
    finally:
        print("\n" + "="*80)
        print(f"🏁 Demo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

if __name__ == "__main__":
    main()
