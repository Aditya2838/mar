#!/usr/bin/env python3
"""
Quick Start Script for Stock Market Analysis System
Easy setup and usage guide
"""

import sys
import os
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        # Install from requirements.txt
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def test_imports():
    """Test if all required modules can be imported"""
    print("\n🔍 Testing imports...")
    
    required_modules = [
        'yfinance', 'pandas', 'numpy', 'matplotlib', 'seaborn', 
        'plotly', 'scikit-learn', 'ta', 'streamlit'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠️ Failed to import: {', '.join(failed_imports)}")
        return False
    
    print("✅ All imports successful")
    return True

def run_basic_test():
    """Run a basic test of the system"""
    print("\n🧪 Running basic test...")
    
    try:
        # Test stock data fetching
        from data.stock_data import StockData
        
        stock = StockData("AAPL")
        data = stock.get_historical_data(period="1mo")
        
        if not data.empty:
            print(f"✅ Successfully fetched {len(data)} data points for AAPL")
            print(f"   Date range: {data['Date'].min()} to {data['Date'].max()}")
            print(f"   Current price: ${data['Close'].iloc[-1]:.2f}")
            return True
        else:
            print("❌ No data retrieved")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def show_usage_examples():
    """Show usage examples"""
    print("\n📚 Usage Examples:")
    print("=" * 50)
    
    print("\n1. Run basic analysis:")
    print("   python main.py --mode analysis")
    
    print("\n2. Start web dashboard:")
    print("   python main.py --mode web")
    print("   # or")
    print("   streamlit run streamlit_app.py")
    
    print("\n3. Export data:")
    print("   python main.py --mode export")
    
    print("\n4. Direct Python usage:")
    print("   python")
    print("   >>> from data.stock_data import StockData")
    print("   >>> stock = StockData('AAPL')")
    print("   >>> data = stock.get_historical_data()")
    print("   >>> print(data.head())")
    
    print("\n5. Technical Analysis:")
    print("   >>> from analysis.technical_indicators import TechnicalAnalysis")
    print("   >>> ta = TechnicalAnalysis(data)")
    print("   >>> rsi = ta.calculate_rsi()")
    print("   >>> print(f'Current RSI: {rsi.iloc[-1]:.2f}')")
    
    print("\n6. Machine Learning Predictions:")
    print("   >>> from analysis.ml_models import PricePredictor")
    print("   >>> predictor = PricePredictor('AAPL')")
    print("   >>> results = predictor.train_models(data)")
    print("   >>> prediction = predictor.predict_next_day(data)")
    
    print("\n7. Trading Strategies:")
    print("   >>> from trading.strategies import MovingAverageStrategy")
    print("   >>> strategy = MovingAverageStrategy('AAPL')")
    print("   >>> results = strategy.backtest(data)")
    print("   >>> print(f'Total Return: {results[\"total_return\"]:.2f}%')")

def create_sample_script():
    """Create a sample analysis script"""
    sample_script = '''#!/usr/bin/env python3
"""
Sample Stock Analysis Script
"""

from data.stock_data import StockData
from analysis.technical_indicators import TechnicalAnalysis
from analysis.ml_models import PricePredictor
from trading.strategies import MovingAverageStrategy

def analyze_stock(symbol):
    """Analyze a stock comprehensively"""
    print(f"📊 Analyzing {symbol}...")
    
    # Get data
    stock = StockData(symbol)
    data = stock.get_historical_data(period="1y")
    data.set_index('Date', inplace=True)
    
    # Technical analysis
    ta = TechnicalAnalysis(data)
    rsi = ta.calculate_rsi()
    macd = ta.calculate_macd()
    
    print(f"Current RSI: {rsi.iloc[-1]:.2f}")
    print(f"Current MACD: {macd['macd_line'].iloc[-1]:.4f}")
    
    # Machine learning prediction
    predictor = PricePredictor(symbol)
    results = predictor.train_models(data)
    prediction = predictor.predict_next_day(data)
    
    if prediction:
        current_price = prediction['current_price']
        predicted_price = prediction['predictions'].get('ensemble', current_price)
        print(f"Current Price: ${current_price:.2f}")
        print(f"Predicted Price: ${predicted_price:.2f}")
    
    # Trading strategy
    strategy = MovingAverageStrategy(symbol)
    backtest_results = strategy.backtest(data)
    
    print(f"Strategy Return: {backtest_results.get('total_return', 0):.2f}%")
    print(f"Buy & Hold Return: {backtest_results.get('buy_hold_return', 0):.2f}%")

if __name__ == "__main__":
    analyze_stock("AAPL")
'''
    
    with open("sample_analysis.py", "w") as f:
        f.write(sample_script)
    
    print("✅ Created sample_analysis.py")

def main():
    """Main quick start function"""
    print("🚀 Stock Market Analysis System - Quick Start")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        return
    
    # Test imports
    if not test_imports():
        print("❌ Import test failed")
        return
    
    # Run basic test
    if not run_basic_test():
        print("❌ Basic test failed")
        return
    
    # Create sample script
    create_sample_script()
    
    # Show usage examples
    show_usage_examples()
    
    print("\n🎉 Setup complete! You can now use the stock market analysis system.")
    print("\n💡 Quick tips:")
    print("   - Use 'python main.py --mode web' to start the web dashboard")
    print("   - Use 'python sample_analysis.py' to run a sample analysis")
    print("   - Check the README.md for detailed documentation")
    print("   - Modify config.py to customize settings")

if __name__ == "__main__":
    main()
