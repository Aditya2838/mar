# 📈 Nifty 50 Options Trading System

A comprehensive Python-based system for trading Nifty 50 index options with advanced features including real-time pricing, strategy building, and portfolio management.

## 🚀 Features

### Core Trading Functionality
- **Options Contract Management**: Create and manage Nifty 50 option contracts (CE/PE)
- **Real-time Pricing**: Get live option quotes with Greeks (Delta, Gamma, Theta, Vega)
- **Order Execution**: Place buy/sell orders for options with market and limit orders
- **Position Management**: Track open positions, calculate P&L, and manage risk
- **Trade History**: Complete audit trail of all executed trades

### Advanced Analytics
- **Options Chain**: View complete options chain for different expiries
- **Payoff Analysis**: Calculate and visualize payoff diagrams for complex strategies
- **Strategy Builder**: Build multi-leg option strategies (straddles, spreads, etc.)
- **Risk Management**: Monitor position exposure and calculate maximum loss/profit

### User Interface
- **Streamlit Dashboard**: Modern web-based interface for easy trading
- **Real-time Updates**: Live data updates and position monitoring
- **Responsive Design**: Works on desktop and mobile devices
- **Interactive Charts**: Plotly-based charts for data visualization

## 🏗️ Architecture

```
nifty50_options_trading.py     # Core trading engine
├── OptionContract             # Option contract data structure
├── OptionQuote               # Real-time quote data
├── OptionTrade               # Trade execution tracking
├── Nifty50OptionsTrader      # Main trading class
└── Database management       # SQLite-based data persistence

nifty50_options_dashboard.py   # Streamlit web interface
├── OptionsDashboard          # Main dashboard class
├── Options Chain view        # Browse available contracts
├── Position Management       # Monitor open positions
├── Trade History            # View executed trades
├── Payoff Analyzer          # Strategy analysis
└── Market Data              # Real-time market information

nifty50_options_demo.py       # Demonstration script
├── Basic trading demo        # Buy/sell options
├── Position management       # Track and manage positions
├── Payoff analysis          # Strategy payoff calculation
└── Options chain demo       # Market data exploration
```

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd stock_market_python
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements_options.txt
   ```

3. **Run the demo**
   ```bash
   python nifty50_options_demo.py
   ```

4. **Launch the dashboard**
   ```bash
   streamlit run nifty50_options_dashboard.py
   ```

## 🎯 Quick Start

### 1. Basic Options Trading

```python
from nifty50_options_trading import Nifty50OptionsTrader, OptionType

# Initialize trader
trader = Nifty50OptionsTrader()

# Get available contracts
contracts = trader.get_available_contracts(
    strike_range=(24800, 25200),
    expiry_filter=OptionExpiry.WEEKLY
)

# Find 25000 CE contract
call_contract = next(c for c in contracts 
                    if c.strike_price == 25000 and c.option_type == OptionType.CALL)

# Get quote
quote = trader.get_option_quote(call_contract)
print(f"Bid: ₹{quote.bid_price}, Ask: ₹{quote.ask_price}")

# Place buy order
trade = trader.place_option_order(
    contract=call_contract,
    action="BUY",
    quantity=1  # 1 lot
)
```

### 2. Position Management

```python
# Get current positions
positions = trader.get_current_positions()

# Calculate P&L
for position_id, position in positions.items():
    contract = position['contract']
    current_quote = trader.get_option_quote(contract)
    
    current_value = position['quantity'] * current_quote.mid_price * contract.lot_size
    cost_basis = position['quantity'] * position['average_price'] * contract.lot_size
    pnl = current_value - cost_basis
    
    print(f"P&L: ₹{pnl:,.2f}")

# Close position
trader.close_position(position_id)
```

### 3. Strategy Building

```python
# Create a straddle strategy
call_contract = # ... get call contract
put_contract = # ... get put contract

# Calculate payoff
payoff_data = trader.calculate_payoff(
    contracts=[call_contract, put_contract],
    quantities=[1, 1],  # Long 1 lot each
    spot_range=(24000, 26000),
    spot_step=100
)

# Find breakeven points
breakeven_points = payoff_data[payoff_data['breakeven']]
print(f"Breakeven: {breakeven_points['spot_price'].tolist()}")
```

## 📊 Dashboard Features

### Main Dashboard
- **Account Summary**: Balance, positions, P&L overview
- **Current Positions**: Real-time position monitoring
- **Trade History**: Complete trade log

### Options Chain
- **Contract Browser**: Filter by strike, expiry, and option type
- **Real-time Quotes**: Live bid/ask prices and Greeks
- **Trading Interface**: One-click buy/sell buttons

### Position Management
- **P&L Tracking**: Real-time profit/loss calculation
- **Risk Metrics**: Position exposure and risk analysis
- **Close Positions**: Easy position closing interface

### Payoff Analyzer
- **Strategy Builder**: Multi-leg option strategy construction
- **Payoff Visualization**: Interactive payoff diagrams
- **Risk Analysis**: Maximum profit/loss calculation

### Market Data
- **Options Chain Table**: Complete market data view
- **Implied Volatility**: IV analysis across strikes
- **Volume & OI**: Market activity indicators

## 🔧 Configuration

### Database Settings
```python
# Default database path
DATABASE_PATH = "nifty50_options.db"

# Custom database path
trader = Nifty50OptionsTrader("custom_database.db")
```

### Account Settings
```python
# Default account balance
DEFAULT_ACCOUNT_BALANCE = 100000

# Lot size (Nifty 50 = 50 shares per lot)
LOT_SIZE = 50
```

### Market Settings
```python
# Current Nifty 50 level
nifty50_current_level = 25000

# Strike price range
strike_range = (24000, 26000)

# Expiry periods
expiry_filter = OptionExpiry.WEEKLY  # WEEKLY, MONTHLY, QUARTERLY
```

## 📈 Trading Strategies

### 1. Long Call
- **Use Case**: Bullish outlook on Nifty 50
- **Risk**: Limited to premium paid
- **Reward**: Unlimited upside potential

### 2. Long Put
- **Use Case**: Bearish outlook on Nifty 50
- **Risk**: Limited to premium paid
- **Reward**: Unlimited downside potential

### 3. Long Straddle
- **Use Case**: Expecting high volatility
- **Risk**: Limited to total premium paid
- **Reward**: Unlimited on both sides

### 4. Bull Call Spread
- **Use Case**: Moderately bullish outlook
- **Risk**: Limited to net premium paid
- **Reward**: Limited to difference between strikes

### 5. Iron Condor
- **Use Case**: Low volatility expectation
- **Risk**: Limited to net premium received
- **Reward**: Limited to premium received

## 🛡️ Risk Management

### Position Sizing
- Maximum 5% of account per trade
- Diversify across different strategies
- Monitor correlation between positions

### Stop Loss
- Set automatic stop losses
- Monitor delta exposure
- Regular position review

### Portfolio Limits
- Maximum 20% in options
- Maximum 10% in any single strategy
- Regular rebalancing

## 🔌 API Integration

### Current Implementation
- **Synthetic Data**: Simulated option pricing for demonstration
- **Local Database**: SQLite for data persistence
- **Offline Mode**: Works without internet connection

### Future Enhancements
- **Broker APIs**: Integration with Zerodha, Upstox, etc.
- **Real-time Data**: Live market data feeds
- **Order Routing**: Direct order execution
- **Risk Management**: Real-time position monitoring

## 🧪 Testing

### Demo Mode
```bash
python nifty50_options_demo.py
```

### Unit Tests
```bash
python -m pytest tests/
```

### Integration Tests
```bash
python -m pytest tests/integration/
```

## 📚 Documentation

### Code Documentation
- Comprehensive docstrings for all classes and methods
- Type hints for better code understanding
- Example usage in docstrings

### User Guide
- Step-by-step trading instructions
- Strategy examples and explanations
- Risk management guidelines

### API Reference
- Complete method documentation
- Parameter descriptions
- Return value explanations

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Code Standards
- Follow PEP 8 style guide
- Add type hints
- Write comprehensive tests
- Update documentation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

**This software is for educational and demonstration purposes only. It is not intended for actual trading or investment decisions.**

- The system uses simulated data and pricing
- No real money is involved
- Always consult with financial advisors before trading
- Past performance does not guarantee future results
- Options trading involves substantial risk

## 🆘 Support

### Issues
- Report bugs via GitHub Issues
- Include system information and error logs
- Provide steps to reproduce the issue

### Questions
- Check the documentation first
- Search existing issues
- Create a new issue for questions

### Feature Requests
- Describe the desired functionality
- Explain the use case
- Provide examples if possible

## 🔮 Roadmap

### Phase 1 (Current)
- ✅ Basic options trading functionality
- ✅ Position management
- ✅ Payoff analysis
- ✅ Streamlit dashboard

### Phase 2 (Next)
- 🔄 Real-time data integration
- 🔄 Advanced strategy builder
- 🔄 Risk management tools
- 🔄 Performance analytics

### Phase 3 (Future)
- 📋 Broker API integration
- 📋 Paper trading mode
- 📋 Mobile app
- 📋 Social trading features

---

**Happy Trading! 📈💰**
