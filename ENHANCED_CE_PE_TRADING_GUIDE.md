# 🚀 Enhanced CE & PE Intraday Nifty 50 Index Options Trading System

## 📊 **CE & PE Options Trading with Entry/Exit Strategies and Stop Loss Management**

### **🎯 What is Enhanced CE & PE Trading?**

Enhanced CE & PE trading involves sophisticated strategies for both CALL (CE) and PUT (PE) options with advanced risk management, position balancing, and complex strategies like straddles and strangles.

**Key Benefits:**
- ✅ **Dual Strategy**: Trade both CE & PE simultaneously
- ✅ **Risk Balancing**: Maintain CE/PE exposure balance
- ✅ **Complex Strategies**: Straddle, Strangle, and more
- ✅ **Advanced Risk Management**: Position limits and spread controls
- ✅ **Time-based Trading**: Optimize strategies by time slots
- ✅ **Professional Analytics**: Enhanced performance tracking

---

## 🕐 **Intraday Time Slots & CE & PE Strategies**

### **1. PRE_MARKET (9:00 - 9:15 AM)**
- **Purpose**: Market preparation and analysis
- **Strategy**: Plan CE & PE trades based on pre-market data
- **Risk Level**: LOW (No trading)

### **2. OPENING (9:15 - 9:30 AM)**
- **Purpose**: Capture opening momentum and gaps
- **Strategies**: 
  - **STRADDLE**: Buy both CE & PE at same strike for gap trading
  - **MOMENTUM_BREAKOUT**: Breakout trading in first 15 minutes
- **Risk Level**: HIGH
- **Best For**: Experienced traders
- **Strike Selection**: At-the-money (ATM)

### **3. MORNING (9:30 - 11:00 AM)**
- **Purpose**: Technical breakout and momentum trading
- **Strategies**:
  - **TECHNICAL_BREAKOUT**: Technical pattern breakouts
  - **STRANGLE**: Buy OTM CE & PE for volatility expansion
- **Risk Level**: MEDIUM
- **Best For**: Technical traders
- **Strike Selection**: Support/Resistance levels, OTM

### **4. MID_DAY (11:00 - 2:00 PM)**
- **Purpose**: Mean reversion and volatility trading
- **Strategies**:
  - **MEAN_REVERSION**: Price correction trades
  - **VOLATILITY_EXPANSION**: Volatility-based trades
- **Risk Level**: LOW to MEDIUM
- **Best For**: Conservative traders
- **Strike Selection**: Moving average levels, Volatility bands

### **5. AFTERNOON (2:00 - 3:00 PM)**
- **Purpose**: Position management and new setups
- **Strategy**: **MEAN_REVERSION**: End-of-day corrections
- **Risk Level**: LOW
- **Best For**: Conservative traders
- **Strike Selection**: Daily pivot points

### **6. CLOSING (3:00 - 3:30 PM)**
- **Purpose**: Close positions and end-of-day trades
- **Strategy**: **MEAN_REVERSION**: Final corrections
- **Risk Level**: LOW
- **Best For**: Conservative traders
- **Strike Selection**: Daily pivot points

---

## 🎯 **Enhanced CE & PE Trading Strategies**

### **1. MOMENTUM_BREAKOUT**
- **When**: Opening and morning sessions
- **How**: Enter when price breaks above resistance or below support
- **CE Strategy**: Buy CE on bullish breakout
- **PE Strategy**: Buy PE on bearish breakout
- **Stop Loss**: 15% below entry (for calls) or above entry (for puts)
- **Target**: 30% above entry (for calls) or below entry (for puts)

### **2. STRADDLE Strategy**
- **When**: Opening session, high volatility periods
- **How**: Buy both CE & PE at same strike price
- **Best For**: Earnings, news events, high volatility
- **Stop Loss**: 15% on both positions
- **Target**: 30% on both positions
- **Risk Level**: HIGH
- **Example**: Buy NIFTY25000CE and NIFTY25000PE

### **3. STRANGLE Strategy**
- **When**: Morning session, volatility expansion
- **How**: Buy OTM CE & PE at different strikes
- **Best For**: High volatility, range-bound markets
- **Stop Loss**: 15% on both positions
- **Target**: 30% on both positions
- **Risk Level**: MEDIUM
- **Example**: Buy NIFTY25100CE and NIFTY24900PE

### **4. TECHNICAL_BREAKOUT**
- **When**: Morning session
- **How**: Breakout from chart patterns (triangles, rectangles, flags)
- **CE Strategy**: Buy CE on bullish breakout
- **PE Strategy**: Buy PE on bearish breakout
- **Stop Loss**: Below pattern support/resistance
- **Target**: Pattern height projected from breakout point

### **5. MEAN_REVERSION**
- **When**: Mid-day and closing sessions
- **How**: Trade price corrections to moving averages
- **CE Strategy**: Buy CE when price oversold
- **PE Strategy**: Buy PE when price overbought
- **Stop Loss**: Beyond key support/resistance levels
- **Target**: Return to mean (moving average)

---

## 🛡️ **Enhanced Risk Management Features**

### **1. Position Sizing & Limits**
- **Max Risk per Trade**: 3% of account balance
- **Max Intraday Positions**: 5 concurrent positions
- **Max CE Positions**: 3 concurrent CE positions
- **Max PE Positions**: 3 concurrent PE positions
- **Max Spread Positions**: 2 complex strategies
- **Lot Size Calculation**: Based on risk amount and stop loss

### **2. Stop Loss Management**
- **Default Stop Loss**: 15% from entry price
- **Trailing Stop**: 5% trailing stop enabled
- **Time-based Exit**: Maximum 6 hours holding time
- **Multiple Exit Conditions**: Stop Loss, Target, Time, Trailing Stop

### **3. CE & PE Balance Management**
- **Balance Tracking**: Monitor CE vs PE exposure
- **Directional Bias Alert**: Warn when CE/PE imbalance > 2
- **Auto-balancing**: Suggest trades to maintain balance
- **Risk Exposure**: Track total directional risk

### **4. Daily Risk Limits**
- **Daily Loss Limit**: 5% of account balance
- **Position Monitoring**: Real-time monitoring of all positions
- **Auto-exit**: Automatic exit on stop loss, target, or time
- **Spread Risk Multiplier**: 1.5x risk for complex strategies

---

## 📱 **How to Use the Enhanced CE & PE System**

### **Step 1: Initialize Enhanced Trader**
```python
from nifty50_ce_pe_intraday_trading import EnhancedIntradayNifty50Trader

trader = EnhancedIntradayNifty50Trader("enhanced_ce_pe.db")
```

### **Step 2: Check Market Status & Get Recommendations**
```python
# Check if market is open
if trader.is_market_open():
    current_slot = trader.get_current_time_slot()
    print(f"Current time slot: {current_slot.value}")
    
    # Get CE & PE strategy recommendations
    recommendations = trader.get_ce_pe_strategy_recommendations(current_slot)
    for rec in recommendations:
        print(f"Strategy: {rec['strategy'].value}")
        print(f"Risk Level: {rec['risk_level']}")
        print(f"Strike Selection: {rec['strike_selection']}")
```

### **Step 3: Place Individual CE & PE Trades**
```python
# Place CE trade
ce_trade = trader.place_ce_pe_intraday_trade(
    contract=ce_contract,
    action="BUY",
    strategy=IntradayStrategy.MOMENTUM_BREAKOUT,
    time_slot=current_slot,
    entry_price=50.0,
    stop_loss_percentage=0.15,  # 15%
    target_percentage=0.30,     # 30%
    risk_amount=3000,           # Risk ₹3000
    max_holding_hours=6
)

# Place PE trade
pe_trade = trader.place_ce_pe_intraday_trade(
    contract=pe_contract,
    action="BUY",
    strategy=IntradayStrategy.MOMENTUM_BREAKOUT,
    time_slot=current_slot,
    entry_price=45.0,
    stop_loss_percentage=0.15,
    target_percentage=0.30,
    risk_amount=3000,
    max_holding_hours=6
)
```

### **Step 4: Place Complex Strategies**
```python
# Place Straddle Trade
straddle = trader.place_straddle_trade(
    strike_price=25000,
    expiry_date=expiry_date,
    time_slot=current_slot,
    entry_price_ce=ce_quote.mid_price,
    entry_price_pe=pe_quote.mid_price,
    stop_loss_percentage=0.15,
    target_percentage=0.30,
    quantity=1,
    risk_amount=5000
)
```

### **Step 5: Monitor CE & PE Positions**
```python
# Check if positions need to exit
positions_to_exit = trader.monitor_ce_pe_positions(current_price)

# Auto-exit positions
exited_positions = trader.auto_exit_ce_pe_positions(current_price)
```

### **Step 6: Get Enhanced Summary**
```python
summary = trader.get_enhanced_intraday_summary()
print(f"Active Positions: {summary['active_positions']}")
print(f"CE Positions: {summary['ce_positions']}")
print(f"PE Positions: {summary['pe_positions']}")
print(f"Spread Positions: {summary['spread_positions']}")
print(f"CE-PE Balance: {summary['ce_pe_balance']}")
print(f"Daily P&L: ₹{summary['daily_pnl']:,.2f}")
```

---

## 🔄 **Automatic Exit Conditions**

### **1. Stop Loss Exit**
- **Trigger**: Price hits stop loss level
- **Action**: Automatic position closure
- **Reason**: STOP_LOSS
- **Applies to**: Both CE & PE positions

### **2. Target Exit**
- **Trigger**: Price hits target level
- **Action**: Automatic position closure
- **Reason**: TARGET_HIT
- **Applies to**: Both CE & PE positions

### **3. Time-based Exit**
- **Trigger**: Maximum holding time reached
- **Action**: Automatic position closure
- **Reason**: TIME_BASED
- **Applies to**: All positions

### **4. Trailing Stop Exit**
- **Trigger**: Price hits trailing stop level
- **Action**: Automatic position closure
- **Reason**: TRAILING_STOP
- **Applies to**: All positions

### **5. Market Close Exit**
- **Trigger**: Market closing time (3:30 PM)
- **Action**: Force close all positions
- **Reason**: MARKET_CLOSE
- **Applies to**: All CE & PE positions

---

## 📊 **Enhanced Performance Analytics**

### **1. Trade Metrics**
- Total trades
- Winning trades
- Losing trades
- Win rate percentage

### **2. P&L Analysis**
- Total P&L
- Daily P&L
- Average P&L per trade
- Maximum drawdown

### **3. CE & PE Specific Metrics**
- CE positions count
- PE positions count
- Spread positions count
- CE-PE balance
- Directional bias analysis

### **4. Risk Metrics**
- Active positions
- Risk exposure
- Position duration
- Exit reasons analysis
- CE/PE balance tracking

---

## 🚀 **Running the Enhanced CE & PE Demo**

### **Option 1: Use Batch File**
```bash
# Double-click run_nifty50_options.bat
# Choose option 4: "Run Enhanced CE & PE Demo"
```

### **Option 2: Command Line**
```bash
python ce_pe_intraday_demo.py
```

---

## 📋 **Demo Scenarios Covered**

1. **Enhanced CE & PE Trading System Setup**
   - Market status and time analysis
   - CE & PE strategy recommendations
   - Enhanced trader initialization

2. **CE & PE Contract Selection**
   - Contract discovery and selection
   - Quote analysis for both CE & PE
   - Strike price analysis

3. **Individual CE & PE Trades**
   - Individual CE trade placement
   - Individual PE trade placement
   - Strategy-based trade execution

4. **Complex Strategy Implementation**
   - Straddle strategy demonstration
   - Risk parameter calculation
   - Multi-leg trade management

5. **Enhanced Position Monitoring**
   - CE & PE position tracking
   - Spread position monitoring
   - Market scenario analysis

6. **Advanced Risk Management**
   - Enhanced risk settings
   - CE/PE balance tracking
   - Position limits and controls

7. **Performance Analytics**
   - Enhanced trading summary
   - CE & PE performance metrics
   - Position exposure analysis

8. **Market Close Procedures**
   - Force close scenarios
   - CE/PE balance reset
   - End-of-day summary

---

## 💡 **Best Practices for Enhanced CE & PE Trading**

### **1. Strategy Selection**
- Choose strategies based on time slot
- Consider market conditions and volatility
- Use technical analysis for entry/exit
- Balance CE & PE exposure

### **2. Risk Management**
- Never risk more than 3% per trade
- Use tight stop losses (15% or less)
- Set realistic targets (30% or less)
- Monitor CE/PE balance continuously
- Limit spread positions to 2 maximum

### **3. Position Management**
- Monitor positions continuously
- Use trailing stops for profitable trades
- Don't hold positions overnight
- Keep position size manageable
- Balance CE & PE exposure

### **4. Complex Strategies**
- Use straddles for high volatility
- Use strangles for range-bound markets
- Monitor spread risk carefully
- Exit complex strategies before market close

---

## 🔧 **Customization Options**

### **1. Risk Parameters**
```python
trader.max_intraday_risk = 0.02          # 2% risk per trade
trader.max_intraday_positions = 7        # 7 concurrent positions
trader.max_ce_positions = 4              # 4 CE positions
trader.max_pe_positions = 4              # 4 PE positions
trader.max_spread_positions = 3          # 3 spread positions
trader.intraday_stop_loss_percentage = 0.10  # 10% stop loss
trader.intraday_target_percentage = 0.25     # 25% target
trader.max_holding_hours = 8             # 8 hours max holding
```

### **2. Time Slots**
- Customize market hours
- Adjust time slot boundaries
- Add custom time-based strategies
- Modify strategy recommendations

### **3. Exit Conditions**
- Modify stop loss percentages
- Adjust target percentages
- Customize trailing stop parameters
- Add custom exit conditions

---

## 🚨 **Important Notes**

### **1. Market Hours**
- **Market Open**: 9:15 AM
- **Market Close**: 3:30 PM
- **Pre-market**: 9:00 - 9:15 AM
- **Post-market**: 3:30 - 4:00 PM

### **2. Position Requirements**
- All positions must be closed before market close
- No overnight positions allowed
- Maximum 6 hours holding time per position
- Maintain CE/PE balance

### **3. Risk Limits**
- Daily loss limit: 5% of account
- Per trade risk: 3% of account
- Maximum positions: 5 concurrent
- Maximum CE positions: 3
- Maximum PE positions: 3
- Maximum spread positions: 2

---

## 🎯 **Next Steps**

1. **Run the Enhanced CE & PE Demo** to see all features in action
2. **Practice with Different Strategies** in various time slots
3. **Test Straddle and Strangle Strategies** for complex trading
4. **Customize Risk Parameters** for your trading style
5. **Test Different Market Scenarios** and exit conditions
6. **Integrate with Real-time Data** for live trading

---

## 📞 **Support & Troubleshooting**

### **Common Issues:**
1. **Market Closed Error**: Check current time vs market hours
2. **Position Limit Error**: Close existing positions first
3. **CE/PE Balance Error**: Balance your CE and PE exposure
4. **Spread Limit Error**: Close existing spread positions
5. **Risk Limit Error**: Reduce position size or risk amount
6. **Database Error**: Check file permissions and disk space

### **Getting Help:**
1. Check demo output for error messages
2. Review risk parameters and limits
3. Verify market hours and time slots
4. Check CE/PE balance and position limits
5. Verify database connectivity

---

**🎉 Happy Enhanced CE & PE Trading with Professional Risk Management! 🎉**

---

*Remember: Enhanced CE & PE trading requires discipline, quick decision-making, and strict risk management. Always close positions before market close, maintain CE/PE balance, and never risk more than you can afford to lose.*
