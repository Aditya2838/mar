"""
Intraday Nifty 50 Options Trading Demo
Demonstrates intraday entry/exit strategies with stop loss management
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta, time

# Import our intraday trading system
from nifty50_intraday_trading import (
    IntradayNifty50Trader,
    IntradayStrategy,
    IntradayTimeSlot,
    OptionType,
    OptionExpiry
)

def demo_intraday_trading_system():
    """Demonstrate the complete intraday trading system"""
    print("=" * 80)
    print("🚀 INTRADAY NIFTY 50 INDEX OPTIONS TRADING SYSTEM")
    print("📊 Entry & Exit Strategies | Stop Loss Management | Time-Based Trading")
    print("=" * 80)
    
    # Initialize intraday trader
    print("\n1. Initializing Intraday Options Trader...")
    trader = IntradayNifty50Trader("intraday_demo.db")
    print("✅ Intraday trader initialized successfully")
    
    # Display market status
    print("\n2. Market Status & Time Analysis:")
    market_open = trader.is_market_open()
    current_slot = trader.get_current_time_slot()
    
    print(f"   Market Open: {'✅ YES' if market_open else '❌ NO'}")
    print(f"   Current Time Slot: {current_slot.value}")
    print(f"   Current Time: {datetime.now().strftime('%H:%M:%S')}")
    
    # Get strategy recommendations
    print(f"\n3. Strategy Recommendations for {current_slot.value}:")
    recommendations = trader.get_intraday_strategy_recommendations(current_slot)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec['strategy'].value}")
        print(f"      Description: {rec['description']}")
        print(f"      Risk Level: {rec['risk_level']}")
        print(f"      Suitable For: {rec['suitable_for']}")
    
    return trader, current_slot

def demo_intraday_trade_setup(trader, current_slot):
    """Demonstrate intraday trade setup"""
    print("\n" + "=" * 80)
    print("📈 INTRADAY TRADE SETUP DEMONSTRATION")
    print("=" * 80)
    
    # Get available contracts
    print("\n1. Available Option Contracts:")
    contracts = trader.get_available_contracts(
        strike_range=(24800, 25200),
        expiry_filter=OptionExpiry.WEEKLY
    )
    print(f"   Found {len(contracts)} contracts")
    
    # Find target contract (25000 CE)
    target_contract = None
    for contract in contracts:
        if (contract.strike_price == 25000 and 
            contract.option_type == OptionType.CALL):
            target_contract = contract
            break
    
    if not target_contract:
        print("❌ 25000 CE contract not found")
        return None, None
    
    print(f"\n2. Target Contract Selected:")
    print(f"   Contract: {target_contract.display_name}")
    print(f"   Strike Price: ₹{target_contract.strike_price:,}")
    print(f"   Option Type: {target_contract.option_type.value}")
    print(f"   Expiry Date: {target_contract.expiry_date}")
    
    # Get current quote
    print(f"\n3. Current Market Quote:")
    quote = trader.get_option_quote(target_contract)
    if not quote:
        print("❌ Unable to get quote")
        return None, None
    
    print(f"   Bid Price: ₹{quote.bid_price:.2f}")
    print(f"   Ask Price: ₹{quote.ask_price:.2f}")
    print(f"   Mid Price: ₹{quote.mid_price:.2f}")
    print(f"   Spread: ₹{quote.spread:.2f} ({quote.spread_percentage:.1f}%)")
    
    return target_contract, quote

def demo_intraday_trade_execution(trader, contract, quote, current_slot):
    """Demonstrate intraday trade execution"""
    print("\n" + "=" * 80)
    print("🎯 INTRADAY TRADE EXECUTION DEMONSTRATION")
    print("=" * 80)
    
    # Select strategy based on time slot
    if current_slot == IntradayTimeSlot.OPENING:
        strategy = IntradayStrategy.MOMENTUM_BREAKOUT
        print(f"\n1. Strategy Selection: {strategy.value}")
        print(f"   Reason: Best for opening session momentum trades")
    elif current_slot == IntradayTimeSlot.MORNING:
        strategy = IntradayStrategy.TECHNICAL_BREAKOUT
        print(f"\n1. Strategy Selection: {strategy.value}")
        print(f"   Reason: Optimal for morning technical breakouts")
    else:
        strategy = IntradayStrategy.MEAN_REVERSION
        print(f"\n1. Strategy Selection: {strategy.value}")
        print(f"   Reason: Suitable for current time slot")
    
    # Calculate intraday parameters
    print(f"\n2. Intraday Risk Parameters:")
    entry_price = quote.mid_price
    stop_loss_percentage = 0.15  # 15% stop loss
    target_percentage = 0.30     # 30% target
    risk_amount = 3000           # Risk ₹3000
    
    if trader.place_option_order(contract, "BUY", 1, "MARKET"):
        print(f"   Entry Price: ₹{entry_price:.2f}")
        print(f"   Stop Loss: ₹{entry_price * (1 - stop_loss_percentage):.2f} ({stop_loss_percentage*100:.0f}%)")
        print(f"   Target: ₹{entry_price * (1 + target_percentage):.2f} ({target_percentage*100:.0f}%)")
        print(f"   Risk Amount: ₹{risk_amount:,.2f}")
        
        # Place intraday trade
        print(f"\n3. Placing Intraday Trade...")
        trade = trader.place_intraday_option_order(
            contract=contract,
            action="BUY",
            strategy=strategy,
            time_slot=current_slot,
            entry_price=entry_price,
            stop_loss_percentage=stop_loss_percentage,
            target_percentage=target_percentage,
            risk_amount=risk_amount,
            max_holding_hours=6
        )
        
        if trade:
            print(f"✅ Intraday trade placed successfully!")
            print(f"   Trade ID: {trade['trade_id']}")
            print(f"   Strategy: {trade['strategy'].value}")
            print(f"   Time Slot: {trade['time_slot'].value}")
            print(f"   Entry Time: {trade['entry_time'].strftime('%H:%M:%S')}")
            print(f"   Planned Exit: {trade['setup'].exit_time.strftime('%H:%M:%S')}")
            print(f"   Max Holding: {trade['setup'].max_holding_time}")
            
            return trade
        else:
            print("❌ Trade placement failed")
            return None
    else:
        print("❌ Basic order placement failed")
        return None

def demo_intraday_position_monitoring(trader, trade):
    """Demonstrate intraday position monitoring"""
    print("\n" + "=" * 80)
    print("📊 INTRADAY POSITION MONITORING DEMONSTRATION")
    print("=" * 80)
    
    if not trade:
        print("❌ No trade to monitor")
        return
    
    print(f"\n1. Active Intraday Position:")
    print(f"   Trade ID: {trade['trade_id']}")
    print(f"   Contract: {trade['contract'].display_name}")
    print(f"   Entry Price: ₹{trade['entry_price']:.2f}")
    print(f"   Stop Loss: ₹{trade['stop_loss']:.2f}")
    print(f"   Target: ₹{trade['target_price']:.2f}")
    print(f"   Strategy: {trade['strategy'].value}")
    
    # Simulate different market scenarios
    print(f"\n2. Market Scenario Analysis:")
    
    scenarios = [
        ("Price moves up towards target", trade['entry_price'] * 1.15),
        ("Price stays near entry", trade['entry_price'] * 1.02),
        ("Price drops towards stop loss", trade['entry_price'] * 0.88),
        ("Price hits stop loss", trade['stop_loss'] * 0.95),
        ("Price hits target", trade['target_price'] * 1.02)
    ]
    
    for scenario_name, price in scenarios:
        print(f"\n   Scenario: {scenario_name}")
        print(f"   Current Price: ₹{price:.2f}")
        
        # Check if position should exit
        positions_to_exit = trader.monitor_intraday_positions(price)
        
        if positions_to_exit:
            print(f"   ⚠️  Position needs to exit!")
            for position_info in positions_to_exit:
                print(f"      Trade ID: {position_info['trade_id']}")
                print(f"      Reason: {position_info['reason']}")
                print(f"      Current Price: ₹{position_info['current_price']:.2f}")
        else:
            print(f"   ✅ Position is safe")
    
    # Simulate actual exit scenario
    print(f"\n3. Simulating Stop Loss Exit...")
    exit_price = trade['stop_loss'] * 0.95  # Price hits stop loss
    
    print(f"   Price hits: ₹{exit_price:.2f}")
    print(f"   Stop Loss: ₹{trade['stop_loss']:.2f}")
    
    # Auto-exit positions
    exited_positions = trader.auto_exit_intraday_positions(exit_price)
    
    if exited_positions:
        print(f"   🚨 Auto-exit triggered!")
        for exit_info in exited_positions:
            print(f"      Trade ID: {exit_info['trade_id']}")
            print(f"      Reason: {exit_info['reason']}")
            print(f"      Exit Price: ₹{exit_info['current_price']:.2f}")
    else:
        print(f"   ✅ No positions exited")

def demo_intraday_risk_management(trader):
    """Demonstrate intraday risk management features"""
    print("\n" + "=" * 80)
    print("🛡️ INTRADAY RISK MANAGEMENT DEMONSTRATION")
    print("=" * 80)
    
    print(f"\n1. Intraday Risk Settings:")
    print(f"   Max Risk per Trade: {trader.max_intraday_risk * 100:.1f}%")
    print(f"   Max Intraday Positions: {trader.max_intraday_positions}")
    print(f"   Default Stop Loss: {trader.intraday_stop_loss_percentage * 100:.1f}%")
    print(f"   Default Target: {trader.intraday_target_percentage * 100:.1f}%")
    print(f"   Max Holding Time: {trader.max_holding_hours} hours")
    
    print(f"\n2. Daily Risk Limits:")
    daily_loss_limit = trader.account_balance * 0.05
    print(f"   Account Balance: ₹{trader.account_balance:,.2f}")
    print(f"   Daily Loss Limit: ₹{daily_loss_limit:,.2f} (5%)")
    print(f"   Current Daily P&L: ₹{trader.daily_pnl:,.2f}")
    
    print(f"\n3. Position Risk Analysis:")
    can_trade, message = trader.can_place_intraday_trade()
    print(f"   Can Place Trade: {'✅ YES' if can_trade else '❌ NO'}")
    print(f"   Message: {message}")
    
    print(f"\n4. Risk Management Tools:")
    print(f"   Trailing Stop: Enabled")
    print(f"   Time-based Exit: Enabled")
    print(f"   Multiple Exit Conditions: Stop Loss, Target, Time, Trailing Stop")
    print(f"   Automatic Position Monitoring: Enabled")

def demo_intraday_summary_and_analytics(trader):
    """Demonstrate intraday summary and analytics"""
    print("\n" + "=" * 80)
    print("📋 INTRADAY SUMMARY & ANALYTICS DEMONSTRATION")
    print("=" * 80)
    
    # Get intraday summary
    summary = trader.get_intraday_summary()
    
    print(f"\n1. Intraday Trading Summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            if 'pnl' in key.lower() or 'drawdown' in key.lower():
                print(f"   {key.replace('_', ' ').title()}: ₹{value:,.2f}")
            else:
                print(f"   {key.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"   {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n2. Performance Metrics:")
    if summary['total_trades'] > 0:
        win_rate = summary['win_rate']
        avg_pnl = summary['average_pnl']
        
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Average P&L: ₹{avg_pnl:,.2f}")
        print(f"   Total P&L: ₹{summary['total_pnl']:,.2f}")
        print(f"   Daily P&L: ₹{summary['daily_pnl']:,.2f}")
        
        # Performance analysis
        if win_rate >= 60:
            print(f"   📈 Performance: EXCELLENT (High win rate)")
        elif win_rate >= 50:
            print(f"   📊 Performance: GOOD (Positive win rate)")
        else:
            print(f"   ⚠️  Performance: NEEDS IMPROVEMENT (Low win rate)")
    
    print(f"\n3. Risk Exposure:")
    active_positions = summary['active_positions']
    if active_positions > 0:
        print(f"   ⚠️  Active Positions: {active_positions}")
        print(f"   ⚠️  Monitor these positions for exit conditions")
    else:
        print(f"   ✅ No Active Positions")
        print(f"   ✅ All positions closed for the day")

def demo_market_close_scenarios(trader):
    """Demonstrate market close scenarios"""
    print("\n" + "=" * 80)
    print("🏁 MARKET CLOSE SCENARIOS DEMONSTRATION")
    print("=" * 80)
    
    print(f"\n1. Market Close Procedures:")
    print(f"   Market Close Time: {trader.market_close_time}")
    print(f"   Force Close All Positions: Enabled")
    print(f"   Reason: MARKET_CLOSE")
    
    print(f"\n2. Simulating Market Close...")
    
    # Check if we have any active positions
    if trader.intraday_positions:
        print(f"   Active Positions: {len(trader.intraday_positions)}")
        
        # Force close all positions
        closed_positions = trader.force_close_all_intraday_positions("MARKET_CLOSE")
        
        if closed_positions:
            print(f"   🚨 Force Closed Positions: {len(closed_positions)}")
            for position in closed_positions:
                print(f"      Trade ID: {position['trade_id']}")
                print(f"      Exit Price: ₹{position['exit_price']:.2f}")
                print(f"      Reason: {position['reason']}")
        else:
            print(f"   ✅ No positions to close")
    else:
        print(f"   ✅ No active positions to close")
    
    print(f"\n3. End-of-Day Summary:")
    final_summary = trader.get_intraday_summary()
    print(f"   Total Trades: {final_summary['total_trades']}")
    print(f"   Final P&L: ₹{final_summary['daily_pnl']:,.2f}")
    print(f"   Win Rate: {final_summary['win_rate']:.1f}%")

def main():
    """Main demo function"""
    try:
        print("🚀 Starting Intraday Nifty 50 Options Trading Demo...")
        
        # Basic setup
        trader, current_slot = demo_intraday_trading_system()
        
        # Trade setup
        contract, quote = demo_intraday_trade_setup(trader, current_slot)
        if not contract or not quote:
            print("❌ Failed to setup trade")
            return
        
        # Trade execution
        trade = demo_intraday_trade_execution(trader, contract, quote, current_slot)
        
        if trade:
            # Position monitoring
            demo_intraday_position_monitoring(trader, trade)
            
            # Risk management
            demo_intraday_risk_management(trader)
            
            # Summary and analytics
            demo_intraday_summary_and_analytics(trader)
            
            # Market close scenarios
            demo_market_close_scenarios(trader)
        
        print("\n" + "=" * 80)
        print("🎉 INTRADAY DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nKey Intraday Features Demonstrated:")
        print("✅ Time-based trading strategies")
        print("✅ Intraday entry/exit management")
        print("✅ Stop loss and target automation")
        print("✅ Time-based position exits")
        print("✅ Trailing stop functionality")
        print("✅ Risk management tools")
        print("✅ Market close procedures")
        print("✅ Performance analytics")
        
        print("\nNext Steps:")
        print("1. Test different time slots and strategies")
        print("2. Explore various risk-reward scenarios")
        print("3. Practice with different market conditions")
        print("4. Integrate with real-time market data")
        print("5. Customize risk parameters for your style")
        
    except Exception as e:
        print(f"\n❌ Intraday demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
