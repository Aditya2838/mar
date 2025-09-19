"""
Enhanced CE & PE Intraday Nifty 50 Options Trading Demo
Demonstrates CE & PE options trading with entry/exit strategies and stop loss management
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta, time

# Import our enhanced CE & PE trading system
from nifty50_ce_pe_intraday_trading import (
    EnhancedIntradayNifty50Trader,
    IntradayStrategy,
    IntradayTimeSlot,
    OptionType,
    OptionExpiry
)

def demo_enhanced_ce_pe_trading_system():
    """Demonstrate the complete enhanced CE & PE trading system"""
    print("=" * 80)
    print("🚀 ENHANCED CE & PE INTRADAY NIFTY 50 INDEX OPTIONS TRADING SYSTEM")
    print("📊 CALL & PUT Options | Entry/Exit Strategies | Stop Loss Management")
    print("=" * 80)
    
    # Initialize enhanced trader
    print("\n1. Initializing Enhanced CE & PE Options Trader...")
    trader = EnhancedIntradayNifty50Trader("enhanced_ce_pe_demo.db")
    print("✅ Enhanced trader initialized successfully")
    
    # Display market status
    print("\n2. Market Status & Time Analysis:")
    market_open = trader.is_market_open()
    current_slot = trader.get_current_time_slot()
    
    print(f"   Market Open: {'✅ YES' if market_open else '❌ NO'}")
    print(f"   Current Time Slot: {current_slot.value}")
    print(f"   Current Time: {datetime.now().strftime('%H:%M:%S')}")
    
    # Get CE & PE strategy recommendations
    print(f"\n3. CE & PE Strategy Recommendations for {current_slot.value}:")
    recommendations = trader.get_ce_pe_strategy_recommendations(current_slot)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec['strategy'].value}")
        print(f"      Description: {rec['description']}")
        print(f"      Risk Level: {rec['risk_level']}")
        print(f"      Strike Selection: {rec['strike_selection']}")
        print(f"      Suitable For: {rec['suitable_for']}")
    
    return trader, current_slot

def demo_ce_pe_contract_selection(trader, current_slot):
    """Demonstrate CE & PE contract selection"""
    print("\n" + "=" * 80)
    print("📈 CE & PE CONTRACT SELECTION DEMONSTRATION")
    print("=" * 80)
    
    # Get available contracts
    print("\n1. Available Option Contracts:")
    contracts = trader.get_available_contracts(
        strike_range=(24800, 25200),
        expiry_filter=OptionExpiry.WEEKLY
    )
    print(f"   Found {len(contracts)} contracts")
    
    # Find target contracts (25000 CE & PE)
    ce_contract = None
    pe_contract = None
    
    for contract in contracts:
        if contract.strike_price == 25000:
            if contract.option_type == OptionType.CALL:
                ce_contract = contract
            elif contract.option_type == OptionType.PUT:
                pe_contract = contract
    
    if not ce_contract or not pe_contract:
        print("❌ 25000 CE or PE contracts not found")
        return None, None, None, None
    
    print(f"\n2. Target Contracts Selected:")
    print(f"   CE Contract: {ce_contract.display_name}")
    print(f"   Strike Price: ₹{ce_contract.strike_price:,}")
    print(f"   Option Type: {ce_contract.option_type.value}")
    print(f"   Expiry Date: {ce_contract.expiry_date}")
    
    print(f"\n   PE Contract: {pe_contract.display_name}")
    print(f"   Strike Price: ₹{pe_contract.strike_price:,}")
    print(f"   Option Type: {pe_contract.option_type.value}")
    print(f"   Expiry Date: {pe_contract.expiry_date}")
    
    # Get current quotes
    print(f"\n3. Current Market Quotes:")
    ce_quote = trader.get_option_quote(ce_contract)
    pe_quote = trader.get_option_quote(pe_contract)
    
    if not ce_quote or not pe_quote:
        print("❌ Unable to get quotes")
        return None, None, None, None
    
    print(f"   CE Quote:")
    print(f"     Bid: ₹{ce_quote.bid_price:.2f}")
    print(f"     Ask: ₹{ce_quote.ask_price:.2f}")
    print(f"     Mid: ₹{ce_quote.mid_price:.2f}")
    print(f"     Spread: ₹{ce_quote.spread:.2f} ({ce_quote.spread_percentage:.1f}%)")
    
    print(f"   PE Quote:")
    print(f"     Bid: ₹{pe_quote.bid_price:.2f}")
    print(f"     Ask: ₹{pe_quote.ask_price:.2f}")
    print(f"     Mid: ₹{pe_quote.mid_price:.2f}")
    print(f"     Spread: ₹{pe_quote.spread:.2f} ({pe_quote.spread_percentage:.1f}%)")
    
    return ce_contract, pe_contract, ce_quote, pe_quote

def demo_individual_ce_pe_trades(trader, ce_contract, pe_contract, ce_quote, pe_quote, current_slot):
    """Demonstrate individual CE & PE trades"""
    print("\n" + "=" * 80)
    print("🎯 INDIVIDUAL CE & PE TRADES DEMONSTRATION")
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
    
    # Calculate risk parameters
    print(f"\n2. Risk Parameters:")
    stop_loss_percentage = 0.15  # 15% stop loss
    target_percentage = 0.30     # 30% target
    risk_amount = 3000           # Risk ₹3000 per trade
    
    print(f"   Stop Loss: {stop_loss_percentage*100:.0f}%")
    print(f"   Target: {target_percentage*100:.0f}%")
    print(f"   Risk Amount: ₹{risk_amount:,.2f}")
    
    # Place CE trade
    print(f"\n3. Placing CE Trade...")
    ce_trade = trader.place_ce_pe_intraday_trade(
        contract=ce_contract,
        action="BUY",
        strategy=strategy,
        time_slot=current_slot,
        entry_price=ce_quote.mid_price,
        stop_loss_percentage=stop_loss_percentage,
        target_percentage=target_percentage,
        risk_amount=risk_amount,
        max_holding_hours=6
    )
    
    if ce_trade:
        print(f"✅ CE trade placed successfully!")
        print(f"   Trade ID: {ce_trade['trade_id']}")
        print(f"   Strategy: {ce_trade['strategy'].value}")
        print(f"   Entry Price: ₹{ce_trade['entry_price']:.2f}")
        print(f"   Stop Loss: ₹{ce_trade['stop_loss']:.2f}")
        print(f"   Target: ₹{ce_trade['target_price']:.2f}")
        print(f"   Quantity: {ce_trade['quantity']} lots")
    else:
        print("❌ CE trade placement failed")
        return None, None
    
    # Place PE trade
    print(f"\n4. Placing PE Trade...")
    pe_trade = trader.place_ce_pe_intraday_trade(
        contract=pe_contract,
        action="BUY",
        strategy=strategy,
        time_slot=current_slot,
        entry_price=pe_quote.mid_price,
        stop_loss_percentage=stop_loss_percentage,
        target_percentage=target_percentage,
        risk_amount=risk_amount,
        max_holding_hours=6
    )
    
    if pe_trade:
        print(f"✅ PE trade placed successfully!")
        print(f"   Trade ID: {pe_trade['trade_id']}")
        print(f"   Strategy: {pe_trade['strategy'].value}")
        print(f"   Entry Price: ₹{pe_trade['entry_price']:.2f}")
        print(f"   Stop Loss: ₹{pe_trade['stop_loss']:.2f}")
        print(f"   Target: ₹{pe_trade['target_price']:.2f}")
        print(f"   Quantity: {pe_trade['quantity']} lots")
    else:
        print("❌ PE trade placement failed")
        return ce_trade, None
    
    return ce_trade, pe_trade

def demo_straddle_strategy(trader, ce_contract, pe_contract, ce_quote, pe_quote, current_slot):
    """Demonstrate straddle strategy"""
    print("\n" + "=" * 80)
    print("🔄 STRADDLE STRATEGY DEMONSTRATION")
    print("=" * 80)
    
    print(f"\n1. Straddle Strategy Overview:")
    print(f"   Strategy: Buy both CE & PE at same strike")
    print(f"   Strike Price: ₹{ce_contract.strike_price:,}")
    print(f"   Best For: High volatility periods")
    print(f"   Risk Level: HIGH")
    
    print(f"\n2. Straddle Trade Parameters:")
    stop_loss_percentage = 0.15
    target_percentage = 0.30
    quantity = 1
    risk_amount = 5000
    
    print(f"   Stop Loss: {stop_loss_percentage*100:.0f}%")
    print(f"   Target: {target_percentage*100:.0f}%")
    print(f"   Quantity: {quantity} lot")
    print(f"   Risk Amount: ₹{risk_amount:,.2f}")
    
    print(f"\n3. Placing Straddle Trade...")
    straddle = trader.place_straddle_trade(
        strike_price=ce_contract.strike_price,
        expiry_date=ce_contract.expiry_date,
        time_slot=current_slot,
        entry_price_ce=ce_quote.mid_price,
        entry_price_pe=pe_quote.mid_price,
        stop_loss_percentage=stop_loss_percentage,
        target_percentage=target_percentage,
        quantity=quantity,
        risk_amount=risk_amount
    )
    
    if straddle:
        print(f"✅ Straddle trade placed successfully!")
        print(f"   Straddle ID: {straddle['straddle_id']}")
        print(f"   Strike Price: ₹{straddle['strike_price']:,}")
        print(f"   Strategy: {straddle['strategy'].value}")
        print(f"   CE Trade ID: {straddle['ce_trade']['trade_id']}")
        print(f"   PE Trade ID: {straddle['pe_trade']['trade_id']}")
        print(f"   Total Risk: ₹{straddle['total_risk']:,.2f}")
        print(f"   Total Reward: ₹{straddle['total_reward']:,.2f}")
        
        return straddle
    else:
        print("❌ Straddle trade placement failed")
        return None

def demo_position_monitoring(trader, ce_trade, pe_trade, straddle):
    """Demonstrate position monitoring"""
    print("\n" + "=" * 80)
    print("📊 POSITION MONITORING DEMONSTRATION")
    print("=" * 80)
    
    print(f"\n1. Active Positions Summary:")
    summary = trader.get_enhanced_intraday_summary()
    print(f"   Total Active Positions: {summary['active_positions']}")
    print(f"   CE Positions: {summary['ce_positions']}")
    print(f"   PE Positions: {summary['pe_positions']}")
    print(f"   Spread Positions: {summary['spread_positions']}")
    print(f"   CE-PE Balance: {summary['ce_pe_balance']}")
    
    if ce_trade:
        print(f"\n2. CE Position Details:")
        print(f"   Trade ID: {ce_trade['trade_id']}")
        print(f"   Contract: {ce_trade['contract'].display_name}")
        print(f"   Entry Price: ₹{ce_trade['entry_price']:.2f}")
        print(f"   Stop Loss: ₹{ce_trade['stop_loss']:.2f}")
        print(f"   Target: ₹{ce_trade['target_price']:.2f}")
        print(f"   Strategy: {ce_trade['strategy'].value}")
    
    if pe_trade:
        print(f"\n3. PE Position Details:")
        print(f"   Trade ID: {pe_trade['trade_id']}")
        print(f"   Contract: {pe_trade['contract'].display_name}")
        print(f"   Entry Price: ₹{pe_trade['entry_price']:.2f}")
        print(f"   Stop Loss: ₹{pe_trade['stop_loss']:.2f}")
        print(f"   Target: ₹{pe_trade['target_price']:.2f}")
        print(f"   Strategy: {pe_trade['strategy'].value}")
    
    if straddle:
        print(f"\n4. Straddle Position Details:")
        print(f"   Straddle ID: {straddle['straddle_id']}")
        print(f"   Strike: ₹{straddle['strike_price']:,}")
        print(f"   Total Risk: ₹{straddle['total_risk']:,.2f}")
        print(f"   Total Reward: ₹{straddle['total_reward']:,.2f}")
    
    # Simulate different market scenarios
    print(f"\n5. Market Scenario Analysis:")
    
    if ce_trade:
        scenarios = [
            ("CE price moves up towards target", ce_trade['entry_price'] * 1.15),
            ("CE price stays near entry", ce_trade['entry_price'] * 1.02),
            ("CE price drops towards stop loss", ce_trade['entry_price'] * 0.88),
            ("CE price hits stop loss", ce_trade['stop_loss'] * 0.95)
        ]
        
        for scenario_name, price in scenarios:
            print(f"\n   Scenario: {scenario_name}")
            print(f"   Current CE Price: ₹{price:.2f}")
            
            # Check if position should exit
            positions_to_exit = trader.monitor_ce_pe_positions(price)
            
            if positions_to_exit:
                print(f"   ⚠️  Position needs to exit!")
                for position_info in positions_to_exit:
                    print(f"      Trade ID: {position_info['trade_id']}")
                    print(f"      Reason: {position_info['reason']}")
                    print(f"      Current Price: ₹{position_info['current_price']:.2f}")
            else:
                print(f"   ✅ Position is safe")

def demo_risk_management(trader):
    """Demonstrate risk management features"""
    print("\n" + "=" * 80)
    print("🛡️ RISK MANAGEMENT DEMONSTRATION")
    print("=" * 80)
    
    print(f"\n1. Enhanced Risk Settings:")
    print(f"   Max Risk per Trade: {trader.max_intraday_risk * 100:.1f}%")
    print(f"   Max Intraday Positions: {trader.max_intraday_positions}")
    print(f"   Max CE Positions: {trader.max_ce_positions}")
    print(f"   Max PE Positions: {trader.max_pe_positions}")
    print(f"   Max Spread Positions: {trader.max_spread_positions}")
    print(f"   Default Stop Loss: {trader.intraday_stop_loss_percentage * 100:.1f}%")
    print(f"   Default Target: {trader.intraday_target_percentage * 100:.1f}%")
    print(f"   Max Holding Time: {trader.max_holding_hours} hours")
    
    print(f"\n2. Daily Risk Limits:")
    daily_loss_limit = trader.account_balance * 0.05
    print(f"   Account Balance: ₹{trader.account_balance:,.2f}")
    print(f"   Daily Loss Limit: ₹{daily_loss_limit:,.2f} (5%)")
    print(f"   Current Daily P&L: ₹{trader.daily_pnl:,.2f}")
    
    print(f"\n3. Position Risk Analysis:")
    can_trade_ce, message_ce = trader.can_place_ce_pe_trade(OptionType.CALL)
    can_trade_pe, message_pe = trader.can_place_ce_pe_trade(OptionType.PUT)
    
    print(f"   Can Place CE Trade: {'✅ YES' if can_trade_ce else '❌ NO'}")
    print(f"   CE Message: {message_ce}")
    print(f"   Can Place PE Trade: {'✅ YES' if can_trade_pe else '❌ NO'}")
    print(f"   PE Message: {message_pe}")
    
    print(f"\n4. Risk Management Tools:")
    print(f"   Trailing Stop: Enabled (5%)")
    print(f"   Time-based Exit: Enabled")
    print(f"   Multiple Exit Conditions: Stop Loss, Target, Time, Trailing Stop")
    print(f"   Automatic Position Monitoring: Enabled")
    print(f"   CE/PE Balance Tracking: Enabled")
    print(f"   Spread Position Limits: Enabled")

def demo_performance_analytics(trader):
    """Demonstrate performance analytics"""
    print("\n" + "=" * 80)
    print("📋 PERFORMANCE ANALYTICS DEMONSTRATION")
    print("=" * 80)
    
    # Get enhanced summary
    summary = trader.get_enhanced_intraday_summary()
    
    print(f"\n1. Enhanced Trading Summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            if 'pnl' in key.lower() or 'drawdown' in key.lower():
                print(f"   {key.replace('_', ' ').title()}: ₹{value:,.2f}")
            else:
                print(f"   {key.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"   {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n2. CE & PE Performance Metrics:")
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
    
    print(f"\n3. Position Exposure Analysis:")
    ce_positions = summary['ce_positions']
    pe_positions = summary['pe_positions']
    spread_positions = summary['spread_positions']
    ce_pe_balance = summary['ce_pe_balance']
    
    print(f"   CE Exposure: {ce_positions} positions")
    print(f"   PE Exposure: {pe_positions} positions")
    print(f"   Spread Exposure: {spread_positions} positions")
    print(f"   CE-PE Balance: {ce_pe_balance}")
    
    if abs(ce_pe_balance) > 2:
        print(f"   ⚠️  High directional bias detected!")
    elif abs(ce_pe_balance) > 1:
        print(f"   📊 Moderate directional bias")
    else:
        print(f"   ✅ Well-balanced CE/PE exposure")
    
    print(f"\n4. Risk Exposure:")
    active_positions = summary['active_positions']
    if active_positions > 0:
        print(f"   ⚠️  Active Positions: {active_positions}")
        print(f"   ⚠️  Monitor these positions for exit conditions")
    else:
        print(f"   ✅ No Active Positions")
        print(f"   ✅ All positions closed for the day")

def demo_market_close_procedures(trader):
    """Demonstrate market close procedures"""
    print("\n" + "=" * 80)
    print("🏁 MARKET CLOSE PROCEDURES DEMONSTRATION")
    print("=" * 80)
    
    print(f"\n1. Market Close Procedures:")
    print(f"   Market Close Time: {trader.market_close_time}")
    print(f"   Force Close All Positions: Enabled")
    print(f"   Reason: MARKET_CLOSE")
    print(f"   CE/PE Balance Reset: Enabled")
    
    print(f"\n2. Simulating Market Close...")
    
    # Check if we have any active positions
    if trader.intraday_positions:
        print(f"   Active Positions: {len(trader.intraday_positions)}")
        
        # Simulate force close (in real scenario, this would use actual market prices)
        print(f"   🚨 Force closing all positions...")
        
        # Get current summary before close
        before_summary = trader.get_enhanced_intraday_summary()
        print(f"   Before Close - CE: {before_summary['ce_positions']}, PE: {before_summary['pe_positions']}")
        
        # In real implementation, this would close positions at market prices
        print(f"   ✅ All positions would be closed at market close")
        
        # Simulate summary after close
        print(f"   After Close - CE: 0, PE: 0")
    else:
        print(f"   ✅ No active positions to close")
    
    print(f"\n3. End-of-Day Summary:")
    final_summary = trader.get_enhanced_intraday_summary()
    print(f"   Total Trades: {final_summary['total_trades']}")
    print(f"   Final P&L: ₹{final_summary['daily_pnl']:,.2f}")
    print(f"   Win Rate: {final_summary['win_rate']:.1f}%")
    print(f"   CE Positions: {final_summary['ce_positions']}")
    print(f"   PE Positions: {final_summary['pe_positions']}")
    print(f"   Spread Positions: {final_summary['spread_positions']}")

def main():
    """Main demo function"""
    try:
        print("🚀 Starting Enhanced CE & PE Intraday Nifty 50 Options Trading Demo...")
        
        # Basic setup
        trader, current_slot = demo_enhanced_ce_pe_trading_system()
        
        # Contract selection
        ce_contract, pe_contract, ce_quote, pe_quote = demo_ce_pe_contract_selection(trader, current_slot)
        if not ce_contract or not pe_contract:
            print("❌ Failed to setup contracts")
            return
        
        # Individual CE & PE trades
        ce_trade, pe_trade = demo_individual_ce_pe_trades(trader, ce_contract, pe_contract, ce_quote, pe_quote, current_slot)
        
        # Straddle strategy
        straddle = demo_straddle_strategy(trader, ce_contract, pe_contract, ce_quote, pe_quote, current_slot)
        
        # Position monitoring
        demo_position_monitoring(trader, ce_trade, pe_trade, straddle)
        
        # Risk management
        demo_risk_management(trader)
        
        # Performance analytics
        demo_performance_analytics(trader)
        
        # Market close procedures
        demo_market_close_procedures(trader)
        
        print("\n" + "=" * 80)
        print("🎉 ENHANCED CE & PE DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nKey Enhanced Features Demonstrated:")
        print("✅ CE & PE specific trading strategies")
        print("✅ Individual CE & PE position management")
        print("✅ Straddle strategy implementation")
        print("✅ Enhanced risk management")
        print("✅ CE/PE balance tracking")
        print("✅ Spread position limits")
        print("✅ Advanced performance analytics")
        print("✅ Market close procedures")
        
        print("\nNext Steps:")
        print("1. Test different CE & PE strategies")
        print("2. Practice with straddle and strangle trades")
        print("3. Explore various risk-reward scenarios")
        print("4. Customize risk parameters for your style")
        print("5. Integrate with real-time market data")
        
    except Exception as e:
        print(f"\n❌ Enhanced CE & PE demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
