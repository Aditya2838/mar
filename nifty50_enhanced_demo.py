"""
Enhanced Nifty 50 Options Trading Demo
Demonstrates entry price, stop loss, and exit functionality
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

# Import our enhanced options trading system
from nifty50_options_trading import (
    Nifty50OptionsTrader, 
    OptionContract, 
    OptionType, 
    OptionExpiry,
    OptionQuote,
    ExitStrategy
)

def demo_enhanced_trading_features():
    """Demonstrate enhanced trading features"""
    print("=" * 70)
    print("🚀 ENHANCED NIFTY 50 OPTIONS TRADING DEMO")
    print("📊 Entry Price | Stop Loss | Exit Strategies")
    print("=" * 70)
    
    # Initialize trader
    print("\n1. Initializing Enhanced Options Trader...")
    trader = Nifty50OptionsTrader("enhanced_options.db")
    print("✅ Enhanced trader initialized successfully")
    
    # Display account summary
    print("\n2. Initial Account Summary:")
    summary = trader.get_account_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    # Get available contracts
    print("\n3. Available Option Contracts:")
    contracts = trader.get_available_contracts(
        strike_range=(24800, 25200),
        expiry_filter=OptionExpiry.WEEKLY
    )
    print(f"   Found {len(contracts)} contracts")
    
    return trader, contracts

def demo_trade_setup_with_risk_management(trader, contracts):
    """Demonstrate trade setup with risk management"""
    print("\n" + "=" * 70)
    print("📈 TRADE SETUP WITH RISK MANAGEMENT DEMO")
    print("=" * 70)
    
    # Find 25000 CE contract
    target_contract = None
    for contract in contracts:
        if (contract.strike_price == 25000 and 
            contract.option_type == OptionType.CALL):
            target_contract = contract
            break
    
    if not target_contract:
        print("❌ 25000 CE contract not found")
        return None
    
    print(f"\n1. Target Contract: {target_contract.display_name}")
    print(f"   Strike Price: ₹{target_contract.strike_price:,}")
    print(f"   Option Type: {target_contract.option_type.value}")
    print(f"   Expiry Date: {target_contract.expiry_date}")
    
    # Get quote
    print("\n2. Getting Option Quote...")
    quote = trader.get_option_quote(target_contract)
    if not quote:
        print("❌ Unable to get quote")
        return None
    
    print(f"   Bid Price: ₹{quote.bid_price:.2f}")
    print(f"   Ask Price: ₹{quote.ask_price:.2f}")
    print(f"   Mid Price: ₹{quote.mid_price:.2f}")
    
    # Calculate position size based on risk
    print("\n3. Risk Management Calculations:")
    risk_amount = 2000  # Risk ₹2000
    stop_loss = quote.mid_price * 0.8  # 20% below entry
    target_price = quote.mid_price * 1.5  # 50% above entry
    
    position_size = trader.calculate_position_size(
        entry_price=quote.mid_price,
        stop_loss=stop_loss,
        risk_amount=risk_amount
    )
    
    print(f"   Risk Amount: ₹{risk_amount:,.2f}")
    print(f"   Entry Price: ₹{quote.mid_price:.2f}")
    print(f"   Stop Loss: ₹{stop_loss:.2f}")
    print(f"   Target Price: ₹{target_price:.2f}")
    print(f"   Calculated Position Size: {position_size} lot(s)")
    
    # Calculate risk-reward metrics
    risk_per_lot = abs(quote.mid_price - stop_loss) * 50
    reward_per_lot = abs(target_price - quote.mid_price) * 50
    risk_reward_ratio = reward_per_lot / risk_per_lot if risk_per_lot > 0 else 0
    
    print(f"   Risk per Lot: ₹{risk_per_lot:.2f}")
    print(f"   Reward per Lot: ₹{reward_per_lot:.2f}")
    print(f"   Risk-Reward Ratio: {risk_reward_ratio:.2f}")
    
    return target_contract, quote, stop_loss, target_price, position_size

def demo_place_trade_with_setup(trader, contract, quote, stop_loss, target_price, position_size):
    """Demonstrate placing trade with setup"""
    print("\n" + "=" * 70)
    print("🎯 PLACING TRADE WITH SETUP DEMO")
    print("=" * 70)
    
    print(f"\n1. Placing Enhanced Order...")
    print(f"   Contract: {contract.display_name}")
    print(f"   Action: BUY")
    print(f"   Entry Price: ₹{quote.mid_price:.2f}")
    print(f"   Stop Loss: ₹{stop_loss:.2f}")
    print(f"   Target Price: ₹{target_price:.2f}")
    print(f"   Quantity: {position_size} lot(s)")
    
    # Place order with setup
    trade = trader.place_option_order_with_setup(
        contract=contract,
        action="BUY",
        entry_price=quote.mid_price,
        stop_loss=stop_loss,
        target_price=target_price,
        quantity=position_size,
        risk_amount=2000
    )
    
    if trade:
        print(f"\n✅ Trade executed successfully with setup!")
        print(f"   Trade ID: {trade.trade_id}")
        print(f"   Status: {trade.status}")
        print(f"   Timestamp: {trade.timestamp}")
        
        # Display trade setup details
        if trade.trade_setup:
            setup = trade.trade_setup
            print(f"\n2. Trade Setup Details:")
            print(f"   Entry Price: ₹{setup.entry_price:.2f}")
            print(f"   Stop Loss: ₹{setup.stop_loss:.2f}")
            print(f"   Target Price: ₹{setup.target_price:.2f}")
            print(f"   Quantity: {setup.quantity} lot(s)")
            print(f"   Risk-Reward Ratio: {setup.risk_reward_ratio:.2f}")
            print(f"   Max Loss: ₹{setup.max_loss:,.2f}")
            print(f"   Max Profit: ₹{setup.max_profit:,.2f}")
        
        # Get trade setup summary
        print(f"\n3. Active Trade Setups:")
        setup_summary = trader.get_trade_setup_summary()
        for setup_info in setup_summary:
            print(f"   Trade ID: {setup_info['trade_id']}")
            print(f"   Contract: {setup_info['contract']}")
            print(f"   Entry: ₹{setup_info['entry_price']:.2f} | SL: ₹{setup_info['stop_loss']:.2f} | Target: ₹{setup_info['target_price']:.2f}")
            print(f"   Risk-Reward: {setup_info['risk_reward_ratio']:.2f}")
        
        return trade
    else:
        print("❌ Trade execution failed")
        return None

def demo_stop_loss_and_target_monitoring(trader, trade, contract):
    """Demonstrate stop loss and target monitoring"""
    print("\n" + "=" * 70)
    print("📊 STOP LOSS & TARGET MONITORING DEMO")
    print("=" * 70)
    
    if not trade or not trade.trade_setup:
        print("❌ No trade setup to monitor")
        return
    
    setup = trade.trade_setup
    print(f"\n1. Monitoring Trade: {trade.trade_id}")
    print(f"   Entry Price: ₹{setup.entry_price:.2f}")
    print(f"   Stop Loss: ₹{setup.stop_loss:.2f}")
    print(f"   Target Price: ₹{setup.target_price:.2f}")
    
    # Simulate different market scenarios
    scenarios = [
        ("Price drops to stop loss", setup.stop_loss * 0.95),
        ("Price moves to target", setup.target_price * 1.05),
        ("Price stays neutral", setup.entry_price * 1.02)
    ]
    
    for scenario_name, price in scenarios:
        print(f"\n2. Scenario: {scenario_name}")
        print(f"   Current Price: ₹{price:.2f}")
        
        # Check if position should exit
        positions_to_exit = trader.check_stop_loss_and_target(price)
        
        if positions_to_exit:
            print(f"   ⚠️  Position needs to exit!")
            for position in positions_to_exit:
                print(f"      Trade ID: {position['trade_id']}")
                print(f"      Reason: {position['reason']}")
                print(f"      Current Price: ₹{position['current_price']:.2f}")
        else:
            print(f"   ✅ Position is safe")
    
    # Simulate actual exit scenario
    print(f"\n3. Simulating Stop Loss Exit...")
    exit_price = setup.stop_loss * 0.95  # Price hits stop loss
    
    print(f"   Price hits: ₹{exit_price:.2f}")
    print(f"   Stop Loss: ₹{setup.stop_loss:.2f}")
    
    # Auto-exit positions
    exited_trades = trader.auto_exit_positions(exit_price)
    
    if exited_trades:
        print(f"   🚨 Auto-exit triggered!")
        for exit_trade in exited_trades:
            print(f"      Exit Trade ID: {exit_trade.trade_id}")
            print(f"      Exit Price: ₹{exit_trade.price:.2f}")
            print(f"      Exit Strategy: {exit_trade.exit_strategy.value if exit_trade.exit_strategy else 'STOP_LOSS'}")
    else:
        print(f"   ✅ No positions exited")

def demo_position_management_features(trader):
    """Demonstrate position management features"""
    print("\n" + "=" * 70)
    print("📋 POSITION MANAGEMENT FEATURES DEMO")
    print("=" * 70)
    
    # Get current positions
    print("\n1. Current Positions:")
    positions = trader.get_current_positions()
    if positions:
        for position_id, position in positions.items():
            contract = position['contract']
            print(f"   Position ID: {position_id}")
            print(f"   Contract: {contract.display_name}")
            print(f"   Quantity: {position['quantity']} lot(s)")
            print(f"   Average Price: ₹{position['average_price']:.2f}")
    else:
        print("   No open positions")
    
    # Get active trade setups
    print(f"\n2. Active Trade Setups:")
    setup_summary = trader.get_trade_setup_summary()
    if setup_summary:
        for setup_info in setup_summary:
            print(f"   Trade ID: {setup_info['trade_id']}")
            print(f"   Contract: {setup_info['contract']}")
            print(f"   Entry: ₹{setup_info['entry_price']:.2f}")
            print(f"   SL: ₹{setup_info['stop_loss']:.2f}")
            print(f"   Target: ₹{setup_info['target_price']:.2f}")
            print(f"   Risk-Reward: {setup_info['risk_reward_ratio']:.2f}")
    else:
        print("   No active trade setups")
    
    # Get trade history
    print(f"\n3. Trade History:")
    trades = trader.get_trade_history()
    if trades:
        for trade in trades[-3:]:  # Show last 3 trades
            print(f"   {trade.timestamp.strftime('%H:%M:%S')} - {trade.action} {trade.quantity} lot(s) of {trade.contract.display_name}")
            if trade.exit_strategy:
                print(f"      Exit Strategy: {trade.exit_strategy.value}")
    else:
        print("   No trade history")

def demo_risk_management_tools(trader):
    """Demonstrate risk management tools"""
    print("\n" + "=" * 70)
    print("🛡️ RISK MANAGEMENT TOOLS DEMO")
    print("=" * 70)
    
    print(f"\n1. Risk Management Settings:")
    print(f"   Max Risk per Trade: {trader.max_risk_per_trade * 100:.1f}%")
    print(f"   Max Portfolio Risk: {trader.max_portfolio_risk * 100:.1f}%")
    print(f"   Trailing Stop: {trader.trailing_stop_percentage * 100:.1f}%")
    
    print(f"\n2. Account Risk Analysis:")
    summary = trader.get_account_summary()
    account_balance = summary['account_balance']
    total_value = summary['total_value']
    
    print(f"   Account Balance: ₹{account_balance:,.2f}")
    print(f"   Total Portfolio Value: ₹{total_value:,.2f}")
    print(f"   Max Risk per Trade: ₹{account_balance * trader.max_risk_per_trade:,.2f}")
    print(f"   Max Portfolio Risk: ₹{total_value * trader.max_portfolio_risk:,.2f}")
    
    print(f"\n3. Position Sizing Calculator:")
    # Example calculations
    entry_price = 50.0
    stop_loss = 40.0
    risk_amount = 2000
    
    position_size = trader.calculate_position_size(entry_price, stop_loss, risk_amount)
    risk_per_lot = abs(entry_price - stop_loss) * 50
    
    print(f"   Example: Entry ₹{entry_price}, SL ₹{stop_loss}, Risk ₹{risk_amount}")
    print(f"   Risk per Lot: ₹{risk_per_lot:.2f}")
    print(f"   Calculated Position Size: {position_size} lot(s)")
    print(f"   Actual Risk: ₹{position_size * risk_per_lot:.2f}")

def main():
    """Main demo function"""
    try:
        print("🚀 Starting Enhanced Nifty 50 Options Trading Demo...")
        
        # Basic setup
        trader, contracts = demo_enhanced_trading_features()
        
        # Trade setup with risk management
        result = demo_trade_setup_with_risk_management(trader, contracts)
        if not result:
            print("❌ Failed to setup trade")
            return
        
        contract, quote, stop_loss, target_price, position_size = result
        
        # Place trade with setup
        trade = demo_place_trade_with_setup(trader, contract, quote, stop_loss, target_price, position_size)
        
        if trade:
            # Monitor stop loss and target
            demo_stop_loss_and_target_monitoring(trader, trade, contract)
            
            # Position management
            demo_position_management_features(trader)
            
            # Risk management tools
            demo_risk_management_tools(trader)
        
        print("\n" + "=" * 70)
        print("🎉 ENHANCED DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nKey Enhanced Features Demonstrated:")
        print("✅ Entry price, stop loss, and target setup")
        print("✅ Risk-based position sizing")
        print("✅ Automatic stop loss and target monitoring")
        print("✅ Auto-exit functionality")
        print("✅ Risk management tools")
        print("✅ Trade setup tracking")
        print("✅ Enhanced database schema")
        
        print("\nNext Steps:")
        print("1. Run the Streamlit dashboard: streamlit run nifty50_options_dashboard.py")
        print("2. Test different risk-reward scenarios")
        print("3. Explore trailing stop features")
        print("4. Integrate with real market data")
        
    except Exception as e:
        print(f"\n❌ Enhanced demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
