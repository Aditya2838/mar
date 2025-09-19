"""
Nifty 50 Options Trading Demo
Demonstrates the key features of the options trading system
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

# Import our options trading system
from nifty50_options_trading import (
    Nifty50OptionsTrader, 
    OptionContract, 
    OptionType, 
    OptionExpiry,
    OptionQuote
)

def demo_basic_options_trading():
    """Demonstrate basic options trading functionality"""
    print("=" * 60)
    print("🚀 NIFTY 50 OPTIONS TRADING DEMO")
    print("=" * 60)
    
    # Initialize trader
    print("\n1. Initializing Options Trader...")
    trader = Nifty50OptionsTrader("demo_options.db")
    print("✅ Trader initialized successfully")
    
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
    
    # Display some sample contracts
    print("\n   Sample contracts:")
    for i, contract in enumerate(contracts[:5]):
        print(f"   {i+1}. {contract.display_name}")
    
    return trader, contracts

def demo_buy_call_option(trader, contracts):
    """Demonstrate buying a call option"""
    print("\n" + "=" * 60)
    print("📈 BUYING CALL OPTION DEMO")
    print("=" * 60)
    
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
    print(f"   Lot Size: {target_contract.lot_size}")
    
    # Get quote
    print("\n2. Getting Option Quote...")
    quote = trader.get_option_quote(target_contract)
    if quote:
        print(f"   Bid Price: ₹{quote.bid_price:.2f}")
        print(f"   Ask Price: ₹{quote.ask_price:.2f}")
        print(f"   Mid Price: ₹{quote.mid_price:.2f}")
        print(f"   Spread: ₹{quote.spread:.2f} ({quote.spread_percentage:.2f}%)")
        print(f"   Implied Volatility: {quote.implied_volatility:.2%}")
        print(f"   Delta: {quote.delta:.3f}")
        print(f"   Gamma: {quote.gamma:.3f}")
        print(f"   Theta: {quote.theta:.3f}")
        print(f"   Vega: {quote.vega:.3f}")
    else:
        print("❌ Unable to get quote")
        return None
    
    # Place buy order
    print("\n3. Placing Buy Order...")
    quantity = 1  # 1 lot
    trade = trader.place_option_order(
        contract=target_contract,
        action="BUY",
        quantity=quantity
    )
    
    if trade:
        print(f"✅ Trade executed successfully!")
        print(f"   Trade ID: {trade.trade_id}")
        print(f"   Action: {trade.action}")
        print(f"   Quantity: {trade.quantity} lot(s)")
        print(f"   Price: ₹{trade.price:.2f}")
        print(f"   Total Value: ₹{trade.total_value:,.2f}")
        print(f"   Status: {trade.status}")
        print(f"   Timestamp: {trade.timestamp}")
        
        # Update account summary
        print("\n4. Updated Account Summary:")
        updated_summary = trader.get_account_summary()
        for key, value in updated_summary.items():
            print(f"   {key}: {value}")
        
        return trade
    else:
        print("❌ Trade execution failed")
        return None

def demo_position_management(trader, trade):
    """Demonstrate position management"""
    print("\n" + "=" * 60)
    print("📋 POSITION MANAGEMENT DEMO")
    print("=" * 60)
    
    if not trade:
        print("❌ No trade to manage")
        return
    
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
            print(f"   Open Date: {position['open_timestamp']}")
            
            # Get current quote for P&L calculation
            current_quote = trader.get_option_quote(contract)
            if current_quote:
                current_value = position['quantity'] * current_quote.mid_price * contract.lot_size
                cost_basis = position['quantity'] * position['average_price'] * contract.lot_size
                pnl = current_value - cost_basis
                pnl_percentage = (pnl / cost_basis) * 100 if cost_basis > 0 else 0
                
                print(f"   Current Price: ₹{current_quote.mid_price:.2f}")
                print(f"   Market Value: ₹{current_value:,.2f}")
                print(f"   P&L: ₹{pnl:,.2f} ({pnl_percentage:+.2f}%)")
    else:
        print("   No open positions")
    
    # Simulate price movement
    print("\n2. Simulating Price Movement...")
    original_level = trader.nifty50_current_level
    
    # Increase Nifty level (good for call options)
    new_level = original_level + 200
    trader.nifty50_current_level = new_level
    print(f"   Nifty 50 moved from ₹{original_level:,} to ₹{new_level:,}")
    
    # Get updated quote
    updated_quote = trader.get_option_quote(trade.contract)
    if updated_quote:
        print(f"   Updated Option Price: ₹{updated_quote.mid_price:.2f}")
        
        # Calculate new P&L
        if positions:
            position = list(positions.values())[0]
            new_value = position['quantity'] * updated_quote.mid_price * trade.contract.lot_size
            cost_basis = position['quantity'] * position['average_price'] * trade.contract.lot_size
            new_pnl = new_value - cost_basis
            new_pnl_percentage = (new_pnl / cost_basis) * 100 if cost_basis > 0 else 0
            
            print(f"   New Market Value: ₹{new_value:,.2f}")
            print(f"   New P&L: ₹{new_pnl:,.2f} ({new_pnl_percentage:+.2f}%)")
    
    # Restore original level
    trader.nifty50_current_level = original_level

def demo_close_position(trader, trade):
    """Demonstrate closing a position"""
    print("\n" + "=" * 60)
    print("🔒 CLOSING POSITION DEMO")
    print("=" * 60)
    
    if not trade:
        print("❌ No trade to close")
        return
    
    # Get current positions
    positions = trader.get_current_positions()
    if not positions:
        print("❌ No open positions to close")
        return
    
    # Close position
    print("\n1. Closing Position...")
    position_id = list(positions.keys())[0]
    
    if trader.close_position(position_id):
        print("✅ Position closed successfully!")
        
        # Updated account summary
        print("\n2. Final Account Summary:")
        final_summary = trader.get_account_summary()
        for key, value in final_summary.items():
            print(f"   {key}: {value}")
        
        # Trade history
        print("\n3. Trade History:")
        trades = trader.get_trade_history()
        for trade in trades:
            print(f"   {trade.timestamp.strftime('%Y-%m-%d %H:%M')} - "
                  f"{trade.action} {trade.quantity} lot(s) of {trade.contract.display_name} "
                  f"@ ₹{trade.price:.2f} - {trade.status}")
    else:
        print("❌ Failed to close position")

def demo_payoff_analysis(trader):
    """Demonstrate payoff analysis"""
    print("\n" + "=" * 60)
    print("📊 PAYOFF ANALYSIS DEMO")
    print("=" * 60)
    
    # Get some contracts for analysis
    contracts = trader.get_available_contracts(
        strike_range=(24800, 25200),
        expiry_filter=OptionExpiry.WEEKLY
    )
    
    if len(contracts) < 2:
        print("❌ Need at least 2 contracts for strategy analysis")
        return
    
    # Create a simple straddle strategy (buy both call and put)
    call_contract = None
    put_contract = None
    
    for contract in contracts:
        if contract.strike_price == 25000:
            if contract.option_type == OptionType.CALL:
                call_contract = contract
            elif contract.option_type == OptionType.PUT:
                put_contract = contract
    
    if not call_contract or not put_contract:
        print("❌ Could not find 25000 CE and PE contracts")
        return
    
    print(f"\n1. Strategy: Long Straddle")
    print(f"   Buy 1 lot of {call_contract.display_name}")
    print(f"   Buy 1 lot of {put_contract.display_name}")
    
    # Calculate payoff
    print("\n2. Calculating Payoff...")
    payoff_data = trader.calculate_payoff(
        contracts=[call_contract, put_contract],
        quantities=[1, 1],  # Long 1 lot each
        spot_range=(24000, 26000),
        spot_step=200
    )
    
    if not payoff_data.empty:
        print("✅ Payoff calculation completed")
        
        # Find breakeven points
        breakeven_points = payoff_data[payoff_data['breakeven']]
        if not breakeven_points.empty:
            breakeven_prices = breakeven_points['spot_price'].tolist()
            print(f"   Breakeven Points: {', '.join([f'₹{price:,.0f}' for price in breakeven_prices])}")
        
        # Calculate max profit and loss
        max_profit = payoff_data['payoff'].max()
        max_loss = payoff_data['payoff'].min()
        
        print(f"   Maximum Profit: ₹{max_profit:,.2f}")
        print(f"   Maximum Loss: ₹{max_loss:,.2f}")
        
        # Display payoff data
        print("\n3. Payoff Data (Sample):")
        sample_data = payoff_data[::5]  # Every 5th row
        for _, row in sample_data.iterrows():
            print(f"   Spot: ₹{row['spot_price']:,.0f} | Payoff: ₹{row['payoff']:,.2f}")
        
        return payoff_data
    else:
        print("❌ Payoff calculation failed")
        return None

def demo_options_chain(trader):
    """Demonstrate options chain functionality"""
    print("\n" + "=" * 60)
    print("📋 OPTIONS CHAIN DEMO")
    print("=" * 60)
    
    # Get options chain for current week
    today = date.today()
    weekly_expiry = None
    
    for expiry in trader.expiry_dates:
        if expiry <= today + timedelta(days=7):
            weekly_expiry = expiry
            break
    
    if not weekly_expiry:
        print("❌ No weekly expiry found")
        return
    
    print(f"\n1. Options Chain for {weekly_expiry.strftime('%d %B %Y')}")
    
    # Get options chain
    options_chain = trader.get_options_chain(expiry_date=weekly_expiry)
    
    if not options_chain:
        print("❌ No options data available")
        return
    
    print(f"   Found {len(options_chain)} strike prices")
    
    # Display options chain
    print("\n2. Options Chain Data:")
    for strike, quotes in sorted(options_chain.items(), key=lambda x: float(x[0])):
        strike_float = float(strike)
        print(f"\n   Strike: ₹{strike_float:,}")
        
        for quote in quotes:
            option_type = quote.contract.option_type.value
            print(f"     {option_type}: Bid ₹{quote.bid_price:.2f} | Ask ₹{quote.ask_price:.2f} | "
                  f"IV {quote.implied_volatility:.2%} | Delta {quote.delta:.3f}")

def main():
    """Main demo function"""
    try:
        print("🚀 Starting Nifty 50 Options Trading Demo...")
        
        # Basic setup
        trader, contracts = demo_basic_options_trading()
        
        # Buy call option
        trade = demo_buy_call_option(trader, contracts)
        
        # Position management
        demo_position_management(trader, trade)
        
        # Payoff analysis
        demo_payoff_analysis(trader)
        
        # Options chain
        demo_options_chain(trader)
        
        # Close position
        demo_close_position(trader, trade)
        
        print("\n" + "=" * 60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nKey Features Demonstrated:")
        print("✅ Options contract creation and management")
        print("✅ Real-time option pricing and quotes")
        print("✅ Buy/sell order execution")
        print("✅ Position tracking and P&L calculation")
        print("✅ Payoff analysis and strategy building")
        print("✅ Options chain visualization")
        print("✅ Position closing and trade history")
        
        print("\nNext Steps:")
        print("1. Run the Streamlit dashboard: streamlit run nifty50_options_dashboard.py")
        print("2. Explore different option strategies")
        print("3. Test with different market scenarios")
        print("4. Integrate with real broker APIs for live trading")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
