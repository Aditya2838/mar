"""
Nifty 50 Options Trading Dashboard
Streamlit web application for trading Nifty 50 index options
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
import json
import sqlite3
from typing import Dict, List, Optional

# Import our options trading system
from nifty50_options_trading import (
    Nifty50OptionsTrader, 
    OptionContract, 
    OptionType, 
    OptionExpiry,
    OptionQuote
)

# Page configuration
st.set_page_config(
    page_title="Nifty 50 Options Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .option-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
    }
    .profit-positive { color: #28a745; font-weight: bold; }
    .profit-negative { color: #dc3545; font-weight: bold; }
    .stButton > button {
        width: 100%;
        margin: 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

class OptionsDashboard:
    """Main dashboard class for Nifty 50 options trading"""
    
    def __init__(self):
        """Initialize the dashboard"""
        self.trader = None
        self.initialize_trader()
        
    def initialize_trader(self):
        """Initialize the options trader"""
        try:
            self.trader = Nifty50OptionsTrader("nifty50_options.db")
            st.session_state.trader = self.trader
        except Exception as e:
            st.error(f"Error initializing trader: {e}")
            self.trader = None
    
    def render_header(self):
        """Render the main header"""
        st.markdown('<h1 class="main-header">📈 Nifty 50 Options Trading Dashboard</h1>', unsafe_allow_html=True)
        
        # Current Nifty 50 level
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            current_nifty = st.number_input(
                "Current Nifty 50 Level",
                min_value=20000.0,
                max_value=30000.0,
                value=25000.0,
                step=50.0,
                format="%.0f",
                help="Update current Nifty 50 level for accurate option pricing"
            )
            
            if st.button("Update Nifty Level", type="primary"):
                if self.trader:
                    self.trader.nifty50_current_level = current_nifty
                    self.trader.available_strikes = self.trader._generate_strike_prices()
                    st.success(f"Nifty 50 level updated to {current_nifty}")
                    st.rerun()
    
    def render_account_summary(self):
        """Render account summary section"""
        st.markdown("## 💰 Account Summary")
        
        if not self.trader:
            st.warning("Trader not initialized")
            return
        
        summary = self.trader.get_account_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Account Balance",
                f"₹{summary['account_balance']:,.2f}",
                help="Available cash balance"
            )
        
        with col2:
            st.metric(
                "Total Positions",
                summary['total_positions'],
                help="Number of open option positions"
            )
        
        with col3:
            pnl_color = "normal"
            if summary['unrealized_pnl'] > 0:
                pnl_color = "inverse"
            elif summary['unrealized_pnl'] < 0:
                pnl_color = "off"
            
            st.metric(
                "Unrealized P&L",
                f"₹{summary['unrealized_pnl']:,.2f}",
                delta=f"{summary['unrealized_pnl']:+.2f}",
                delta_color=pnl_color,
                help="Current unrealized profit/loss"
            )
        
        with col4:
            st.metric(
                "Total Value",
                f"₹{summary['total_value']:,.2f}",
                help="Total portfolio value including positions"
            )
    
    def render_options_chain(self):
        """Render options chain section"""
        st.markdown("## 📊 Options Chain")
        
        if not self.trader:
            st.warning("Trader not initialized")
            return
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            expiry_filter = st.selectbox(
                "Expiry Filter",
                ["All", "Weekly", "Monthly", "Quarterly"],
                help="Filter options by expiry period"
            )
        
        with col2:
            strike_range = st.slider(
                "Strike Range",
                min_value=24000,
                max_value=26000,
                value=(24800, 25200),
                step=50,
                help="Select strike price range"
            )
        
        with col3:
            option_type_filter = st.selectbox(
                "Option Type",
                ["All", "Call (CE)", "Put (PE)"],
                help="Filter by option type"
            )
        
        # Get available contracts
        expiry_map = {
            "Weekly": OptionExpiry.WEEKLY,
            "Monthly": OptionExpiry.MONTHLY,
            "Quarterly": OptionExpiry.QUARTERLY
        }
        
        expiry_filter_enum = expiry_map.get(expiry_filter) if expiry_filter != "All" else None
        
        contracts = self.trader.get_available_contracts(
            strike_range=strike_range,
            expiry_filter=expiry_filter_enum
        )
        
        # Filter by option type
        if option_type_filter != "All":
            option_type = OptionType.CALL if option_type_filter == "Call (CE)" else OptionType.PUT
            contracts = [c for c in contracts if c.option_type == option_type]
        
        if not contracts:
            st.info("No contracts found with the selected filters")
            return
        
        # Display options chain
        st.markdown(f"**Found {len(contracts)} contracts**")
        
        # Group contracts by strike price
        strikes = sorted(list(set([c.strike_price for c in contracts])))
        
        for strike in strikes:
            strike_contracts = [c for c in contracts if c.strike_price == strike]
            
            st.markdown(f"### Strike: {strike}")
            
            for contract in strike_contracts:
                self.render_option_contract(contract)
    
    def render_option_contract(self, contract: OptionContract):
        """Render individual option contract"""
        quote = self.trader.get_option_quote(contract)
        
        if not quote:
            return
        
        # Create option card
        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 2])
        
        with col1:
            st.markdown(f"**{contract.display_name}**")
            st.markdown(f"*{contract.option_type.value}*")
        
        with col2:
            st.markdown(f"**Bid:** ₹{quote.bid_price:.2f}")
            st.markdown(f"**Ask:** ₹{quote.ask_price:.2f}")
        
        with col3:
            st.markdown(f"**Spread:** ₹{quote.spread:.2f}")
            st.markdown(f"**IV:** {quote.implied_volatility:.2%}")
        
        with col4:
            st.markdown(f"**Delta:** {quote.delta:.3f}")
            st.markdown(f"**Gamma:** {quote.gamma:.3f}")
        
        with col5:
            # Trading buttons
            if contract.option_type == OptionType.CALL:
                if st.button(f"Buy {contract.strike_price} CE", key=f"buy_{contract.contract_id}"):
                    self.execute_trade(contract, "BUY")
            else:
                if st.button(f"Buy {contract.strike_price} PE", key=f"buy_{contract.contract_id}"):
                    self.execute_trade(contract, "BUY")
            
            if st.button(f"Sell {contract.strike_price} {contract.option_type.value}", key=f"sell_{contract.contract_id}"):
                self.execute_trade(contract, "SELL")
    
    def execute_trade(self, contract: OptionContract, action: str):
        """Execute a trade"""
        if not self.trader:
            st.error("Trader not initialized")
            return
        
        try:
            # Get quantity from user
            quantity = st.number_input(
                f"Quantity (lots) for {action} {contract.display_name}",
                min_value=1,
                max_value=100,
                value=1,
                step=1,
                key=f"qty_{contract.contract_id}_{action}"
            )
            
            if st.button(f"Confirm {action}", key=f"confirm_{contract.contract_id}_{action}"):
                trade = self.trader.place_option_order(
                    contract=contract,
                    action=action,
                    quantity=quantity
                )
                
                if trade:
                    st.success(f"Trade executed successfully! Trade ID: {trade.trade_id}")
                    st.rerun()
                else:
                    st.error("Trade execution failed")
        
        except Exception as e:
            st.error(f"Error executing trade: {e}")
    
    def render_positions(self):
        """Render current positions section"""
        st.markdown("## 📋 Current Positions")
        
        if not self.trader:
            st.warning("Trader not initialized")
            return
        
        positions = self.trader.get_current_positions()
        
        if not positions:
            st.info("No open positions")
            return
        
        # Display positions
        for position_id, position in positions.items():
            contract = position['contract']
            current_quote = self.trader.get_option_quote(contract)
            
            if not current_quote:
                continue
            
            # Calculate P&L
            current_value = position['quantity'] * current_quote.mid_price * contract.lot_size
            cost_basis = position['quantity'] * position['average_price'] * contract.lot_size
            pnl = current_value - cost_basis
            pnl_percentage = (pnl / cost_basis) * 100 if cost_basis > 0 else 0
            
            # Position card
            col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
            
            with col1:
                st.markdown(f"**{contract.display_name}**")
                st.markdown(f"*{position['quantity']} lots @ ₹{position['average_price']:.2f}*")
            
            with col2:
                st.markdown(f"**Current Price:** ₹{current_quote.mid_price:.2f}")
                st.markdown(f"**Market Value:** ₹{current_value:,.2f}")
            
            with col3:
                pnl_color = "profit-positive" if pnl >= 0 else "profit-negative"
                st.markdown(f"**P&L:** <span class='{pnl_color}'>₹{pnl:,.2f}</span>", unsafe_allow_html=True)
                st.markdown(f"**P&L %:** <span class='{pnl_color}'>{pnl_percentage:+.2f}%</span>", unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"**Open Date:** {position['open_timestamp'].strftime('%Y-%m-%d')}")
                st.markdown(f"**Days Held:** {(datetime.now() - position['open_timestamp']).days}")
            
            with col5:
                if st.button(f"Close Position", key=f"close_{position_id}"):
                    if self.trader.close_position(position_id):
                        st.success("Position closed successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to close position")
    
    def render_trade_history(self):
        """Render trade history section"""
        st.markdown("## 📈 Trade History")
        
        if not self.trader:
            st.warning("Trader not initialized")
            return
        
        trades = self.trader.get_trade_history()
        
        if not trades:
            st.info("No trade history")
            return
        
        # Convert trades to DataFrame for display
        trade_data = []
        for trade in trades:
            trade_data.append({
                'Date': trade.timestamp.strftime('%Y-%m-%d %H:%M'),
                'Contract': trade.contract.display_name,
                'Action': trade.action,
                'Quantity': trade.quantity,
                'Price': f"₹{trade.price:.2f}",
                'Total Value': f"₹{trade.total_value:,.2f}",
                'Status': trade.status
            })
        
        df = pd.DataFrame(trade_data)
        st.dataframe(df, use_container_width=True)
    
    def render_payoff_analyzer(self):
        """Render payoff analyzer section"""
        st.markdown("## 📊 Payoff Analyzer")
        
        if not self.trader:
            st.warning("Trader not initialized")
            return
        
        # Strategy builder
        st.markdown("### Build Option Strategy")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Select contracts
            available_contracts = self.trader.get_available_contracts(
                strike_range=(24000, 26000),
                expiry_filter=OptionExpiry.WEEKLY
            )
            
            if not available_contracts:
                st.info("No contracts available for analysis")
                return
            
            # Contract selection
            selected_contracts = []
            quantities = []
            
            for i in range(3):  # Allow up to 3 legs
                st.markdown(f"**Leg {i+1}:**")
                
                contract_idx = st.selectbox(
                    "Contract",
                    range(len(available_contracts)),
                    format_func=lambda x: available_contracts[x].display_name,
                    key=f"contract_{i}"
                )
                
                quantity = st.number_input(
                    "Quantity",
                    min_value=-10,
                    max_value=10,
                    value=1 if i == 0 else 0,
                    step=1,
                    key=f"qty_{i}"
                )
                
                if contract_idx < len(available_contracts) and quantity != 0:
                    selected_contracts.append(available_contracts[contract_idx])
                    quantities.append(quantity)
        
        with col2:
            # Strategy parameters
            spot_range = st.slider(
                "Spot Price Range",
                min_value=24000,
                max_value=26000,
                value=(24000, 26000),
                step=100
            )
            
            if st.button("Calculate Payoff", type="primary"):
                if selected_contracts and quantities:
                    self.calculate_and_display_payoff(selected_contracts, quantities, spot_range)
                else:
                    st.warning("Please select at least one contract with non-zero quantity")
    
    def calculate_and_display_payoff(self, contracts: List[OptionContract], quantities: List[int], spot_range: tuple):
        """Calculate and display payoff diagram"""
        try:
            payoff_data = self.trader.calculate_payoff(
                contracts=contracts,
                quantities=quantities,
                spot_range=spot_range,
                spot_step=100
            )
            
            if payoff_data.empty:
                st.error("Error calculating payoff")
                return
            
            # Create payoff chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=payoff_data['spot_price'],
                y=payoff_data['payoff'],
                mode='lines+markers',
                name='Payoff',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ))
            
            # Add breakeven line
            breakeven_points = payoff_data[payoff_data['breakeven']]
            if not breakeven_points.empty:
                fig.add_trace(go.Scatter(
                    x=breakeven_points['spot_price'],
                    y=breakeven_points['payoff'],
                    mode='markers',
                    name='Breakeven',
                    marker=dict(color='red', size=10, symbol='diamond')
                ))
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            
            # Update layout
            fig.update_layout(
                title="Option Strategy Payoff Diagram",
                xaxis_title="Nifty 50 Spot Price",
                yaxis_title="Payoff (₹)",
                hovermode='x unified',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display strategy summary
            st.markdown("### Strategy Summary")
            
            strategy_summary = []
            for contract, quantity in zip(contracts, quantities):
                action = "Long" if quantity > 0 else "Short"
                strategy_summary.append(f"{action} {abs(quantity)} lot(s) of {contract.display_name}")
            
            st.markdown("**Strategy:** " + " + ".join(strategy_summary))
            
            # Find breakeven points
            if not breakeven_points.empty:
                breakeven_prices = breakeven_points['spot_price'].tolist()
                st.markdown(f"**Breakeven Points:** {', '.join([f'₹{price:,.0f}' for price in breakeven_prices])}")
            
            # Calculate max profit and loss
            max_profit = payoff_data['payoff'].max()
            max_loss = payoff_data['payoff'].min()
            
            st.markdown(f"**Maximum Profit:** ₹{max_profit:,.2f}")
            st.markdown(f"**Maximum Loss:** ₹{max_loss:,.2f}")
            
        except Exception as e:
            st.error(f"Error calculating payoff: {e}")
    
    def render_market_data(self):
        """Render market data section"""
        st.markdown("## 📊 Market Data")
        
        if not self.trader:
            st.warning("Trader not initialized")
            return
        
        # Get options chain for current week
        today = date.today()
        weekly_expiry = None
        
        for expiry in self.trader.expiry_dates:
            if expiry <= today + timedelta(days=7):
                weekly_expiry = expiry
                break
        
        if not weekly_expiry:
            st.info("No weekly expiry found")
            return
        
        st.markdown(f"### Options Chain - {weekly_expiry.strftime('%d %B %Y')}")
        
        # Get options chain
        options_chain = self.trader.get_options_chain(expiry_date=weekly_expiry)
        
        if not options_chain:
            st.info("No options data available")
            return
        
        # Create options chain table
        chain_data = []
        for strike, quotes in options_chain.items():
            strike_float = float(strike)
            
            # Find call and put quotes
            call_quote = None
            put_quote = None
            
            for quote in quotes:
                if quote.contract.option_type == OptionType.CALL:
                    call_quote = quote
                else:
                    put_quote = quote
            
            row = {
                'Strike': strike_float,
                'Call Bid': call_quote.bid_price if call_quote else '-',
                'Call Ask': call_quote.ask_price if call_quote else '-',
                'Call IV': f"{call_quote.implied_volatility:.2%}" if call_quote else '-',
                'Put Bid': put_quote.bid_price if put_quote else '-',
                'Put Ask': put_quote.ask_price if put_quote else '-',
                'Put IV': f"{put_quote.implied_volatility:.2%}" if put_quote else '-'
            }
            chain_data.append(row)
        
        df = pd.DataFrame(chain_data)
        df = df.sort_values('Strike')
        
        st.dataframe(df, use_container_width=True)
    
    def run(self):
        """Run the main dashboard"""
        try:
            # Initialize trader if not exists
            if 'trader' not in st.session_state:
                self.initialize_trader()
            
            # Render dashboard
            self.render_header()
            
            # Sidebar navigation
            st.sidebar.title("Navigation")
            page = st.sidebar.selectbox(
                "Select Page",
                ["Dashboard", "Options Chain", "Positions", "Trade History", "Payoff Analyzer", "Market Data"]
            )
            
            # Render selected page
            if page == "Dashboard":
                self.render_account_summary()
                self.render_positions()
                self.render_trade_history()
            
            elif page == "Options Chain":
                self.render_options_chain()
            
            elif page == "Positions":
                self.render_positions()
            
            elif page == "Trade History":
                self.render_trade_history()
            
            elif page == "Payoff Analyzer":
                self.render_payoff_analyzer()
            
            elif page == "Market Data":
                self.render_market_data()
            
        except Exception as e:
            st.error(f"Dashboard error: {e}")
            st.exception(e)

# Run the dashboard
if __name__ == "__main__":
    dashboard = OptionsDashboard()
    dashboard.run()
