"""
Comprehensive Indian Stocks Intraday Trading Dashboard
Streamlit application for ALL Indian stocks with detailed entry/exit points and trading signals
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import logging

# Import our custom modules
from analysis.intraday_trading import IntradayTradingAnalysis
from data.nifty50_data import Nifty50Data
from config import Config, INDIAN_STOCKS, ALL_INDIAN_STOCKS, POPULAR_INTRADAY_STOCKS, INTRADAY_TRADING_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Indian Stocks Intraday Trading Dashboard",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'intraday_analysis' not in st.session_state:
    st.session_state.intraday_analysis = IntradayTradingAnalysis()
if 'nifty_data' not in st.session_state:
    st.session_state.nifty_data = Nifty50Data()

def display_stock_selector():
    """Display stock selection interface"""
    st.sidebar.header("📊 Stock Selection")
    
    # Category selection
    category = st.sidebar.selectbox(
        "Select Category",
        options=["Popular Intraday", "Nifty 50", "Nifty Next 50", "Midcap 100", "Smallcap 100", 
                "Bank Nifty", "IT Sector", "Pharma Sector", "Auto Sector", "FMCG Sector", 
                "Metal Sector", "Energy Sector", "All Stocks"],
        index=0
    )
    
    # Get stocks based on category
    if category == "Popular Intraday":
        stocks = POPULAR_INTRADAY_STOCKS
    elif category == "Nifty 50":
        stocks = Config.NIFTY_50_SYMBOLS
    elif category == "Nifty Next 50":
        stocks = INDIAN_STOCKS["NIFTY_NEXT_50"]
    elif category == "Midcap 100":
        stocks = INDIAN_STOCKS["NIFTY_MIDCAP_100"]
    elif category == "Smallcap 100":
        stocks = INDIAN_STOCKS["NIFTY_SMALLCAP_100"]
    elif category == "Bank Nifty":
        stocks = INDIAN_STOCKS["BANK_NIFTY"]
    elif category == "IT Sector":
        stocks = INDIAN_STOCKS["IT_SECTOR"]
    elif category == "Pharma Sector":
        stocks = INDIAN_STOCKS["PHARMA_SECTOR"]
    elif category == "Auto Sector":
        stocks = INDIAN_STOCKS["AUTO_SECTOR"]
    elif category == "FMCG Sector":
        stocks = INDIAN_STOCKS["FMCG_SECTOR"]
    elif category == "Metal Sector":
        stocks = INDIAN_STOCKS["METAL_SECTOR"]
    elif category == "Energy Sector":
        stocks = INDIAN_STOCKS["ENERGY_SECTOR"]
    else:  # All Stocks
        stocks = ALL_INDIAN_STOCKS[:100]  # Limit to first 100 for performance
    
    # Stock selection
    selected_stock = st.sidebar.selectbox(
        "Select Stock",
        options=stocks,
        index=0
    )
    
    return selected_stock, category

def display_trading_controls():
    """Display trading control parameters"""
    st.sidebar.header("🎛️ Trading Controls")
    
    # Time interval selection
    interval = st.sidebar.selectbox(
        "Data Interval",
        options=["1m", "5m", "15m", "30m", "1h"],
        index=2  # Default to 15m
    )
    
    # Analysis period
    period = st.sidebar.selectbox(
        "Analysis Period",
        options=["1d", "5d"],
        index=1  # Default to 5d
    )
    
    # Risk management
    st.sidebar.subheader("💰 Risk Management")
    stop_loss_pct = st.sidebar.slider(
        "Stop Loss (%)",
        min_value=0.5,
        max_value=5.0,
        value=INTRADAY_TRADING_CONFIG["stop_loss_percentage"],
        step=0.1
    )
    
    target_pct = st.sidebar.slider(
        "Target (%)",
        min_value=1.0,
        max_value=10.0,
        value=3.0,
        step=0.5
    )
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()
    
    return interval, period, stop_loss_pct, target_pct, auto_refresh

def display_trading_signal(signal_data, stock_symbol, stop_loss_pct, target_pct):
    """Display comprehensive trading signal with entry/exit points"""
    
    signal = signal_data['signal']
    confidence = signal_data['confidence']
    
    # Create signal display
    if signal == 'BUY':
        st.success(f"🟢 **BUY SIGNAL** - {stock_symbol}")
        st.metric(
            "Action Required",
            "BUY NOW",
            delta=f"Confidence: {confidence}%",
            delta_color="normal"
        )
        st.info("💰 **Recommended Action:** Consider buying this stock for intraday gains")
        
    elif signal == 'SELL':
        st.error(f"🔴 **SELL SIGNAL** - {stock_symbol}")
        st.metric(
            "Action Required",
            "SELL NOW",
            delta=f"Confidence: {confidence}%",
            delta_color="inverse"
        )
        st.info("💸 **Recommended Action:** Consider selling this stock to avoid losses")
        
    else:  # HOLD
        st.warning(f"🟡 **HOLD SIGNAL** - {stock_symbol}")
        st.metric(
            "Action Required",
            "HOLD/WAIT",
            delta=f"Confidence: {confidence}%",
            delta_color="off"
        )
        st.info("⏳ **Recommended Action:** Wait for better entry/exit points")
    
    # Display signal details
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Buy Strength", f"{signal_data['buy_strength']:.1%}")
    
    with col2:
        st.metric("Sell Strength", f"{signal_data['sell_strength']:.1%}")
    
    with col3:
        st.metric("Signal Quality", f"{confidence}%")
    
    # Display reasons
    st.subheader("📋 Signal Reasons")
    reasons = signal_data['reason'].split('; ')
    for i, reason in enumerate(reasons, 1):
        st.write(f"{i}. {reason}")
    
    return signal

def display_entry_exit_points(analysis, stock_symbol, stop_loss_pct, target_pct):
    """Display detailed entry and exit points for trading"""
    
    st.subheader("🎯 Entry & Exit Points")
    
    current_price = analysis['current_price']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Price", f"₹{current_price:.2f}")
        st.metric("VWAP", f"₹{analysis.get('vwap', current_price):.2f}")
        st.metric("Volume", f"{analysis.get('volume', 0):,.0f}")
    
    with col2:
        if analysis['signals']['signal'] == 'BUY':
            # Calculate entry points for BUY
            entry_1 = current_price * (1 - stop_loss_pct/200)  # Half stop loss below
            entry_2 = current_price * (1 - stop_loss_pct/100)  # Full stop loss below
            stop_loss = current_price * (1 - stop_loss_pct/100)
            
            st.metric("Entry Price 1", f"₹{entry_1:.2f}")
            st.metric("Entry Price 2", f"₹{entry_2:.2f}")
            st.metric("Stop Loss", f"₹{stop_loss:.2f}")
            
        elif analysis['signals']['signal'] == 'SELL':
            # Calculate entry points for SELL
            entry_1 = current_price * (1 + stop_loss_pct/200)  # Half stop loss above
            entry_2 = current_price * (1 + stop_loss_pct/100)  # Full stop loss above
            stop_loss = current_price * (1 + stop_loss_pct/100)
            
            st.metric("Entry Price 1", f"₹{entry_1:.2f}")
            st.metric("Entry Price 2", f"₹{entry_2:.2f}")
            st.metric("Stop Loss", f"₹{stop_loss:.2f}")
    
    with col3:
        if analysis['signals']['signal'] == 'BUY':
            # Calculate target prices for BUY
            target_1 = current_price * (1 + target_pct/100)
            target_2 = current_price * (1 + target_pct*1.5/100)
            target_3 = current_price * (1 + target_pct*2/100)
            
            st.metric("Target 1", f"₹{target_1:.2f}")
            st.metric("Target 2", f"₹{target_2:.2f}")
            st.metric("Target 3", f"₹{target_3:.2f}")
            
        elif analysis['signals']['signal'] == 'SELL':
            # Calculate target prices for SELL
            target_1 = current_price * (1 - target_pct/100)
            target_2 = current_price * (1 - target_pct*1.5/100)
            target_3 = current_price * (1 - target_pct*2/100)
            
            st.metric("Target 1", f"₹{target_1:.2f}")
            st.metric("Target 2", f"₹{target_2:.2f}")
            st.metric("Target 3", f"₹{target_3:.2f}")
    
    # Risk-Reward Analysis
    st.subheader("⚖️ Risk-Reward Analysis")
    
    if analysis['signals']['signal'] == 'BUY':
        risk = stop_loss_pct
        reward = target_pct
        rr_ratio = reward / risk
        
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.metric("Risk", f"{risk:.1f}%")
        with col_r2:
            st.metric("Reward", f"{reward:.1f}%")
        with col_r3:
            st.metric("R:R Ratio", f"{rr_ratio:.2f}")
            
        if rr_ratio >= 2:
            st.success("✅ **Excellent Risk-Reward Ratio** - Risk is justified")
        elif rr_ratio >= 1.5:
            st.info("⚠️ **Good Risk-Reward Ratio** - Consider position sizing")
        else:
            st.warning("❌ **Poor Risk-Reward Ratio** - Reconsider trade")

def display_market_timing(analysis):
    """Display market timing analysis"""
    
    st.subheader("⏰ Market Timing")
    timing = analysis['timing']
    
    col_t1, col_t2, col_t3 = st.columns(3)
    
    with col_t1:
        st.info(f"**Current Timing:** {timing['timing']}")
        st.write(f"**Reason:** {timing['reason']}")
    
    with col_t2:
        st.write(f"**Best Volume Hour:** {timing['best_volume_hour']}:00")
        st.write(f"**Best Momentum Hour:** {timing['best_momentum_hour']}:00")
    
    with col_t3:
        st.write(f"**Best Volatility Hour:** {timing['best_volatility_hour']}:00")
        st.write(f"**Current Hour:** {timing['current_hour']}:00")

def display_price_predictions(analysis, signal):
    """Display price predictions with targets"""
    
    st.subheader("🔮 Price Predictions (Next 6 intervals)")
    predictions_df = pd.DataFrame(analysis['predictions'])
    
    if not predictions_df.empty:
        # Create prediction chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=predictions_df['interval'],
            y=predictions_df['predicted_price'],
            mode='lines+markers',
            name='Predicted Price',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_hline(
            y=analysis['current_price'],
            line_dash="dash",
            line_color="red",
            annotation_text="Current Price"
        )
        
        # Add target lines for BUY/SELL
        if signal == 'BUY':
            fig.add_hline(
                y=analysis['current_price'] * 1.02,
                line_dash="dot",
                line_color="green",
                annotation_text="Target (+2%)"
            )
            fig.add_hline(
                y=analysis['current_price'] * 1.03,
                line_dash="dot",
                line_color="darkgreen",
                annotation_text="Target (+3%)"
            )
        elif signal == 'SELL':
            fig.add_hline(
                y=analysis['current_price'] * 0.98,
                line_dash="dot",
                line_color="red",
                annotation_text="Target (-2%)"
            )
            fig.add_hline(
                y=analysis['current_price'] * 0.97,
                line_dash="dot",
                line_color="darkred",
                annotation_text="Target (-3%)"
            )
        
        fig.update_layout(
            title="Intraday Price Predictions",
            xaxis_title="Time Intervals (15min)",
            yaxis_title="Price (₹)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display predictions table
        st.dataframe(
            predictions_df.style.format({
                'predicted_price': '₹{:.2f}',
                'predicted_change': '{:.2f}%'
            }),
            use_container_width=True
        )

def display_technical_charts(selected_stock, period, interval):
    """Display comprehensive technical analysis charts"""
    
    st.subheader("📊 Technical Analysis")
    
    # Get intraday data for charting
    intraday_data = st.session_state.intraday_analysis.get_intraday_data(
        selected_stock, period, interval
    )
    
    if not intraday_data.empty:
        # Calculate indicators
        intraday_data = st.session_state.intraday_analysis.calculate_intraday_indicators(intraday_data)
        
        # Create subplots for comprehensive analysis
        fig = make_subplots(
            rows=5, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price & VWAP', 'RSI', 'MACD', 'Bollinger Bands', 'Volume'),
            row_heights=[0.3, 0.2, 0.2, 0.2, 0.1]
        )
        
        # Price and VWAP
        fig.add_trace(
            go.Candlestick(
                x=intraday_data.index,
                open=intraday_data['Open'],
                high=intraday_data['High'],
                low=intraday_data['Low'],
                close=intraday_data['Close'],
                name='OHLC'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=intraday_data.index,
                y=intraday_data['VWAP'],
                mode='lines',
                name='VWAP',
                line=dict(color='purple', width=2)
            ),
            row=1, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(
                x=intraday_data.index,
                y=intraday_data['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='orange', width=2)
            ),
            row=2, col=1
        )
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        fig.add_trace(
            go.Scatter(
                x=intraday_data.index,
                y=intraday_data['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='blue', width=2)
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=intraday_data.index,
                y=intraday_data['MACD_Signal'],
                mode='lines',
                name='Signal',
                line=dict(color='red', width=2)
            ),
            row=3, col=1
        )
        
        # Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=intraday_data.index,
                y=intraday_data['BB_Upper'],
                mode='lines',
                name='BB Upper',
                line=dict(color='gray', width=1)
            ),
            row=4, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=intraday_data.index,
                y=intraday_data['BB_Middle'],
                mode='lines',
                name='BB Middle',
                line=dict(color='gray', width=1)
            ),
            row=4, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=intraday_data.index,
                y=intraday_data['BB_Lower'],
                mode='lines',
                name='BB Lower',
                line=dict(color='gray', width=1)
            ),
            row=4, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=intraday_data.index,
                y=intraday_data['Close'],
                mode='lines',
                name='Close',
                line=dict(color='black', width=2)
            ),
            row=4, col=1
        )
        
        # Volume
        fig.add_trace(
            go.Bar(
                x=intraday_data.index,
                y=intraday_data['Volume'],
                name='Volume',
                marker_color='lightblue'
            ),
            row=5, col=1
        )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_market_summary():
    """Display market summary and opportunities"""
    
    st.sidebar.header("📋 Market Summary")
    
    try:
        summary = st.session_state.intraday_analysis.get_nifty50_intraday_summary()
        
        if 'error' not in summary:
            # Summary metrics
            st.sidebar.metric("Total Stocks", summary['total_stocks'])
            st.sidebar.metric("Analyzed", summary['stocks_analyzed'])
            
            # Signal distribution
            col_sum1, col_sum2, col_sum3 = st.sidebar.columns(3)
            
            with col_sum1:
                st.metric("🟢 BUY", summary['buy_signals'])
            
            with col_sum2:
                st.metric("🔴 SELL", summary['sell_signals'])
            
            with col_sum3:
                st.metric("🟡 HOLD", summary['hold_signals'])
            
            # Top opportunities
            if summary['stock_details']:
                st.sidebar.subheader("🏆 Top Opportunities")
                
                # Sort by confidence
                top_stocks = sorted(
                    summary['stock_details'],
                    key=lambda x: x['confidence'],
                    reverse=True
                )[:5]
                
                for i, stock in enumerate(top_stocks, 1):
                    signal_color = {
                        'BUY': '🟢',
                        'SELL': '🔴',
                        'HOLD': '🟡'
                    }.get(stock['signal'], '⚪')
                    
                    st.sidebar.write(f"{i}. {signal_color} **{stock['symbol']}**")
                    st.sidebar.write(f"   {stock['signal']} ({stock['confidence']}%)")
                    st.sidebar.write(f"   ₹{stock['current_price']:.2f}")
                    
                    # Add action button
                    if stock['signal'] == 'BUY':
                        st.sidebar.button(f"🟢 BUY {stock['symbol']}", key=f"buy_{i}")
                    elif stock['signal'] == 'SELL':
                        st.sidebar.button(f"🔴 SELL {stock['symbol']}", key=f"sell_{i}")
                    else:
                        st.sidebar.button(f"🟡 WATCH {stock['symbol']}", key=f"watch_{i}")
                
                st.sidebar.write(f"*Last updated: {summary['timestamp']}*")
                
    except Exception as e:
        st.sidebar.error(f"Error in summary: {str(e)}")

def main():
    st.title("🇮🇳 Indian Stocks Intraday Trading Dashboard")
    st.markdown("### **Complete Entry & Exit Points for ALL Indian Stocks**")
    st.markdown("---")
    
    # Sidebar controls
    selected_stock, category = display_stock_selector()
    interval, period, stop_loss_pct, target_pct, auto_refresh = display_trading_controls()
    display_market_summary()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📊 {selected_stock} - {category}")
        
        # Get intraday data and analysis
        try:
            analysis = st.session_state.intraday_analysis.get_intraday_predictions(
                selected_stock, 
                prediction_horizon=6
            )
            
            if 'error' not in analysis:
                # Display current status
                current_col1, current_col2, current_col3, current_col4 = st.columns(4)
                
                with current_col1:
                    st.metric(
                        "Current Price",
                        f"₹{analysis['current_price']:.2f}",
                        delta=f"{analysis['current_momentum']:.2f}%"
                    )
                
                with current_col2:
                    st.metric(
                        "VWAP",
                        f"₹{analysis.get('vwap', 0):.2f}",
                        delta="Reference"
                    )
                
                with current_col3:
                    st.metric(
                        "Volume",
                        f"{analysis.get('volume', 0):,.0f}",
                        delta="Today"
                    )
                
                with current_col4:
                    st.metric(
                        "Volatility",
                        f"{analysis['current_volatility']:.4f}",
                        delta="ATR"
                    )
                
                st.markdown("---")
                
                # Display trading signal prominently
                signal = display_trading_signal(analysis['signals'], selected_stock, stop_loss_pct, target_pct)
                
                st.markdown("---")
                
                # Display entry/exit points
                display_entry_exit_points(analysis, selected_stock, stop_loss_pct, target_pct)
                
                st.markdown("---")
                
                # Market timing analysis
                display_market_timing(analysis)
                
                st.markdown("---")
                
                # Price predictions
                display_price_predictions(analysis, signal)
                
                st.markdown("---")
                
                # Technical indicators chart
                display_technical_charts(selected_stock, period, interval)
                
            else:
                st.error(f"Error analyzing {selected_stock}: {analysis['error']}")
                
        except Exception as e:
            st.error(f"Error in analysis: {str(e)}")
            logger.error(f"Dashboard error: {str(e)}")
    
    with col2:
        st.subheader("📋 Quick Actions")
        
        # Quick trade setup
        st.info("**Quick Trade Setup**")
        
        if st.button("📝 Set Buy Order", use_container_width=True):
            st.success("Buy order set for analysis")
        
        if st.button("📝 Set Sell Order", use_container_width=True):
            st.success("Sell order set for analysis")
        
        if st.button("📊 View Portfolio", use_container_width=True):
            st.info("Portfolio view coming soon...")
        
        # Market alerts
        st.subheader("🔔 Market Alerts")
        
        if st.checkbox("Price Alerts"):
            st.write("Price alerts enabled")
        
        if st.checkbox("Signal Alerts"):
            st.write("Signal alerts enabled")
        
        if st.checkbox("Volume Alerts"):
            st.write("Volume alerts enabled")
        
        # Trading tips
        st.subheader("💡 Trading Tips")
        st.info("""
        **Intraday Trading Tips:**
        1. Always use stop-loss
        2. Don't chase losses
        3. Book profits at targets
        4. Monitor volume patterns
        5. Follow market timing
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <p>🇮🇳 Indian Stocks Intraday Trading Dashboard | Complete Entry/Exit Points</p>
        <p>⚠️ This is for educational purposes only. Always do your own research before trading.</p>
        <p>📊 Real-time analysis with comprehensive risk management</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()



