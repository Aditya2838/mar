"""
Enhanced Intraday Trading Dashboard for Nifty 50
Streamlit application for real-time intraday analysis and clear BUY/SELL trading signals
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
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Nifty 50 Intraday Trading Signals",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'intraday_analysis' not in st.session_state:
    st.session_state.intraday_analysis = IntradayTradingAnalysis()
if 'nifty_data' not in st.session_state:
    st.session_state.nifty_data = Nifty50Data()

def display_trading_signal(signal_data, stock_symbol):
    """Display trading signal with clear visual indicators"""
    
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

def display_entry_exit_points(analysis, stock_symbol):
    """Display entry and exit points for trading"""
    
    st.subheader("🎯 Entry & Exit Points")
    
    current_price = analysis['current_price']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Price", f"₹{current_price:.2f}")
    
    with col2:
        if analysis['signals']['signal'] == 'BUY':
            # Calculate entry points for BUY
            entry_1 = current_price * 0.995  # 0.5% below current
            entry_2 = current_price * 0.99   # 1% below current
            stop_loss = current_price * 0.985  # 1.5% below current
            
            st.metric("Entry Price 1", f"₹{entry_1:.2f}")
            st.metric("Entry Price 2", f"₹{entry_2:.2f}")
            st.metric("Stop Loss", f"₹{stop_loss:.2f}")
            
        elif analysis['signals']['signal'] == 'SELL':
            # Calculate entry points for SELL
            entry_1 = current_price * 1.005  # 0.5% above current
            entry_2 = current_price * 1.01   # 1% above current
            stop_loss = current_price * 1.015  # 1.5% above current
            
            st.metric("Entry Price 1", f"₹{entry_1:.2f}")
            st.metric("Entry Price 2", f"₹{entry_2:.2f}")
            st.metric("Stop Loss", f"₹{stop_loss:.2f}")
    
    with col3:
        if analysis['signals']['signal'] == 'BUY':
            # Calculate target prices for BUY
            target_1 = current_price * 1.01   # 1% above current
            target_2 = current_price * 1.02   # 2% above current
            target_3 = current_price * 1.03   # 3% above current
            
            st.metric("Target 1", f"₹{target_1:.2f}")
            st.metric("Target 2", f"₹{target_2:.2f}")
            st.metric("Target 3", f"₹{target_3:.2f}")
            
        elif analysis['signals']['signal'] == 'SELL':
            # Calculate target prices for SELL
            target_1 = current_price * 0.99   # 1% below current
            target_2 = current_price * 0.98   # 2% below current
            target_3 = current_price * 0.97   # 3% below current
            
            st.metric("Target 1", f"₹{target_1:.2f}")
            st.metric("Target 2", f"₹{target_2:.2f}")
            st.metric("Target 3", f"₹{target_3:.2f}")

def main():
    st.title("🎯 Nifty 50 Intraday Trading Signals")
    st.markdown("### **Where to BUY and Where to SELL - Real-time Analysis**")
    st.markdown("---")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🎛️ Dashboard Controls")
        
        # Stock selection
        selected_stock = st.selectbox(
            "Select Stock",
            options=Config.NIFTY_50_SYMBOLS,
            index=0
        )
        
        # Time interval selection
        interval = st.selectbox(
            "Data Interval",
            options=["1m", "5m", "15m", "30m", "1h"],
            index=2  # Default to 15m
        )
        
        # Analysis period
        period = st.selectbox(
            "Analysis Period",
            options=["1d", "5d"],
            index=1  # Default to 5d
        )
        
        # Auto-refresh
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=True)
        
        # Manual refresh button
        if st.button("🔄 Refresh Now"):
            st.rerun()
        
        st.markdown("---")
        st.markdown("**📊 Quick Stats**")
        
        # Get quick summary
        try:
            summary = st.session_state.intraday_analysis.get_nifty50_intraday_summary()
            if 'error' not in summary:
                st.metric("BUY Signals", summary['buy_signals'])
                st.metric("SELL Signals", summary['sell_signals'])
                st.metric("HOLD Signals", summary['hold_signals'])
        except:
            pass
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📊 {selected_stock} - Trading Analysis")
        
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
                signal = display_trading_signal(analysis['signals'], selected_stock)
                
                st.markdown("---")
                
                # Display entry/exit points
                display_entry_exit_points(analysis, selected_stock)
                
                st.markdown("---")
                
                # Market timing analysis
                st.subheader("⏰ Market Timing")
                timing = analysis['timing']
                
                col_t1, col_t2 = st.columns(2)
                
                with col_t1:
                    st.info(f"**Current Timing:** {timing['timing']}")
                    st.write(f"**Reason:** {timing['reason']}")
                
                with col_t2:
                    st.write(f"**Best Volume Hour:** {timing['best_volume_hour']}:00")
                    st.write(f"**Best Momentum Hour:** {timing['best_momentum_hour']}:00")
                
                # Price predictions
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
                    elif signal == 'SELL':
                        fig.add_hline(
                            y=analysis['current_price'] * 0.98,
                            line_dash="dot",
                            line_color="red",
                            annotation_text="Target (-2%)"
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
                
                # Technical indicators chart
                st.subheader("📊 Technical Indicators")
                
                # Get intraday data for charting
                intraday_data = st.session_state.intraday_analysis.get_intraday_data(
                    selected_stock, period, interval
                )
                
                if not intraday_data.empty:
                    # Calculate indicators
                    intraday_data = st.session_state.intraday_analysis.calculate_intraday_indicators(intraday_data)
                    
                    # Create subplots for price and indicators
                    fig = make_subplots(
                        rows=4, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=('Price & VWAP', 'RSI', 'MACD', 'Volume'),
                        row_heights=[0.4, 0.2, 0.2, 0.2]
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
                    
                    # Volume
                    fig.add_trace(
                        go.Bar(
                            x=intraday_data.index,
                            y=intraday_data['Volume'],
                            name='Volume',
                            marker_color='lightblue'
                        ),
                        row=4, col=1
                    )
                    
                    fig.update_layout(
                        height=600,
                        showlegend=True,
                        xaxis_rangeslider_visible=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error(f"Error analyzing {selected_stock}: {analysis['error']}")
                
        except Exception as e:
            st.error(f"Error in analysis: {str(e)}")
            logger.error(f"Dashboard error: {str(e)}")
    
    with col2:
        st.subheader("📋 Nifty 50 Trading Summary")
        
        # Get summary of all stocks
        try:
            summary = st.session_state.intraday_analysis.get_nifty50_intraday_summary()
            
            if 'error' not in summary:
                # Summary metrics
                st.metric("Total Stocks", summary['total_stocks'])
                st.metric("Analyzed", summary['stocks_analyzed'])
                
                # Signal distribution
                col_sum1, col_sum2, col_sum3 = st.columns(3)
                
                with col_sum1:
                    st.metric("🟢 BUY", summary['buy_signals'], delta="Signals")
                
                with col_sum2:
                    st.metric("🔴 SELL", summary['sell_signals'], delta="Signals")
                
                with col_sum3:
                    st.metric("🟡 HOLD", summary['hold_signals'], delta="Signals")
                
                # Signal distribution chart
                if summary['buy_signals'] + summary['sell_signals'] + summary['hold_signals'] > 0:
                    signal_counts = {
                        'BUY': summary['buy_signals'],
                        'SELL': summary['sell_signals'],
                        'HOLD': summary['hold_signals']
                    }
                    
                    fig_pie = px.pie(
                        values=list(signal_counts.values()),
                        names=list(signal_counts.keys()),
                        title="Signal Distribution",
                        color=list(signal_counts.keys()),
                        color_discrete_map={
                            'BUY': 'green',
                            'SELL': 'red',
                            'HOLD': 'orange'
                        }
                    )
                    
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Top stocks by signal strength
                if summary['stock_details']:
                    st.subheader("🏆 Top Trading Opportunities")
                    
                    # Sort by confidence
                    top_stocks = sorted(
                        summary['stock_details'],
                        key=lambda x: x['confidence'],
                        reverse=True
                    )[:8]
                    
                    for i, stock in enumerate(top_stocks, 1):
                        signal_color = {
                            'BUY': '🟢',
                            'SELL': '🔴',
                            'HOLD': '🟡'
                        }.get(stock['signal'], '⚪')
                        
                        signal_text = {
                            'BUY': 'BUY',
                            'SELL': 'SELL',
                            'HOLD': 'HOLD'
                        }.get(stock['signal'], 'UNKNOWN')
                        
                        st.write(f"{i}. {signal_color} **{stock['symbol']}**")
                        st.write(f"   **{signal_text}** ({stock['confidence']}%)")
                        st.write(f"   ₹{stock['current_price']:.2f}")
                        
                        # Add action button
                        if stock['signal'] == 'BUY':
                            st.button(f"🟢 BUY {stock['symbol']}", key=f"buy_{i}", use_container_width=True)
                        elif stock['signal'] == 'SELL':
                            st.button(f"🔴 SELL {stock['symbol']}", key=f"sell_{i}", use_container_width=True)
                        else:
                            st.button(f"🟡 WATCH {stock['symbol']}", key=f"watch_{i}", use_container_width=True)
                
                st.write(f"*Last updated: {summary['timestamp']}*")
                
            else:
                st.error(f"Error getting summary: {summary['error']}")
                
        except Exception as e:
            st.error(f"Error in summary: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <p>🎯 Nifty 50 Intraday Trading Signals | Clear BUY/SELL Recommendations</p>
        <p>⚠️ This is for educational purposes only. Always do your own research before trading.</p>
        <p>📊 Real-time analysis with entry/exit points and stop-loss levels</p>
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
