"""
Nifty 50 Analysis Dashboard
Comprehensive Streamlit application for Indian Nifty 50 market analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data.nifty50_data import Nifty50Data
from analysis.nifty50_predictions import Nifty50Predictions
from analysis.technical_indicators import TechnicalAnalysis
from config import Config

# Page configuration
st.set_page_config(
    page_title="Nifty 50 Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
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
    .prediction-card {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff7f0e;
    }
    .sector-card {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2ca02c;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">🇮🇳 Nifty 50 Market Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Dashboard Controls")
        
        # Time period selection
        period = st.selectbox(
            "Select Time Period",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3
        )
        
        # Analysis type
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Overview", "Technical Analysis", "Sector Analysis", "Predictions", "Market Breadth"]
        )
        
        # Stock selection for detailed analysis
        if analysis_type == "Technical Analysis":
            selected_stock = st.selectbox(
                "Select Stock for Detailed Analysis",
                Config.NIFTY_50_SYMBOLS,
                index=0
            )
        
        # Prediction settings
        if analysis_type == "Predictions":
            prediction_days = st.slider("Days to Predict", 1, 10, 5)
            selected_model = st.selectbox(
                "Select Prediction Model",
                ["RandomForest", "XGBoost", "LightGBM", "Ensemble"],
                index=0
            )
    
    # Main content
    if analysis_type == "Overview":
        show_overview(period)
    elif analysis_type == "Technical Analysis":
        show_technical_analysis(period, selected_stock)
    elif analysis_type == "Sector Analysis":
        show_sector_analysis(period)
    elif analysis_type == "Predictions":
        show_predictions(prediction_days, selected_model)
    elif analysis_type == "Market Breadth":
        show_market_breadth(period)

def show_overview(period):
    """Show Nifty 50 overview"""
    st.header("📈 Nifty 50 Market Overview")
    
    try:
        # Initialize data
        nifty_data = Nifty50Data()
        
        # Get Nifty 50 index data
        with st.spinner("Fetching Nifty 50 data..."):
            index_data = nifty_data.get_nifty50_index_data(period=period)
        
        if index_data.empty:
            st.error("Failed to fetch Nifty 50 data")
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        current_price = index_data['Close'].iloc[-1]
        prev_price = index_data['Close'].iloc[-2]
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100
        
        with col1:
            st.metric(
                "Current Level",
                f"₹{current_price:,.2f}",
                f"{price_change:+.2f} ({price_change_pct:+.2f}%)",
                delta_color="normal" if price_change >= 0 else "inverse"
            )
        
        with col2:
            ytd_return = ((current_price - index_data['Close'].iloc[0]) / index_data['Close'].iloc[0]) * 100
            st.metric("YTD Return", f"{ytd_return:+.2f}%")
        
        with col3:
            high_52w = index_data['High'].max()
            st.metric("52-Week High", f"₹{high_52w:,.2f}")
        
        with col4:
            low_52w = index_data['Low'].min()
            st.metric("52-Week Low", f"₹{low_52w:,.2f}")
        
        # Price chart
        st.subheader("Nifty 50 Price Chart")
        
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=index_data['Date'],
            open=index_data['Open'],
            high=index_data['High'],
            low=index_data['Low'],
            close=index_data['Close'],
            name="Nifty 50"
        ))
        
        # Add moving averages
        fig.add_trace(go.Scatter(
            x=index_data['Date'],
            y=index_data['Close'].rolling(window=20).mean(),
            mode='lines',
            name='SMA 20',
            line=dict(color='orange', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=index_data['Date'],
            y=index_data['Close'].rolling(window=50).mean(),
            mode='lines',
            name='SMA 50',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title="Nifty 50 Price Movement",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume analysis
        st.subheader("Volume Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Volume chart
            volume_fig = px.bar(
                index_data,
                x='Date',
                y='Volume',
                title="Trading Volume"
            )
            st.plotly_chart(volume_fig, use_container_width=True)
        
        with col2:
            # Volume statistics
            avg_volume = index_data['Volume'].mean()
            current_volume = index_data['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume
            
            st.metric("Current Volume", f"{current_volume:,.0f}")
            st.metric("Average Volume", f"{avg_volume:,.0f}")
            st.metric("Volume Ratio", f"{volume_ratio:.2f}x")
        
        # Market statistics
        st.subheader("Market Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Returns distribution
            returns = index_data['Close'].pct_change().dropna()
            
            returns_fig = px.histogram(
                x=returns,
                title="Daily Returns Distribution",
                nbins=30
            )
            returns_fig.update_layout(xaxis_title="Daily Return", yaxis_title="Frequency")
            st.plotly_chart(returns_fig, use_container_width=True)
        
        with col2:
            # Volatility analysis
            volatility = returns.rolling(window=20).std() * np.sqrt(252)
            
            vol_fig = px.line(
                x=index_data['Date'][19:],
                y=volatility,
                title="20-Day Rolling Volatility"
            )
            vol_fig.update_layout(xaxis_title="Date", yaxis_title="Annualized Volatility")
            st.plotly_chart(vol_fig, use_container_width=True)
        
        with col3:
            # Performance metrics
            st.markdown("**Performance Metrics**")
            
            metrics_data = {
                "Metric": ["Total Return", "Annualized Return", "Sharpe Ratio", "Max Drawdown"],
                "Value": [
                    f"{((current_price / index_data['Close'].iloc[0]) - 1) * 100:.2f}%",
                    f"{((current_price / index_data['Close'].iloc[0]) ** (252/len(index_data)) - 1) * 100:.2f}%",
                    f"{returns.mean() / returns.std() * np.sqrt(252):.2f}",
                    f"{((index_data['Close'] / index_data['Close'].expanding().max()) - 1).min() * 100:.2f}%"
                ]
            }
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in overview: {e}")

def show_technical_analysis(period, selected_stock):
    """Show technical analysis for selected stock"""
    st.header(f"🔍 Technical Analysis - {selected_stock}")
    
    try:
        # Get stock data
        ticker = yf.Ticker(selected_stock)
        stock_data = ticker.history(period=period)
        
        if stock_data.empty:
            st.error(f"No data available for {selected_stock}")
            return
        
        stock_data.reset_index(inplace=True)
        
        # Calculate technical indicators
        tech_indicators = TechnicalAnalysis(stock_data)
        stock_data = tech_indicators.calculate_all_indicators()
        
        # Price and indicators chart
        st.subheader("Price and Technical Indicators")
        
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price & Moving Averages', 'RSI', 'MACD', 'Volume'),
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )
        
        # Price chart
        fig.add_trace(go.Candlestick(
            x=stock_data['Date'],
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close'],
            name="Price"
        ), row=1, col=1)
        
        # Moving averages
        fig.add_trace(go.Scatter(
            x=stock_data['Date'],
            y=stock_data['SMA_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='orange')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=stock_data['Date'],
            y=stock_data['SMA_50'],
            mode='lines',
            name='SMA 50',
            line=dict(color='red')
        ), row=1, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(
            x=stock_data['Date'],
            y=stock_data['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='purple')
        ), row=2, col=1)
        
        # RSI overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        fig.add_trace(go.Scatter(
            x=stock_data['Date'],
            y=stock_data['MACD'],
            mode='lines',
            name='MACD',
            line=dict(color='blue')
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=stock_data['Date'],
            y=stock_data['MACD_Signal'],
            mode='lines',
            name='Signal',
            line=dict(color='red')
        ), row=3, col=1)
        
        # Volume
        fig.add_trace(go.Bar(
            x=stock_data['Date'],
            y=stock_data['Volume'],
            name='Volume',
            marker_color='lightblue'
        ), row=4, col=1)
        
        fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical signals
        st.subheader("Technical Signals")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_rsi = stock_data['RSI'].iloc[-1]
            rsi_signal = "Oversold" if current_rsi < 30 else "Overbought" if current_rsi > 70 else "Neutral"
            
            st.metric("RSI", f"{current_rsi:.2f}", rsi_signal)
        
        with col2:
            current_macd = stock_data['MACD'].iloc[-1]
            macd_signal = stock_data['MACD_Signal'].iloc[-1]
            macd_signal_text = "Bullish" if current_macd > macd_signal else "Bearish"
            
            st.metric("MACD", f"{current_macd:.4f}", macd_signal_text)
        
        with col3:
            current_price = stock_data['Close'].iloc[-1]
            sma_20 = stock_data['SMA_20'].iloc[-1]
            trend_signal = "Above MA20" if current_price > sma_20 else "Below MA20"
            
            st.metric("Trend", f"₹{current_price:.2f}", trend_signal)
        
        # Support and resistance levels
        st.subheader("Support & Resistance Levels")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            support_level = stock_data['Low'].rolling(window=20).min().iloc[-1]
            st.metric("Support Level", f"₹{support_level:.2f}")
        
        with col2:
            resistance_level = stock_data['High'].rolling(window=20).max().iloc[-1]
            st.metric("Resistance Level", f"₹{resistance_level:.2f}")
        
        with col3:
            current_price = stock_data['Close'].iloc[-1]
            support_distance = ((current_price - support_level) / current_price) * 100
            st.metric("Distance from Support", f"{support_distance:.2f}%")
        
    except Exception as e:
        st.error(f"Error in technical analysis: {e}")

def show_sector_analysis(period):
    """Show sector-wise analysis"""
    st.header("🏭 Sector Analysis")
    
    try:
        nifty_data = Nifty50Data()
        
        with st.spinner("Analyzing sector performance..."):
            sector_analysis = nifty_data.get_sector_analysis(period=period)
        
        if not sector_analysis:
            st.error("Failed to get sector analysis")
            return
        
        # Sector performance overview
        st.subheader("Sector Performance Overview")
        
        # Create sector performance dataframe
        sector_data = []
        for sector, data in sector_analysis.items():
            sector_data.append({
                'Sector': sector,
                'Average Return (%)': round(data['avg_return'] * 100, 2),
                'Top Performer': data['top_performer'],
                'Worst Performer': data['worst_performer'],
                'Number of Stocks': len(data['stocks'])
            })
        
        sector_df = pd.DataFrame(sector_data)
        sector_df = sector_df.sort_values('Average Return (%)', ascending=False)
        
        # Display sector performance
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Sector performance chart
            fig = px.bar(
                sector_df,
                x='Sector',
                y='Average Return (%)',
                title="Sector Performance",
                color='Average Return (%)',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(sector_df, use_container_width=True)
        
        # Top and worst performing sectors
        st.subheader("Top & Worst Performing Sectors")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🏆 Top Performers**")
            top_sectors = sector_df.head(3)
            for _, row in top_sectors.iterrows():
                st.markdown(f"**{row['Sector']}**: {row['Average Return (%)']}%")
        
        with col2:
            st.markdown("**📉 Worst Performers**")
            worst_sectors = sector_df.tail(3)
            for _, row in worst_sectors.iterrows():
                st.markdown(f"**{row['Sector']}**: {row['Average Return (%)']}%")
        
        # Detailed sector analysis
        st.subheader("Detailed Sector Analysis")
        
        selected_sector = st.selectbox(
            "Select Sector for Detailed Analysis",
            list(sector_analysis.keys())
        )
        
        if selected_sector:
            sector_data = sector_analysis[selected_sector]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**{selected_sector} Sector Details**")
                
                # Sector statistics
                st.metric("Average Return", f"{sector_data['avg_return']*100:.2f}%")
                st.metric("Total Return", f"{sector_data['total_return']*100:.2f}%")
                st.metric("Number of Stocks", len(sector_data['stocks']))
                st.metric("Top Performer", sector_data['top_performer'])
                st.metric("Worst Performer", sector_data['worst_performer'])
            
            with col2:
                # Individual stock performance
                st.markdown("**Individual Stock Performance**")
                
                stock_performance = []
                for stock, data in sector_data['stocks'].items():
                    stock_performance.append({
                        'Stock': stock,
                        'Return (%)': round(data['returns'] * 100, 2),
                        'Current Price': f"₹{data['current_price']:.2f}"
                    })
                
                stock_df = pd.DataFrame(stock_performance)
                stock_df = stock_df.sort_values('Return (%)', ascending=False)
                
                st.dataframe(stock_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in sector analysis: {e}")

def show_predictions(prediction_days, selected_model):
    """Show Nifty 50 predictions"""
    st.header("🔮 Nifty 50 Predictions")
    
    try:
        # Initialize prediction models
        predictions = Nifty50Predictions()
        
        # Get historical data for training
        st.subheader("Model Training")
        
        with st.spinner("Training prediction models..."):
            nifty_data = Nifty50Data()
            training_data = nifty_data.get_nifty50_index_data(period="2y")
            
            if not training_data.empty:
                # Prepare features and train models
                feature_data = predictions.prepare_features(training_data)
                if not feature_data.empty:
                    predictions.train_models(feature_data)
                    st.success("Models trained successfully!")
                else:
                    st.error("Failed to prepare features for training")
                    return
            else:
                st.error("No training data available")
                return
        
        # Make predictions
        st.subheader("Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Next day prediction
            if selected_model == "Ensemble":
                prediction = predictions.ensemble_prediction()
            else:
                prediction = predictions.predict_next_day(selected_model)
            
            if 'error' not in prediction:
                st.markdown("**📊 Next Day Prediction**")
                
                # Prediction card
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>{prediction['predicted_direction']} 📈</h3>
                    <p><strong>Predicted Return:</strong> {prediction['predicted_return']*100:.2f}%</p>
                    <p><strong>Current Level:</strong> ₹{prediction['current_level']:,.2f}</p>
                    <p><strong>Predicted Level:</strong> ₹{prediction['predicted_level']:,.2f}</p>
                    <p><strong>Confidence:</strong> {prediction['confidence']*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"Prediction failed: {prediction['error']}")
        
        with col2:
            # Multi-day predictions
            if selected_model == "Ensemble":
                multi_predictions = predictions.predict_multiple_days(prediction_days, "RandomForest")
            else:
                multi_predictions = predictions.predict_multiple_days(prediction_days, selected_model)
            
            if multi_predictions and 'error' not in multi_predictions[0]:
                st.markdown(f"**📅 {prediction_days}-Day Forecast**")
                
                # Create forecast chart
                forecast_data = []
                for pred in multi_predictions:
                    forecast_data.append({
                        'Day': pred['day'],
                        'Predicted Level': pred['predicted_level'],
                        'Return (%)': pred['predicted_return'] * 100
                    })
                
                forecast_df = pd.DataFrame(forecast_data)
                
                fig = px.line(
                    forecast_df,
                    x='Day',
                    y='Predicted Level',
                    title=f"{prediction_days}-Day Price Forecast",
                    markers=True
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Forecast table
                st.dataframe(forecast_df, use_container_width=True)
            else:
                st.error("Failed to generate multi-day predictions")
        
        # Model performance summary
        st.subheader("Model Performance Summary")
        
        performance_summary = predictions.get_model_performance_summary()
        
        if 'error' not in performance_summary:
            # Create performance comparison
            performance_data = []
            for model, data in performance_summary.items():
                performance_data.append({
                    'Model': model,
                    'Last Prediction (%)': round(data['last_prediction'] * 100, 2),
                    'Direction': data['predicted_direction'],
                    'Confidence (%)': round(data['confidence'] * 100, 1)
                })
            
            perf_df = pd.DataFrame(performance_data)
            st.dataframe(perf_df, use_container_width=True)
            
            # Feature importance for tree-based models
            if selected_model in ['RandomForest', 'XGBoost', 'LightGBM']:
                if selected_model in predictions.feature_importance:
                    st.subheader(f"Feature Importance - {selected_model}")
                    
                    feature_importance = predictions.feature_importance[selected_model]
                    feature_df = pd.DataFrame([
                        {'Feature': k, 'Importance': v} 
                        for k, v in feature_importance.items()
                    ]).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(
                        feature_df.head(15),
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title=f"Top 15 Features - {selected_model}"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Failed to get performance summary: {performance_summary['error']}")
        
        # Export predictions
        st.subheader("Export Predictions")
        
        if st.button("Export Predictions Report"):
            predictions.export_predictions_report()
            st.success("Predictions report exported successfully!")
        
    except Exception as e:
        st.error(f"Error in predictions: {e}")

def show_market_breadth(period):
    """Show market breadth analysis"""
    st.header("📊 Market Breadth Analysis")
    
    try:
        nifty_data = Nifty50Data()
        
        with st.spinner("Calculating market breadth..."):
            market_breadth = nifty_data.get_market_breadth()
            volatility_analysis = nifty_data.get_volatility_analysis(period=period)
        
        if not market_breadth:
            st.error("Failed to get market breadth data")
            return
        
        # Market breadth overview
        st.subheader("Market Breadth Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Advancing Stocks",
                market_breadth['advancing'],
                f"{market_breadth['advancing_percentage']:.1f}%"
            )
        
        with col2:
            st.metric(
                "Declining Stocks",
                market_breadth['declining'],
                f"{market_breadth['declining_percentage']:.1f}%"
            )
        
        with col3:
            st.metric(
                "Advance/Decline Ratio",
                f"{market_breadth['advance_decline_ratio']:.2f}"
            )
        
        with col4:
            st.metric(
                "Unchanged",
                market_breadth['unchanged']
            )
        
        # New highs and lows
        st.subheader("New Highs & Lows")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("New 52-Week Highs", market_breadth['new_highs'])
        
        with col2:
            st.metric("New 52-Week Lows", market_breadth['new_lows'])
        
        # Market breadth visualization
        st.subheader("Market Breadth Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart of advancing/declining/unchanged
            breadth_data = {
                'Category': ['Advancing', 'Declining', 'Unchanged'],
                'Count': [
                    market_breadth['advancing'],
                    market_breadth['declining'],
                    market_breadth['unchanged']
                ]
            }
            
            fig = px.pie(
                breadth_data,
                values='Count',
                names='Category',
                title="Market Breadth Distribution",
                color_discrete_map={
                    'Advancing': 'green',
                    'Declining': 'red',
                    'Unchanged': 'gray'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Bar chart of new highs vs lows
            new_hl_data = {
                'Type': ['New Highs', 'New Lows'],
                'Count': [market_breadth['new_highs'], market_breadth['new_lows']]
            }
            
            fig = px.bar(
                new_hl_data,
                x='Type',
                y='Count',
                title="New Highs vs New Lows",
                color=['green', 'red']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Volatility analysis
        if volatility_analysis:
            st.subheader("Volatility Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Daily Volatility",
                    f"{volatility_analysis['daily_volatility']*100:.2f}%"
                )
            
            with col2:
                st.metric(
                    "Annualized Volatility",
                    f"{volatility_analysis['annualized_volatility']*100:.2f}%"
                )
            
            with col3:
                st.metric(
                    "Current Rolling Volatility",
                    f"{volatility_analysis['current_rolling_volatility']*100:.2f}%"
                )
            
            # Volatility trend
            vol_trend = volatility_analysis['volatility_trend']
            if vol_trend == 'increasing':
                st.warning("⚠️ Volatility is increasing - Market may become more volatile")
            elif vol_trend == 'decreasing':
                st.success("✅ Volatility is decreasing - Market may become more stable")
            
            # Risk metrics
            st.subheader("Risk Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Value at Risk (95%)",
                    f"{volatility_analysis['var_95']*100:.2f}%"
                )
                
                st.metric(
                    "Value at Risk (99%)",
                    f"{volatility_analysis['var_99']*100:.2f}%"
                )
            
            with col2:
                st.metric(
                    "Maximum Daily Return",
                    f"{volatility_analysis['max_daily_return']*100:.2f}%"
                )
                
                st.metric(
                    "Minimum Daily Return",
                    f"{volatility_analysis['min_daily_return']*100:.2f}%"
                )
        
        # Market sentiment interpretation
        st.subheader("Market Sentiment Interpretation")
        
        # Calculate sentiment score
        sentiment_score = 0
        
        # Advance/Decline ratio contribution
        if market_breadth['advance_decline_ratio'] > 1.5:
            sentiment_score += 2
        elif market_breadth['advance_decline_ratio'] > 1.0:
            sentiment_score += 1
        elif market_breadth['advance_decline_ratio'] < 0.5:
            sentiment_score -= 2
        elif market_breadth['advance_decline_ratio'] < 1.0:
            sentiment_score -= 1
        
        # New highs vs lows contribution
        if market_breadth['new_highs'] > market_breadth['new_lows'] * 2:
            sentiment_score += 1
        elif market_breadth['new_lows'] > market_breadth['new_highs'] * 2:
            sentiment_score -= 1
        
        # Volatility contribution
        if volatility_analysis and volatility_analysis['volatility_trend'] == 'decreasing':
            sentiment_score += 1
        elif volatility_analysis and volatility_analysis['volatility_trend'] == 'increasing':
            sentiment_score -= 1
        
        # Display sentiment
        if sentiment_score >= 2:
            st.success("🟢 Bullish Market Sentiment - Market showing strength")
        elif sentiment_score >= 0:
            st.info("🟡 Neutral Market Sentiment - Mixed signals")
        else:
            st.error("🔴 Bearish Market Sentiment - Market showing weakness")
        
        st.markdown(f"**Sentiment Score: {sentiment_score}**")
        
    except Exception as e:
        st.error(f"Error in market breadth analysis: {e}")

if __name__ == "__main__":
    main()
