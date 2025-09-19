"""
Streamlit Web Application for Stock Market Analysis
Provides interactive dashboard for stock analysis, predictions, and visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.stock_data import StockData
from analysis.technical_indicators import TechnicalAnalysis
from analysis.ml_models import PricePredictor, SentimentAnalyzer

# Page configuration
st.set_page_config(
    page_title="Stock Market Analysis Dashboard",
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
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Market Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Stock symbol input
        symbol = st.text_input("Stock Symbol", value="AAPL", help="Enter stock symbol (e.g., AAPL, GOOGL, MSFT)")
        
        # Date range
        st.subheader("📅 Date Range")
        period = st.selectbox(
            "Time Period",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3
        )
        
        # Analysis options
        st.subheader("📊 Analysis Options")
        show_technical = st.checkbox("Show Technical Indicators", value=True)
        show_predictions = st.checkbox("Show Price Predictions", value=True)
        show_sentiment = st.checkbox("Show Sentiment Analysis", value=False)
        
        # Update button
        if st.button("🔄 Update Analysis", type="primary"):
            st.rerun()
    
    # Main content
    if symbol:
        try:
            # Initialize stock data
            stock_data = StockData(symbol)
            
            # Fetch data
            with st.spinner(f"Fetching data for {symbol}..."):
                data = stock_data.get_historical_data(period=period)
            
            if data.empty:
                st.error(f"No data found for symbol {symbol}")
                return
            
            # Convert to proper format for analysis
            data.set_index('Date', inplace=True)
            
            # Display basic info
            display_basic_info(stock_data, data)
            
            # Price chart
            display_price_chart(data, symbol)
            
            # Technical analysis
            if show_technical:
                display_technical_analysis(data, symbol)
            
            # Machine learning predictions
            if show_predictions:
                display_predictions(data, symbol)
            
            # Sentiment analysis
            if show_sentiment:
                display_sentiment_analysis(symbol)
            
            # Data summary
            display_data_summary(data, symbol)
            
        except Exception as e:
            st.error(f"Error analyzing {symbol}: {str(e)}")

def display_basic_info(stock_data: StockData, data: pd.DataFrame):
    """Display basic stock information"""
    
    st.header("📋 Stock Information")
    
    try:
        # Get real-time data
        real_time = stock_data.get_real_time_data()
        company_info = stock_data.get_company_info()
        
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current Price",
                f"${real_time['current_price']:.2f}",
                f"{real_time['change_percent']:.2f}%"
            )
        
        with col2:
            st.metric(
                "Volume",
                f"{real_time['volume']:,}",
                f"High: ${real_time['high']:.2f}"
            )
        
        with col3:
            st.metric(
                "Market Cap",
                f"${real_time['market_cap']:,}" if real_time['market_cap'] != 'N/A' else "N/A"
            )
        
        with col4:
            st.metric(
                "P/E Ratio",
                f"{real_time['pe_ratio']:.2f}" if real_time['pe_ratio'] != 'N/A' else "N/A"
            )
        
        # Company information
        st.subheader("🏢 Company Details")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Company:** {company_info['name']}")
            st.write(f"**Sector:** {company_info['sector']}")
            st.write(f"**Industry:** {company_info['industry']}")
        
        with col2:
            st.write(f"**Employees:** {company_info['employees']:,}" if company_info['employees'] != 'N/A' else "**Employees:** N/A")
            st.write(f"**Website:** {company_info['website']}")
            st.write(f"**Dividend Yield:** {real_time['dividend_yield']:.2f}%" if real_time['dividend_yield'] != 'N/A' else "**Dividend Yield:** N/A")
        
    except Exception as e:
        st.warning(f"Could not fetch real-time data: {str(e)}")

def display_price_chart(data: pd.DataFrame, symbol: str):
    """Display interactive price chart"""
    
    st.header("📈 Price Chart")
    
    # Create candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=symbol
    )])
    
    fig.update_layout(
        title=f"{symbol} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Volume chart
    fig_volume = px.bar(
        data, 
        x=data.index, 
        y='Volume',
        title=f"{symbol} Trading Volume"
    )
    
    fig_volume.update_layout(
        xaxis_title="Date",
        yaxis_title="Volume",
        height=300
    )
    
    st.plotly_chart(fig_volume, use_container_width=True)

def display_technical_analysis(data: pd.DataFrame, symbol: str):
    """Display technical analysis indicators"""
    
    st.header("📊 Technical Analysis")
    
    try:
        # Initialize technical analysis
        ta = TechnicalAnalysis(data)
        
        # Calculate indicators
        rsi = ta.calculate_rsi()
        macd_data = ta.calculate_macd()
        bb_data = ta.calculate_bollinger_bands()
        sma_data = ta.calculate_moving_averages()
        
        # Create subplots for indicators
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Price with Bollinger Bands', 'RSI', 'MACD'),
            vertical_spacing=0.1,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Price with Bollinger Bands
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name=symbol
            ),
            row=1, col=1
        )
        
        if len(bb_data['upper_band']) > 0:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=bb_data['upper_band'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='gray', dash='dash')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=bb_data['lower_band'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='gray', dash='dash'),
                    fill='tonexty'
                ),
                row=1, col=1
            )
        
        # RSI
        if len(rsi) > 0:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=rsi,
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple')
                ),
                row=2, col=1
            )
            
            # Add overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        if len(macd_data['macd_line']) > 0:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=macd_data['macd_line'],
                    mode='lines',
                    name='MACD',
                    line=dict(color='blue')
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=macd_data['signal_line'],
                    mode='lines',
                    name='Signal',
                    line=dict(color='orange')
                ),
                row=3, col=1
            )
        
        fig.update_layout(height=800, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical summary
        summary = ta.get_technical_summary()
        if summary:
            st.subheader("📋 Technical Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Trend", summary.get('trend', 'Unknown'))
                st.metric("RSI", f"{summary.get('rsi', 0):.2f}")
            
            with col2:
                st.metric("RSI Signal", summary.get('rsi_signal', 'Unknown'))
                st.metric("MACD", f"{summary.get('macd', 0):.4f}")
            
            with col3:
                st.metric("BB Position", summary.get('bollinger_position', 'Unknown'))
                st.metric("Volume", f"{summary.get('volume', 0):,}")
        
    except Exception as e:
        st.error(f"Error in technical analysis: {str(e)}")

def display_predictions(data: pd.DataFrame, symbol: str):
    """Display machine learning predictions"""
    
    st.header("🤖 Price Predictions")
    
    try:
        # Initialize predictor
        predictor = PricePredictor(symbol)
        
        # Train models
        with st.spinner("Training prediction models..."):
            results = predictor.train_models(data)
        
        if results:
            # Display model performance
            st.subheader("📊 Model Performance")
            
            # Create performance chart
            models = list(results.keys())
            r2_scores = [results[model]['r2'] for model in models]
            rmse_scores = [results[model]['rmse'] for model in models]
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_r2 = px.bar(
                    x=models,
                    y=r2_scores,
                    title="R² Scores",
                    labels={'x': 'Model', 'y': 'R² Score'}
                )
                st.plotly_chart(fig_r2, use_container_width=True)
            
            with col2:
                fig_rmse = px.bar(
                    x=models,
                    y=rmse_scores,
                    title="RMSE Scores",
                    labels={'x': 'Model', 'y': 'RMSE'}
                )
                st.plotly_chart(fig_rmse, use_container_width=True)
            
            # Make predictions
            with st.spinner("Making predictions..."):
                prediction = predictor.predict_next_day(data)
            
            if prediction:
                st.subheader("🔮 Price Forecast")
                
                # Display predictions
                predictions = prediction['predictions']
                current_price = prediction['current_price']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Current Price",
                        f"${current_price:.2f}"
                    )
                
                with col2:
                    ensemble_pred = predictions.get('ensemble', current_price)
                    change = ensemble_pred - current_price
                    change_percent = (change / current_price) * 100
                    
                    st.metric(
                        "Predicted Price",
                        f"${ensemble_pred:.2f}",
                        f"{change_percent:+.2f}%"
                    )
                
                with col3:
                    confidence = prediction.get('confidence', 0)
                    st.metric(
                        "Prediction Confidence",
                        f"{confidence:.1%}"
                    )
                
                # Individual model predictions
                st.subheader("📈 Model Predictions")
                
                pred_df = pd.DataFrame([
                    {'Model': model, 'Prediction': pred}
                    for model, pred in predictions.items()
                ])
                
                fig_pred = px.bar(
                    pred_df,
                    x='Model',
                    y='Prediction',
                    title="Predictions by Model"
                )
                st.plotly_chart(fig_pred, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in predictions: {str(e)}")

def display_sentiment_analysis(symbol: str):
    """Display sentiment analysis"""
    
    st.header("😊 Sentiment Analysis")
    
    try:
        analyzer = SentimentAnalyzer()
        
        # Sample news data (in real application, this would come from news API)
        sample_news = [
            {
                'title': f'{symbol} shows strong growth in Q4',
                'content': 'The company reported positive earnings and strong revenue growth.',
                'date': '2024-01-15',
                'source': 'Financial News'
            },
            {
                'title': f'{symbol} faces market challenges',
                'content': 'Recent market volatility has affected the stock performance.',
                'date': '2024-01-14',
                'source': 'Market Watch'
            }
        ]
        
        # Analyze sentiment
        sentiment_result = analyzer.analyze_news_sentiment(sample_news)
        
        if sentiment_result:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sentiment_score = sentiment_result['overall_sentiment']
                st.metric(
                    "Overall Sentiment",
                    f"{sentiment_score:.3f}",
                    help="Positive values indicate bullish sentiment"
                )
            
            with col2:
                confidence = sentiment_result['confidence']
                st.metric(
                    "Confidence",
                    f"{confidence:.1%}"
                )
            
            with col3:
                article_count = sentiment_result['article_count']
                st.metric(
                    "Articles Analyzed",
                    article_count
                )
            
            # Sentiment visualization
            fig_sentiment = px.bar(
                x=['Sentiment Score'],
                y=[sentiment_score],
                title="News Sentiment Score",
                color=[sentiment_score],
                color_continuous_scale=['red', 'yellow', 'green']
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in sentiment analysis: {str(e)}")

def display_data_summary(data: pd.DataFrame, symbol: str):
    """Display data summary statistics"""
    
    st.header("📊 Data Summary")
    
    try:
        # Calculate summary statistics
        summary_stats = {
            'Total Days': len(data),
            'Start Date': data.index[0].strftime('%Y-%m-%d'),
            'End Date': data.index[-1].strftime('%Y-%m-%d'),
            'Current Price': f"${data['Close'].iloc[-1]:.2f}",
            'Highest Price': f"${data['High'].max():.2f}",
            'Lowest Price': f"${data['Low'].min():.2f}",
            'Average Price': f"${data['Close'].mean():.2f}",
            'Price Volatility': f"${data['Close'].std():.2f}",
            'Total Volume': f"{data['Volume'].sum():,}",
            'Average Volume': f"{data['Volume'].mean():,.0f}",
            'Total Return': f"{((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100):.2f}%"
        }
        
        # Display in columns
        col1, col2, col3 = st.columns(3)
        
        for i, (key, value) in enumerate(summary_stats.items()):
            col = [col1, col2, col3][i % 3]
            with col:
                st.metric(key, value)
        
        # Price distribution
        st.subheader("📈 Price Distribution")
        
        fig_dist = px.histogram(
            data,
            x='Close',
            nbins=50,
            title=f"{symbol} Price Distribution"
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in data summary: {str(e)}")

if __name__ == "__main__":
    main()
