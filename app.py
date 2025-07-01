import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# --- Robustly Load Environment Variables ---
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path, override=True)

from agent.config import AnalysisConfig
from agent.state import AgentState
from financial_agent_modular import StockAnalysisAgent

# --- Streamlit Page Configuration (moved to top to prevent re-runs) ---
st.set_page_config(
    page_title="AI Financial Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Better Styling ---
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    .analysis-card {
        background: #1e2130;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        color: #f0f2f6;
    }
    
    .news-card {
        background: #262b3e;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #28a745;
        color: #f0f2f6;
    }
    
    .news-card a {
        color: #4da6ff !important;
    }
    
    .news-card p {
        color: #e0e0e0 !important;
    }
    
    .sidebar-header {
        color: #1f77b4;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .stAlert > div {
        border-radius: 8px;
    }
    
    /* Dark theme adjustments */
    .dataframe {
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# --- App Header ---
st.markdown('<div class="main-header">🤖 AI Financial Analyst</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced Multi-Agent Stock Analysis Platform</div>', unsafe_allow_html=True)

# --- Helper Functions for Visualizations ---
@st.cache_data(ttl=3600)  # Cache to prevent repeated execution
def create_stock_chart(df, ticker):
    """Create an interactive stock price chart with technical indicators"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price & Bollinger Bands', 'MACD', 'RSI'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Price and Bollinger Bands
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ), row=1, col=1)
    
    if 'BB_High' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['BB_High'],
            line=dict(color='rgba(255,0,0,0.3)'),
            name='BB Upper'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['BB_Low'],
            line=dict(color='rgba(255,0,0,0.3)'),
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.1)',
            name='BB Lower'
        ), row=1, col=1)
    
    # MACD
    if 'MACD' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['MACD'],
            line=dict(color='blue'),
            name='MACD'
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['MACD_Signal'],
            line=dict(color='red'),
            name='Signal'
        ), row=2, col=1)
    
    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['RSI'],
            line=dict(color='purple'),
            name='RSI'
        ), row=3, col=1)
        
        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    fig.update_layout(
        title=f'{ticker} Stock Analysis',
        xaxis_rangeslider_visible=False,
        height=800,
        template='plotly_dark'  # Changed to dark theme
    )
    
    return fig

def create_metrics_cards(financial_metrics, latest_data):
    """Create metric cards for key financial data"""
    cols = st.columns(4)
    
    metrics = [
        ("Current Price", f"${latest_data['Close']:.2f}", "📈"),
        ("Market Cap", format_large_number(financial_metrics.get('Market Cap')), "💰"),
        ("P/E Ratio", f"{financial_metrics.get('Forward P/E', 'N/A'):.2f}" if financial_metrics.get('Forward P/E') else "N/A", "📊"),
        ("RSI", f"{latest_data.get('RSI', 0):.1f}", "🎯")
    ]
    
    for i, (label, value, icon) in enumerate(metrics):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 1.5rem;">{icon}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

def format_large_number(num):
    """Format large numbers with appropriate suffixes"""
    if num is None:
        return "N/A"
    
    if num >= 1e12:
        return f"${num/1e12:.1f}T"
    elif num >= 1e9:
        return f"${num/1e9:.1f}B"
    elif num >= 1e6:
        return f"${num/1e6:.1f}M"
    else:
        return f"${num:,.0f}"

def create_news_display(news_data):
    """Create an attractive news display"""
    if not news_data:
        st.info("📰 No recent news found for this ticker.")
        return
    
    st.markdown("### 📰 Recent News & Sentiment")
    
    for article in news_data:
        st.markdown(f"""
        <div class="news-card">
            <h4 style="margin: 0 0 0.5rem 0; color: #4da6ff;">
                <a href="{article['url']}" target="_blank" style="text-decoration: none;">
                    {article['title']}
                </a>
            </h4>
            <p style="margin: 0; font-size: 0.9rem;">
                {article.get('summary', 'No summary available')}
            </p>
        </div>
        """, unsafe_allow_html=True)

@st.cache_data(ttl=3600)  # Cache to prevent repeated execution
def create_performance_chart(df):
    """Create a performance comparison chart"""
    if df.empty:
        return None
    
    # Calculate returns
    df['Returns'] = df['Close'].pct_change() * 100
    df['Cumulative_Returns'] = (1 + df['Close'].pct_change()).cumprod() - 1
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Cumulative_Returns'] * 100,
        mode='lines',
        name='Cumulative Returns (%)',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        title='Cumulative Returns Over Time',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns (%)',
        template='plotly_dark',  # Changed to dark theme
        height=400
    )
    
    return fig

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.markdown('<div class="sidebar-header">📊 Analysis Configuration</div>', unsafe_allow_html=True)
    
    ticker = st.text_input("📈 Stock Ticker", "NVDA", help="Enter a valid stock ticker symbol").upper()
    
    st.markdown("---")
    st.markdown("### 🎯 Analysis Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        period = st.selectbox("📅 Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    with col2:
        interval = st.selectbox("⏱️ Interval", ["1d", "5d", "1wk", "1mo"], index=0)
    
    st.markdown("### 🔧 Technical Indicators")
    rsi_window = st.slider("RSI Window", 10, 30, 14)
    bb_window = st.slider("Bollinger Bands Window", 10, 30, 20)
    
    st.markdown("---")
    analyze_button = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.info("💡 **Pro Tip**: This AI agent combines technical analysis, fundamental metrics, news sentiment, and analyst ratings for comprehensive insights.")

# --- Main Dashboard Area ---
# Use a session state to prevent multiple executions
if 'processed' not in st.session_state:
    st.session_state.processed = False

# Only process if button is clicked and not already processed
if analyze_button and not st.session_state.processed:
    st.session_state.processed = True  # Mark as processed to prevent double execution
    
    if not ticker:
        st.error("⚠️ Please enter a stock ticker.")
        st.session_state.processed = False  # Reset for next attempt
    else:
        # Initialize the agent and config
        config = AnalysisConfig(period=period, rsi_window=rsi_window, bb_window=bb_window)
        agent = StockAnalysisAgent()

        # Use a status container for a cleaner loading experience
        with st.status(f"🔍 Running comprehensive analysis for {ticker}...", expanded=True) as status:
            try:
                # Run the analysis
                final_report, final_state = agent.run_analysis(ticker, config)
                
                status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

                # --- Display Results ---
                st.markdown("---")
                
                # Key Metrics Cards
                if not final_state['df'].empty and final_state['financial_metrics']:
                    latest_data = final_state['df'].iloc[-1]
                    create_metrics_cards(final_state['financial_metrics'], latest_data)
                
                st.markdown("---")
                
                # Create two columns for layout
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Stock Chart
                    if not final_state['df'].empty:
                        st.markdown("### 📈 Technical Analysis")
                        chart = create_stock_chart(final_state['df'], ticker)
                        st.plotly_chart(chart, use_container_width=True)
                        
                        # Performance Chart
                        perf_chart = create_performance_chart(final_state['df'])
                        if perf_chart:
                            st.plotly_chart(perf_chart, use_container_width=True)
                
                with col2:
                    # Financial Metrics
                    if final_state['financial_metrics']:
                        st.markdown("### 💰 Financial Metrics")
                        metrics_df = pd.DataFrame(
                            list(final_state['financial_metrics'].items()),
                            columns=['Metric', 'Value']
                        )
                        st.dataframe(metrics_df, use_container_width=True, height=400)
                    
                    # Analyst Ratings
                    if final_state['analyst_ratings']:
                        st.markdown("### 📊 Analyst Consensus")
                        st.markdown(f"""
                        <div class="analysis-card">
                            {final_state['analyst_ratings']}
                        </div>
                        """, unsafe_allow_html=True)

                # News Section
                create_news_display(final_state['news'])
                
                st.markdown("---")
                
                # Final Report
                st.markdown("### 🎯 AI Investment Analysis")
                st.markdown(f"""
                <div class="analysis-card">
                    {final_report}
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                status.update(label="❌ Analysis Failed", state="error")
                st.error(f"🚨 An unexpected error occurred: {e}")
                st.info("💡 Try checking your API keys and internet connection.")
                st.session_state.processed = False  # Reset for next attempt

elif not analyze_button:
    # Reset the processed flag when the button is not clicked
    st.session_state.processed = False
    
    # Welcome screen when no analysis is running
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h3>🎯 Ready to Analyze Your Next Investment?</h3>
            <p style="color: #666; font-size: 1.1rem;">
                Enter a stock ticker in the sidebar and click "Run Full Analysis" to get started.
            </p>
            <div style="margin: 2rem 0;">
                <div style="display: inline-block; margin: 0.5rem; padding: 1rem; background: #262b3e; border-radius: 8px; color: #f0f2f6;">
                    📈 Technical Analysis
                </div>
                <div style="display: inline-block; margin: 0.5rem; padding: 1rem; background: #262b3e; border-radius: 8px; color: #f0f2f6;">
                    💰 Financial Metrics
                </div>
                <div style="display: inline-block; margin: 0.5rem; padding: 1rem; background: #262b3e; border-radius: 8px; color: #f0f2f6;">
                    📰 News Sentiment
                </div>
                <div style="display: inline-block; margin: 0.5rem; padding: 1rem; background: #262b3e; border-radius: 8px; color: #f0f2f6;">
                    📊 Analyst Ratings
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🤖 Powered by AI • Built with Streamlit • Financial data from yfinance</p>
    <p><em>Disclaimer: This tool is for educational purposes only. Not financial advice.</em></p>
</div>
""", unsafe_allow_html=True)