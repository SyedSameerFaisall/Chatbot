import streamlit as st
import sys
import os
import pandas as pd
from dotenv import load_dotenv

# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# --- Robustly Load Environment Variables ---
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path, override=True)

from agent.config import AnalysisConfig
from agent.state import AgentState
from financial_agent_modular import StockAnalysisAgent

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="AI Financial Analyst",
    page_icon="🤖",
    layout="wide"
)

# --- App Header ---
st.title("🤖 AI Financial Analyst")
st.markdown("An advanced agent that performs a comprehensive, multi-faceted analysis of any stock ticker.")

st.divider()

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("Analysis Configuration")
    ticker = st.text_input("Enter Stock Ticker", "NVDA").upper()
    
    st.subheader("Analysis Parameters")
    period = st.selectbox("Period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
    rsi_window = st.slider("RSI Window", 10, 30, 14)
    bb_window = st.slider("Bollinger Bands Window", 10, 30, 20)

    analyze_button = st.button("Run Full Analysis", type="primary", use_container_width=True)
    
    st.info("This agent uses AI to synthesize technicals, valuation, news, and analyst ratings into a final investment recommendation.")

# --- Main Dashboard Area ---
if analyze_button:
    if not ticker:
        st.error("Please enter a stock ticker.")
    else:
        # Initialize the agent and config
        config = AnalysisConfig(period=period, rsi_window=rsi_window, bb_window=bb_window)
        agent = StockAnalysisAgent()

        # Use a status container for a cleaner loading experience
        with st.status(f"Running comprehensive analysis for {ticker}...", expanded=True) as status:
            try:
                # The agent's print statements will show progress in the terminal
                final_report = agent.run_analysis(ticker, config)
                
                status.update(label="Analysis Complete!", state="complete", expanded=False)

                # --- Display the Final Report in a Wider Centered Layout ---
                # Changed the column ratios to reduce side margins
                col1, col2, col3 = st.columns([0.5, 7, 0.5]) 

                with col2: # All content goes in the wider central column
                    st.subheader(f"Comprehensive Investment Report for {ticker}")
                    st.markdown(final_report)

            except Exception as e:
                status.update(label="Analysis Failed", state="error")
                st.error(f"An unexpected error occurred: {e}")
