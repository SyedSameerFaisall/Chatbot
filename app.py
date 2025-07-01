import streamlit as st
import sys
import os
import pandas as pd

# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from agent.config import AnalysisConfig
from agent.state import AgentState
from financial_agent_modular import StockAnalysisAgent
from agent.charts import create_stock_chart # Import the new charting function

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
                # This is a placeholder to run the agent and get the final state
                # In a real app, you might stream logs here if the agent supported it
                st.write("🧠 Initializing Agent...")
                initial_state = AgentState(
                    messages=[], ticker=ticker, df=pd.DataFrame(), config=config,
                    news=[], analyst_ratings="", financial_metrics={}, final_report=""
                )
                
                st.write("⚙️ Executing analysis graph...")
                final_state = agent.graph.invoke(initial_state)
                
                st.write("✅ Analysis complete!")
                status.update(label="Analysis Complete!", state="complete", expanded=False)

                # --- Display the Results in a Sophisticated Layout ---
                
                # Get data from the final state
                report = final_state.get('final_report', "Report not found.")
                df_data = final_state.get('df')
                
                # Create the interactive chart
                if not df_data.empty:
                    st.subheader("Interactive Technical Chart")
                    chart = create_stock_chart(df_data, ticker)
                    st.plotly_chart(chart, use_container_width=True)
                
                st.divider()

                # Display the final report
                st.subheader("Comprehensive Investment Report")
                st.markdown(report)

            except Exception as e:
                status.update(label="Analysis Failed", state="error")
                st.error(f"An unexpected error occurred: {e}")