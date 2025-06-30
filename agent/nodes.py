import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage

# Import the state definition and the tools
from .state import AgentState
from .tools import fetch_stock_data, compute_technical_indicators, get_llm_analysis

def start_analysis(state: AgentState) -> AgentState:
    """
    Entry node that takes the initial ticker and prepares the first message.
    """
    ticker = state.get("ticker")
    print(f"📊 Starting analysis for {ticker}...")
    user_msg = HumanMessage(content=f"Analyze {ticker} stock performance.")
    return AgentState(messages=[user_msg], ticker=ticker, df=pd.DataFrame())

def fetch_data_node(state: AgentState) -> AgentState:
    """
    Node to fetch stock data using the tool and update the state.
    """
    ticker = state["ticker"]
    df = fetch_stock_data(ticker)
    
    if df.empty:
        ai_msg = AIMessage(content=f"Error: Could not retrieve stock data for {ticker}.")
    else:
        ai_msg = AIMessage(content=f"Successfully retrieved {len(df)} days of stock data for {ticker}.")
        
    state["df"] = df
    state["messages"].append(ai_msg)
    return state

def analyze_indicators_node(state: AgentState) -> AgentState:
    """
    Node to compute technical indicators and update the state.
    """
    df = state["df"]
    df_with_indicators = compute_technical_indicators(df)
    
    if df_with_indicators.empty:
        ai_msg = AIMessage(content="Error: Could not compute technical indicators.")
    else:
        ai_msg = AIMessage(content=f"Technical indicators computed successfully. {len(df_with_indicators)} complete data points.")
        
    state["df"] = df_with_indicators
    state["messages"].append(ai_msg)
    return state

def analyze_with_llm_node(state: AgentState) -> AgentState:
    """
    Node to perform the final analysis with the LLM.
    """
    analysis_report = get_llm_analysis(
        ticker=state["ticker"],
        df=state["df"],
        messages=state["messages"]
    )
    ai_msg = AIMessage(content=analysis_report)
    state["messages"].append(ai_msg)
    return state

def final_response_node(state: AgentState) -> AgentState:
    """
    Node to display the final analysis report to the user.
    """
    print("\n" + "="*60)
    print("📈 FINANCIAL ANALYST INSIGHT:")
    print("="*60)
    final_report = state["messages"][-1].content
    print(final_report)
    print("="*60)
    return state