import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage

# Import the state definition and the tools
from .state import AgentState
from .tools import (
    fetch_stock_data, 
    compute_technical_indicators, 
    get_news_search_results, 
    scrape_and_summarize_article,
    get_llm_analysis
)

def start_analysis(state: AgentState) -> AgentState:
    """
    Entry node that takes the initial ticker and config, preparing the first message.
    """
    ticker = state["ticker"]
    print(f"📊 Starting analysis for {ticker}...")
    user_msg = HumanMessage(content=f"Analyze {ticker} stock performance.")
    
    state["messages"] = [user_msg]
    state["df"] = pd.DataFrame()
    state["news"] = [] # Initialize news as an empty list
    return state

def fetch_data_node(state: AgentState) -> AgentState:
    """
    Node to fetch stock data, passing the config from the state to the tool.
    """
    df = fetch_stock_data(ticker=state["ticker"], config=state["config"])
    state["df"] = df
    state["messages"].append(AIMessage(content=f"Fetched {len(df)} data points."))
    return state

def analyze_indicators_node(state: AgentState) -> AgentState:
    """
    Node to compute technical indicators, passing the config from the state to the tool.
    """
    df_with_indicators = compute_technical_indicators(df=state["df"], config=state["config"])
    state["df"] = df_with_indicators
    state["messages"].append(AIMessage(content="Computed technical indicators."))
    return state

def fetch_and_summarize_news_node(state: AgentState) -> AgentState:
    """
    Orchestrator node that gets news search results and then scrapes and summarizes each one.
    """
    search_results = get_news_search_results(ticker=state["ticker"])
    
    summarized_articles = []
    for article in search_results:
        summary = scrape_and_summarize_article(article["url"])
        
        # Only include the article if the summary is not empty
        if summary:
            article["summary"] = summary
            summarized_articles.append(article)

    ai_msg = AIMessage(content=f"Found and successfully summarized {len(summarized_articles)} relevant news articles.")
    
    state["news"] = summarized_articles
    state["messages"].append(ai_msg)
    return state

def analyze_with_llm_node(state: AgentState) -> AgentState:
    """
    Node to perform the final analysis with the LLM, now including news summaries.
    """
    analysis_report = get_llm_analysis(
        ticker=state["ticker"],
        df=state["df"],
        messages=state["messages"],
        news=state["news"]
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