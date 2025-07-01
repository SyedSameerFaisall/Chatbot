import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage

# Import the state definition and all the tools
from .state import AgentState
from .tools import (
    fetch_stock_data, 
    compute_technical_indicators, 
    get_news_search_results, 
    scrape_and_summarize_article,
    get_analyst_ratings,
    get_financial_metrics,
    get_llm_analysis
)

def start_analysis(state: AgentState) -> AgentState:
    """
    Entry node that takes the initial ticker and config.
    """
    ticker = state["ticker"]
    print(f"📊 Starting analysis for {ticker}...")
    state["messages"] = [HumanMessage(content=f"Analyze {ticker} stock performance.")]
    state["df"] = pd.DataFrame()
    state["news"] = []
    state["analyst_ratings"] = ""
    state["financial_metrics"] = {}
    return state

def fetch_data_node(state: AgentState) -> AgentState:
    """
    Fetches the historical stock data and computes technical indicators.
    """
    df = fetch_stock_data(ticker=state["ticker"], config=state["config"])
    df_with_indicators = compute_technical_indicators(df=df, config=state["config"])
    state["df"] = df_with_indicators
    return state

def fetch_news_node(state: AgentState) -> AgentState:
    """
    Gets news search results and then scrapes and summarizes each one.
    """
    search_results = get_news_search_results(ticker=state["ticker"])
    summarized_articles = []
    for article in search_results:
        summary = scrape_and_summarize_article(article["url"])
        if summary:
            article["summary"] = summary
            summarized_articles.append(article)
    state["news"] = summarized_articles
    return state

def fetch_ratings_node(state: AgentState) -> AgentState:
    """
    Fetches a summary of analyst ratings.
    """
    ratings_summary = get_analyst_ratings(ticker=state["ticker"])
    state["analyst_ratings"] = ratings_summary
    return state

def fetch_financials_node(state: AgentState) -> AgentState:
    """
    Fetches key financial metrics.
    """
    metrics = get_financial_metrics(ticker=state["ticker"])
    state["financial_metrics"] = metrics
    return state

def synthesize_report_node(state: AgentState) -> AgentState:
    """
    Node to perform the final analysis with the LLM, using all gathered data.
    """
    # The get_llm_analysis function now takes the whole state dictionary
    analysis_report = get_llm_analysis(state)
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