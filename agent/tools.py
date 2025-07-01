import os
import pandas as pd
import yfinance as yf
import ta
import textwrap
import requests
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from typing import Sequence, List, Dict, Any
from tavily import TavilyClient

# Import the config class to use for type hinting
from .config import AnalysisConfig

def fetch_stock_data(ticker: str, config: AnalysisConfig) -> pd.DataFrame:
    """
    Fetches historical OHLCV stock data.
    """
    print(f"📈 Fetching stock data for {ticker}...")
    try:
        df = yf.Ticker(ticker).history(period=config.period, interval=config.interval).reset_index()
        if df.empty: raise ValueError(f"No data found for ticker {ticker}")
        return df
    except Exception as e:
        return pd.DataFrame()

def compute_technical_indicators(df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """
    Computes technical indicators for the stock data.
    """
    print("🔍 Computing technical indicators...")
    if df.empty: return df
    df_copy = df.copy()
    try:
        df_copy["RSI"] = ta.momentum.RSIIndicator(df_copy["Close"], window=config.rsi_window).rsi()
        macd = ta.trend.MACD(df_copy["Close"], window_fast=config.macd_fast, window_slow=config.macd_slow, window_sign=config.macd_signal)
        df_copy["MACD"], df_copy["MACD_Signal"] = macd.macd(), macd.macd_signal()
        bb = ta.volatility.BollingerBands(df_copy["Close"], window=config.bb_window)
        df_copy["BB_High"], df_copy["BB_Low"] = bb.bollinger_hband(), bb.bollinger_lband()
        return df_copy.dropna()
    except Exception as e:
        return pd.DataFrame()

def get_news_search_results(ticker: str) -> List[Dict[str, str]]:
    """
    Gets top news search results for a ticker using Tavily.
    """
    print(f"📰 Searching for news articles for {ticker}...")
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key: return []
    try:
        client = TavilyClient(api_key=api_key)
        query = f"What are the most significant recent news, events, or announcements for {ticker} stock?"
        response = client.search(query=query, search_depth="basic")
        return [{"title": item.get("title"), "url": item.get("url")} for item in response.get("results", [])[:3]]
    except Exception:
        return []

def scrape_and_summarize_article(url: str) -> str:
    """
    Scrapes and summarizes a single news article.
    """
    print(f"    - Scraping and summarizing: {url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200: return ""
        soup = BeautifulSoup(response.content, 'html.parser')
        main_content = soup.find('article') or soup.find('main') or soup.body
        if not main_content: return ""
        paragraphs = main_content.find_all('p')
        article_text = ' '.join([p.get_text() for p in paragraphs])
        if not article_text.strip(): return ""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = f"Please summarize the following article text in 2-3 sentences:\n\n{article_text[:4000]}"
        summary_response = llm.invoke([HumanMessage(content=prompt)])
        return summary_response.content.strip()
    except Exception:
        return ""

def get_analyst_ratings(ticker: str) -> str:
    """
    Searches for and summarizes analyst ratings and price targets for a stock.
    """
    print(f"🧐 Searching for analyst ratings for {ticker}...")
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key: return "Could not find TAVILY_API_KEY."
    try:
        client = TavilyClient(api_key=api_key)
        query = f"What are the latest analyst ratings, price targets, and consensus for {ticker} stock?"
        response = client.search(query=query, search_depth="advanced")
        summary = " ".join([item['content'] for item in response.get("results", [])])
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = f"Please summarize the following analyst commentary into a concise paragraph, highlighting the consensus rating (e.g., Buy, Hold, Sell) and the average price target:\n\n{summary}"
        summary_response = llm.invoke([HumanMessage(content=prompt)])
        return summary_response.content.strip()
    except Exception as e:
        return f"Error finding analyst ratings: {e}"

def get_financial_metrics(ticker: str) -> Dict[str, Any]:
    """
    Fetches key financial valuation metrics for a stock using yfinance.
    """
    print(f"💰 Fetching financial metrics for {ticker}...")
    try:
        stock_info = yf.Ticker(ticker).info
        metrics = {
            "Market Cap": stock_info.get("marketCap"),
            "Forward P/E": stock_info.get("forwardPE"),
            "PEG Ratio": stock_info.get("pegRatio"),
            "Price/Sales": stock_info.get("priceToSalesTrailing12Months"),
            "Recommendation": stock_info.get("recommendationKey", "N/A").upper()
        }
        return metrics
    except Exception as e:
        return {"Error": f"Could not fetch financial metrics: {e}"}

def get_llm_analysis(state: Dict) -> str:
    """
    Generates the final comprehensive report using all gathered data.
    """
    print("🤖 Generating final comprehensive analysis with LLM...")
    
    # Unpack all the data from the state
    ticker = state['ticker']
    df = state['df']
    news = state['news']
    analyst_ratings = state['analyst_ratings']
    financial_metrics = state['financial_metrics']

    if df.empty:
        return "Cannot perform LLM analysis: No technical data available."

    # Format all the data for the final prompt
    formatted_news = "\n".join([f"- **{item['title']}** ([Source]({item['url']}))\n  - *Summary*: {item['summary']}" for item in news]) or "No recent news could be scraped or summarized."
    formatted_metrics = "\n".join([f"- **{key}**: {value if value is not None else 'N/A'}" for key, value in financial_metrics.items()])

    try:
        latest = df.iloc[-1]
        prompt = textwrap.dedent(f"""
            You are a Senior Financial Analyst. Your task is to provide a comprehensive, data-driven investment report for {ticker}. The report must be objective and synthesize all available information.

            # **Comprehensive Investment Report: {ticker}**

            ## **1. Executive Summary**
            Provide a concise, high-level overview of the key findings from all four analysis pillars (Technical, Valuation, News, Analyst Consensus). Conclude with a clear, actionable recommendation and your confidence level.

            ---

            ## **2. Valuation Analysis**
            Analyze the company's valuation based on the following metrics. Is the stock currently overvalued, undervalued, or fairly valued compared to its peers and growth prospects?
            {formatted_metrics}

            ---

            ## **3. Technical Analysis**
            Provide a detailed interpretation of the latest technical indicators.
            - **Closing Price**: ${latest['Close']:.2f}
            - **RSI**: {latest['RSI']:.2f}
            - **MACD**: {latest['MACD']:.4f} (Signal: {latest['MACD_Signal']:.4f})
            - **Bollinger Bands**: Upper ${latest['BB_High']:.2f}, Lower ${latest['BB_Low']:.2f}

            ---

            ## **4. Analyst Consensus & News Sentiment**
            
            ### **Analyst Ratings:**
            Summarize the current Wall Street analyst consensus and average price target.
            {analyst_ratings}

            ### **Recent News Summaries:**
            Analyze the sentiment (Positive, Negative, Neutral) of the following news and its potential market impact.
            {formatted_news}

            ---

            ## **5. Synthesized Recommendation & Final Verdict**
            Synthesize all four pillars—Valuation, Technicals, Analyst Consensus, and News Sentiment—to form a final investment thesis. It is crucial to take a decisive stance.

            - **Recommendation**: State clearly: **BUY**, **HOLD**, or **SELL**.
            - **Confidence Level**: Rate your confidence in this recommendation (High, Medium, Low).
            - **Justification**: Provide a clear, evidence-based rationale. Explain how you weighed the different signals (e.g., "While the news sentiment is positive, the extremely high valuation metrics and bearish analyst consensus suggest a high risk of a downward correction, leading to a 'SELL' recommendation.").
            - **Risk Factors**: Briefly mention key risks that could invalidate this analysis.
        """)
        
        llm = ChatOpenAI(model="gpt-4o-mini")
        response = llm.invoke([HumanMessage(content=prompt)])
        print("✅ LLM analysis completed")
        return response.content
    except Exception as e:
        return f"❌ Error in LLM analysis: {e}"
