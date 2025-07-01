import os
import pandas as pd
import yfinance as yf
import ta
import textwrap
import requests
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from typing import Sequence, List, Dict, Any, Iterator
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
        print(f"❌ Error fetching stock data: {e}")
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
        print(f"❌ Error computing indicators: {e}")
        return pd.DataFrame()

def get_news_search_results(ticker: str) -> List[Dict[str, str]]:
    """
    Gets top news search results for a ticker using Tavily's advanced search.
    """
    print(f"📰 Searching for impactful news for {ticker}...")
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key: 
        print("❌ TAVILY_API_KEY not found.")
        return []
    try:
        client = TavilyClient(api_key=api_key)
        query = f"What are the most significant recent news, events, or announcements for {ticker} stock?"
        response = client.search(query=query, search_depth="advanced", max_results=3)
        return [{"title": item.get("title"), "url": item.get("url")} for item in response.get("results", [])]
    except Exception as e:
        print(f"❌ An error occurred during Tavily news search: {e}")
        return []

def scrape_and_summarize_article(url: str) -> str:
    """
    Scrapes and summarizes a single news article with improved error logging.
    """
    print(f"    - Scraping and summarizing: {url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"    - Failed to retrieve content, status code: {response.status_code}")
            return ""

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
    except Exception as e:
        print(f"    - Error during scraping or summarization for {url}: {e}")
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
        if not summary.strip(): return "No analyst ratings found."
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = f"Please summarize the following analyst commentary into a concise paragraph, highlighting the consensus rating (e.g., Buy, Hold, Sell) and the average price target:\n\n{summary}"
        summary_response = llm.invoke([HumanMessage(content=prompt)])
        return summary_response.content.strip()
    except Exception as e:
        print(f"❌ An error occurred while fetching analyst ratings: {e}")
        return f"Error finding analyst ratings: {e}"

def get_financial_metrics(ticker: str) -> Dict[str, Any]:
    """
    Fetches key financial valuation metrics for a stock using yfinance.
    """
    print(f"💰 Fetching financial metrics for {ticker}...")
    try:
        stock_info = yf.Ticker(ticker).info
        return {
            "Market Cap": stock_info.get("marketCap"),
            "Forward P/E": stock_info.get("forwardPE"),
            "PEG Ratio": stock_info.get("pegRatio"),
            "Price/Sales": stock_info.get("priceToSalesTrailing12Months")
        }
    except Exception as e:
        return {"Error": f"Could not fetch financial metrics: {e}"}


def get_llm_analysis_stream(state: Dict) -> Iterator:
    """
    Prepares the LLM prompt using technical, fundamental, and sentiment data.
    Returns a streamable iterator containing the formatted markdown-based investment report.
    """
    print("🤖 Preparing for LLM analysis stream...")

    ticker, df, news, analyst_ratings, financial_metrics = (
        state['ticker'], state['df'], state['news'],
        state['analyst_ratings'], state['financial_metrics']
    )

    if df.empty:
        def error_stream(): yield "❌ Cannot perform LLM analysis: No technical data available."
        return error_stream()

    # Format recent news headlines and financial metrics
    formatted_news = "\n".join([
        f"- **{item['title']}** ([Source]({item['url']}))\n  - *Summary*: {item['summary']}"
        for item in news
    ]) or "No recent news could be scraped or summarized."

    formatted_metrics = "\n".join([
        f"- **{key}**: {value if value is not None else 'N/A'}"
        for key, value in financial_metrics.items()
    ])

    try:
        latest = df.iloc[-1]

        prompt = textwrap.dedent(f"""
        # 🧠 LLM-Powered Investment Report: **{ticker}**

        You are a **Senior Financial Analyst**. Your objective is to produce a **comprehensive, objective, and data-driven investment report** for the stock `{ticker}`. Base your judgment on the following four analysis pillars:

        1. **Valuation Metrics**
        2. **Technical Indicators**
        3. **Analyst Ratings**
        4. **News Sentiment**

        Avoid bias and ensure that your final decision is clear and well-supported.

        ---

        ## 📌 Executive Summary

        Provide a concise, high-level synthesis of your findings across all four pillars. Conclude with a **clear recommendation** (**BUY**, **HOLD**, or **SELL**) and a **confidence level** (High, Medium, Low).

        ---

        ## 💰 Valuation Analysis

        Analyze valuation using the metrics below. Discuss whether the stock appears:
        - **Overvalued**
        - **Undervalued**
        - **Fairly Valued**

        Do not recommend any action here—just provide a value-based assessment.

        ### 🔢 Financial Metrics:
        {formatted_metrics}

        ---

        ## 📉 Technical Analysis

        Offer an objective interpretation of technical indicators. Avoid any buy/sell recommendations in this section.

        | Indicator        | Value |
        |------------------|--------|
        | Closing Price    | ${latest['Close']:.2f} |
        | RSI (14-day)     | {latest['RSI']:.2f} |
        | MACD             | {latest['MACD']:.4f} (Signal: {latest['MACD_Signal']:.4f}) |
        | Bollinger Bands  | Upper ${latest['BB_High']:.2f}, Lower ${latest['BB_Low']:.2f} |

        Describe momentum, volatility, and trend conditions.

        ---

        ## 📰 Analyst Consensus & News Sentiment

        ### 📊 Analyst Ratings

        Summarize analyst consensus and average price targets.

        > {analyst_ratings}

        ### 🗞 News Summary and Sentiment Analysis

        Assess each headline for tone (Positive, Negative, Neutral) and potential market impact.

        {formatted_news}

        ---

        ## ✅ Final Verdict & Investment Thesis

        Integrate all pillars into a **cohesive investment recommendation**.

        - **📌 Recommendation**: BUY / HOLD / SELL
        - **🔍 Confidence Level**: High / Medium / Low
        - **🧠 Justification**: Weigh signals across valuation, technicals, news, and analyst sentiment. If there's a conflict, explain what you prioritized and why.
        - **⚠️ Risk Factors**: Briefly mention key risks that could invalidate your thesis.

        Be decisive, nuanced, and evidence-driven.
        """)

        llm = ChatOpenAI(model="gpt-4o-mini")
        return llm.stream([HumanMessage(content=prompt)])

    except Exception as e:
        def error_stream(): yield f"❌ Error in LLM analysis: {e}"
        return error_stream()

        llm = ChatOpenAI(model="gpt-4o-mini")
        return llm.stream([HumanMessage(content=prompt)])
    except Exception as e:
        def error_stream(): yield f"❌ Error in LLM analysis: {e}"
        return error_stream()
