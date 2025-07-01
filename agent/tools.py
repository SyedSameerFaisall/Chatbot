import os
import pandas as pd
import yfinance as yf
import ta
import textwrap
import requests
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from typing import Sequence, List, Dict
from tavily import TavilyClient

# Import the config class to use for type hinting
from .config import AnalysisConfig

def fetch_stock_data(ticker: str, config: AnalysisConfig) -> pd.DataFrame:
    """
    Fetches historical OHLCV stock data using parameters from the config object.
    """
    print(f"📈 Fetching stock data for {ticker} (Period: {config.period}, Interval: {config.interval})...")
    try:
        df = yf.Ticker(ticker).history(period=config.period, interval=config.interval).reset_index()
        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        print(f"✅ Retrieved {len(df)} days of data")
        return df
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return pd.DataFrame()

def compute_technical_indicators(df: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """
    Computes technical indicators using window parameters from the config object.
    """
    print("🔍 Computing technical indicators...")
    if df.empty:
        print("⚠️ Cannot compute indicators: DataFrame is empty.")
        return df
        
    df_copy = df.copy()
    try:
        df_copy["RSI"] = ta.momentum.RSIIndicator(df_copy["Close"], window=config.rsi_window).rsi()
        macd = ta.trend.MACD(df_copy["Close"], window_fast=config.macd_fast, window_slow=config.macd_slow, window_sign=config.macd_signal)
        df_copy["MACD"] = macd.macd()
        df_copy["MACD_Signal"] = macd.macd_signal()
        bb = ta.volatility.BollingerBands(df_copy["Close"], window=config.bb_window)
        df_copy["BB_High"] = bb.bollinger_hband()
        df_copy["BB_Low"] = bb.bollinger_lband()
        df_clean = df_copy.dropna()
        print(f"✅ Indicators computed. {len(df_clean)} complete data points remaining.")
        return df_clean
    except Exception as e:
        print(f"❌ Error computing indicators: {e}")
        return pd.DataFrame()

def get_news_search_results(ticker: str) -> List[Dict[str, str]]:
    """
    Gets top news search results for a ticker using Tavily, returning titles and URLs.
    """
    print(f"📰 Searching for impactful news for {ticker} with Tavily...")
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        print("❌ TAVILY_API_KEY not found.")
        return []

    try:
        client = TavilyClient(api_key=api_key)
        # Using a more specific query to find impactful news articles.
        query = f"What are the most significant recent news, events, or announcements for {ticker} stock that can impact its stock price?"
        response = client.search(query=query, search_depth="basic")
        
        results = response.get("results", [])
        # Return a list of dicts with title and URL, limited to the top 3 for efficiency
        return [{"title": item.get("title"), "url": item.get("url")} for item in results[:3]]
    except Exception as e:
        print(f"❌ An unexpected error occurred while searching for news with Tavily: {e}")
        return []

def scrape_and_summarize_article(url: str) -> str:
    """
    Scrapes the text content of a given URL and uses an LLM to summarize it.
    Returns an empty string if scraping or summarization fails.
    """
    print(f"    - Scraping and summarizing: {url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            print(f"    - Failed to retrieve content, status code: {response.status_code}")
            return ""

        soup = BeautifulSoup(response.content, 'html.parser')
        
        # A more robust way to find the main article content
        main_content = soup.find('article') or soup.find('main') or soup.body
        if not main_content:
            return ""

        paragraphs = main_content.find_all('p')
        article_text = ' '.join([p.get_text() for p in paragraphs])

        if not article_text.strip():
            print("    - No text content found to summarize.")
            return ""
            
        # Use an LLM to summarize the extracted text
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = f"Please summarize the following article text in 2-3 sentences:\n\n{article_text[:4000]}"
        summary_response = llm.invoke([HumanMessage(content=prompt)])
        
        return summary_response.content.strip()
    except Exception as e:
        print(f"    - Error during scraping or summarization: {e}")
        return ""

def get_llm_analysis(ticker: str, df: pd.DataFrame, messages: Sequence[BaseMessage], news: List[Dict[str, str]]) -> str:
    """
    Generates the final comprehensive report using all gathered data.
    """
    print("🤖 Generating final comprehensive analysis with LLM...")
    if df.empty:
        return "Cannot perform LLM analysis: No data available."

    # Format the summarized news for the prompt, now including the URL
    formatted_news = "\n".join([f"- **{item['title']}** ([Source]({item['url']}))\n  - *Summary*: {item['summary']}" for item in news])
    if not formatted_news:
        formatted_news = "No recent news could be scraped or summarized."

    try:
        latest = df.iloc[-1]
        prompt = textwrap.dedent(f"""
            You are a Senior Financial Analyst providing a detailed report on {ticker}. Your analysis must be objective, data-driven, and professionally articulated. You must weigh the evidence from both technicals and news to form a decisive conclusion.

            # **Comprehensive Analysis Report: {ticker}**

            ## **1. Executive Summary**
            Provide a concise, high-level overview of the key findings from the technical and news analysis. Conclude with a clear, actionable recommendation and your confidence level in that recommendation.

            ---

            ## **2. Technical Analysis**
            Based on the latest data point, provide a detailed interpretation of the following indicators:
            - **Closing Price**: ${latest['Close']:.2f}
            - **RSI**: {latest['RSI']:.2f}
            - **MACD**: {latest['MACD']:.4f} (Signal: {latest['MACD_Signal']:.4f})
            - **Bollinger Bands**: Upper ${latest['BB_High']:.2f}, Lower ${latest['BB_Low']:.2f}

            ### **Interpretation:**
            - **Trend & Momentum**: Is the MACD indicating bullish or bearish momentum? Is the trend accelerating or decelerating?
            - **Overbought/Oversold Conditions**: What does the RSI value suggest about the current market sentiment? Is a price correction likely?
            - **Volatility**: How is the price positioned relative to the Bollinger Bands? Does this suggest high or low volatility?

            ---

            ## **3. News & Market Sentiment Analysis**
            The following are summaries of recent news articles. Analyze the overall sentiment (Positive, Negative, Neutral) and explain its potential impact on the stock price.

            ### **Recent News Summaries:**
            {formatted_news}

            ### **Sentiment Impact:**
            Discuss how these developments could influence investor perception and market behavior in the short to medium term.

            ---

            ## **4. Synthesized Recommendation**
            Synthesize the findings from **both the technical and news analysis** to provide a final investment recommendation. It is crucial to take a stance; avoid defaulting to 'HOLD' unless the evidence is perfectly balanced. Weigh the strength of each indicator and news event.

            - **Recommendation**: State clearly: **BUY**, **HOLD**, or **SELL**.
            - **Confidence Level**: Rate your confidence in this recommendation (e.g., High, Medium, Low).
            - **Justification**: Provide a clear, evidence-based rationale for your recommendation. If signals are mixed, explain which factors you are prioritizing and why (e.g., "Despite the bullish MACD, the extremely overbought RSI and negative news sentiment suggest a higher probability of a near-term pullback, leading to a SELL recommendation.").
            - **Risk Factors**: Briefly mention key risks that could invalidate this analysis (e.g., market downturn, negative earnings report, unexpected company news).
        """)
        full_chat = list(messages) + [HumanMessage(content=prompt)]
        llm = ChatOpenAI(model="gpt-4o-mini")
        response = llm.invoke(full_chat)
        print("✅ LLM analysis completed")
        return response.content
    except Exception as e:
        print(f"❌ Error in LLM analysis: {e}")
        return f"Error in LLM analysis: {e}"