import pandas as pd
import yfinance as yf
import ta
import textwrap
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from typing import Sequence

def fetch_stock_data(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """
    Fetches historical OHLCV stock data for the given ticker from Yahoo Finance.
    """
    print(f"📈 Fetching stock data for {ticker}...")
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval).reset_index()
        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        print(f"✅ Retrieved {len(df)} days of data")
        return df
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return pd.DataFrame()

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes and adds key technical indicators to the stock DataFrame.
    """
    print("🔍 Computing technical indicators...")
    if df.empty:
        print("⚠️ Cannot compute indicators: DataFrame is empty.")
        return df
        
    df_copy = df.copy()
    try:
        df_copy["RSI"] = ta.momentum.RSIIndicator(df_copy["Close"]).rsi()
        macd = ta.trend.MACD(df_copy["Close"])
        df_copy["MACD"] = macd.macd()
        df_copy["MACD_Signal"] = macd.macd_signal()
        bb = ta.volatility.BollingerBands(df_copy["Close"])
        df_copy["BB_High"] = bb.bollinger_hband()
        df_copy["BB_Low"] = bb.bollinger_lband()
        df_clean = df_copy.dropna()
        print(f"✅ Indicators computed. {len(df_clean)} complete data points remaining.")
        return df_clean
    except Exception as e:
        print(f"❌ Error computing indicators: {e}")
        return pd.DataFrame()

def get_llm_analysis(ticker: str, df: pd.DataFrame, messages: Sequence[BaseMessage]) -> str:
    """
    Uses an LLM to analyze the stock's technical indicators and generate insights.
    """
    print("🤖 Analyzing with LLM...")
    if df.empty:
        return "Cannot perform LLM analysis: No data available."

    try:
        latest = df.iloc[-1]
        prompt = textwrap.dedent(f"""
            You are a professional financial analyst. Analyze the stock "{ticker}" based on the following technical indicators:

            - 📉 **Close Price**: ${latest['Close']:.2f}
            - 📊 **RSI (14)**: {latest['RSI']:.2f}
            - 📈 **MACD**: {latest['MACD']:.4f}
            - 📈 **MACD Signal**: {latest['MACD_Signal']:.4f}
            - 🎯 **Bollinger High**: ${latest['BB_High']:.2f}
            - 🎯 **Bollinger Low**: ${latest['BB_Low']:.2f}

            Write a clear, structured very detailed **Markdown report** under the following headings:

            # 📈 Technical Analysis Report for {ticker}
            ## 1. Price Trend
            ## 2. RSI Analysis
            ## 3. MACD Analysis
            ## 4. Bollinger Bands
            ## 5. 🧠 Insight & Recommendation
            Include any other observations if found and make everything detailed.
        """)
        full_chat = list(messages) + [HumanMessage(content=prompt)]
        llm = ChatOpenAI(model="gpt-4o-mini")
        response = llm.invoke(full_chat)
        print("✅ LLM analysis completed")
        return response.content
    except Exception as e:
        print(f"❌ Error in LLM analysis: {e}")
        return f"Error in LLM analysis: {e}"