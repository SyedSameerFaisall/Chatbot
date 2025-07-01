from typing import Annotated, Sequence, TypedDict, List, Dict, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import pandas as pd

# Import the configuration class
from .config import AnalysisConfig

class AgentState(TypedDict):
    """
    Represents the state of our financial analysis agent.
    
    Attributes:
        messages: The history of messages in the conversation.
        ticker: The stock ticker symbol being analyzed.
        df: The pandas DataFrame holding the stock data and indicators.
        config: A configuration object with analysis parameters.
        news: A list of dictionaries containing summarized news articles.
        analyst_ratings: A summary of analyst ratings and price targets.
        financial_metrics: Key valuation metrics for the stock.
        final_report: A string to hold the complete report after streaming.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    ticker: str
    df: pd.DataFrame
    config: AnalysisConfig
    news: List[Dict[str, str]]
    analyst_ratings: str
    financial_metrics: Dict[str, Any]
    final_report: str
