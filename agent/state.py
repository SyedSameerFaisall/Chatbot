from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import pandas as pd

# Import the new configuration class
from .config import AnalysisConfig

class AgentState(TypedDict):
    """
    Represents the state of our financial analysis agent.
    
    Attributes:
        messages: The history of messages in the conversation.
        ticker: The stock ticker symbol being analyzed.
        df: The pandas DataFrame holding the stock data and indicators.
        config: A configuration object with analysis parameters.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    ticker: str
    df: pd.DataFrame
    config: AnalysisConfig
