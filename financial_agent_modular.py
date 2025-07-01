from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
import pandas as pd

# Import the state and config definitions, and the nodes
from agent.config import AnalysisConfig
from agent.state import AgentState
from agent.nodes import (
    start_analysis,
    fetch_data_node,
    fetch_news_node,
    fetch_ratings_node,
    fetch_financials_node,
    synthesize_and_stream_report_node
)

# Load environment variables from .env file
load_dotenv()

class StockAnalysisAgent:
    """
    A class to encapsulate the stock analysis agent, including its graph and execution logic.
    """
    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Builds the LangGraph workflow."""
        builder = StateGraph(AgentState)

        # Add nodes to the graph
        builder.add_node("start_analysis", start_analysis)
        builder.add_node("fetch_data_and_technicals", fetch_data_node)
        builder.add_node("fetch_news", fetch_news_node)
        builder.add_node("fetch_ratings", fetch_ratings_node)
        builder.add_node("fetch_financials", fetch_financials_node)
        builder.add_node("synthesize_and_stream_report", synthesize_and_stream_report_node)

        # Define the sequential workflow
        builder.set_entry_point("start_analysis")
        builder.add_edge("start_analysis", "fetch_data_and_technicals")
        builder.add_edge("fetch_data_and_technicals", "fetch_news")
        builder.add_edge("fetch_news", "fetch_ratings")
        builder.add_edge("fetch_ratings", "fetch_financials")
        builder.add_edge("fetch_financials", "synthesize_and_stream_report")
        builder.add_edge("synthesize_and_stream_report", END)

        return builder.compile()

    def run_analysis(self, ticker: str, config: AnalysisConfig) -> str:
        """
        Runs the stock analysis for a given ticker and returns the final report as a string.
        """
        print("🚀 Starting Stock Analysis Agent...")
        
        initial_state = AgentState(
            messages=[],
            ticker=ticker,
            df=pd.DataFrame(),
            config=config,
            news=[],
            analyst_ratings="",
            financial_metrics={},
            final_report=""
        )

        # Let the graph run to completion
        final_state = self.graph.invoke(initial_state)

        # After the graph has finished, get the final report from the state
        final_report = final_state.get('final_report', "Error: No final report was generated.")
        
        print(f"\n✅ Analysis complete!")
        # Return the final report instead of writing to a file
        return final_report