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
    synthesize_report_node,
    final_response_node
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
        builder.add_node("synthesize_report", synthesize_report_node)
        builder.add_node("final_response", final_response_node)

        # Define the workflow edges sequentially to avoid state conflicts
        builder.set_entry_point("start_analysis")
        builder.add_edge("start_analysis", "fetch_data_and_technicals")
        builder.add_edge("fetch_data_and_technicals", "fetch_news")
        builder.add_edge("fetch_news", "fetch_ratings")
        builder.add_edge("fetch_ratings", "fetch_financials")
        builder.add_edge("fetch_financials", "synthesize_report")
        builder.add_edge("synthesize_report", "final_response")
        builder.add_edge("final_response", END)

        return builder.compile()

    def run_analysis(self, ticker: str, config: AnalysisConfig, output_path: str = "stock_analysis_output.txt"):
        """
        Runs the stock analysis for a given ticker using the provided configuration.
        """
        print("🚀 Starting Stock Analysis Agent...")
        
        initial_state = AgentState(
            messages=[],
            ticker=ticker,
            df=pd.DataFrame(),
            config=config,
            news=[],
            analyst_ratings="",
            financial_metrics={}
        )

        # Let the graph run to completion
        final_state = self.graph.invoke(initial_state)

        # After the graph has finished, get the final report
        final_report = final_state['messages'][-1].content

        # Save the final, clean report to the output file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_report)
        
        print(f"\n✅ Analysis complete! Full report saved to {output_path}")

def get_user_config() -> AnalysisConfig:
    """
    Prompts the user for custom analysis configuration.
    Returns a default config if the user declines.
    """
    custom = input("Do you want to set custom analysis parameters? (yes/no): ").strip().lower()
    
    if custom == 'yes':
        print("Please provide the following parameters:")
        try:
            period = input("Analysis Period (e.g., 6mo, 1y, 2y): ")
            rsi_window = int(input("RSI Window (e.g., 14): "))
            bb_window = int(input("Bollinger Bands Window (e.g., 20): "))
            print("✅ Custom configuration set.")
            return AnalysisConfig(period=period, rsi_window=rsi_window, bb_window=bb_window)
        except ValueError:
            print("❌ Invalid input. Using default configuration.")
            return AnalysisConfig()
    else:
        print("✅ Using default analysis parameters (Period: 6mo, RSI: 14, BB: 20).")
        return AnalysisConfig()

def main():
    """Main function to run the agent."""
    ticker = input("Enter the ticker symbol (e.g., TSLA, MSFT): ").strip().upper()
    if not ticker:
        print("❌ No ticker provided. Exiting.")
        return
        
    # Get configuration from the user
    config = get_user_config()
    
    agent = StockAnalysisAgent()
    agent.run_analysis(ticker, config=config)

if __name__ == "__main__":
    main()
