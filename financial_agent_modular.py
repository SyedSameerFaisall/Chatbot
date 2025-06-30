from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
import pandas as pd

# Import the state definition and the nodes
from agent.state import AgentState
from agent.nodes import (
    start_analysis,
    fetch_data_node,
    analyze_indicators_node,
    analyze_with_llm_node,
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
        builder.add_node("fetch_data", fetch_data_node)
        builder.add_node("analyze_indicators", analyze_indicators_node)
        builder.add_node("analyze_with_llm", analyze_with_llm_node)
        builder.add_node("final_response", final_response_node)

        # Define the workflow edges, connecting the nodes in sequence
        builder.set_entry_point("start_analysis")
        builder.add_edge("start_analysis", "fetch_data")
        builder.add_edge("fetch_data", "analyze_indicators")
        builder.add_edge("analyze_indicators", "analyze_with_llm")
        builder.add_edge("analyze_with_llm", "final_response")
        builder.add_edge("final_response", END)

        return builder.compile()

    def run_analysis(self, ticker: str, output_path: str = "stock_analysis_output.txt"):
        """
        Runs the stock analysis for a given ticker and saves the final output.
        """
        print("🚀 Starting Stock Analysis Agent...")
        
        initial_state = AgentState(
            messages=[],
            ticker=ticker,
            df=pd.DataFrame()
        )

        # Let the graph run to completion
        final_state = self.graph.invoke(initial_state)

        # After the graph has finished, get the final report
        final_report = final_state['messages'][-1].content

        # Save the final, clean report to the output file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_report)
        
        print(f"\n✅ Analysis complete! Full report saved to {output_path}")

def main():
    """Main function to run the agent."""
    ticker = input("Enter the ticker symbol (e.g., TSLA, MSFT): ").strip().upper()
    if not ticker:
        print("❌ No ticker provided. Exiting.")
        return
        
    agent = StockAnalysisAgent()
    agent.run_analysis(ticker)

if __name__ == "__main__":
    main()
