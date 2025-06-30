from dataclasses import dataclass, field
from typing import List

@dataclass
class AnalysisConfig:
    """
    A dataclass to hold all configuration parameters for the stock analysis.
    This makes the agent's behavior easy to configure and manage.
    """
    period: str = "6mo"
    interval: str = "1d"
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_window: int = 20
