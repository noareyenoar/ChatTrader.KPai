# ChatTrader.KPai — Phase 5 Multi-Agent System
# agents/__init__.py

from agents.base_agent import BaseAgent, EvidencePacket
from agents.analyst_agents import (
    TrendAnalyst,
    MeanReversionAnalyst,
    ScalperAnalyst,
    StatArbAnalyst,
    DiscretionaryAnalyst,
    MarketMakerAnalyst,
)
from agents.shadow_agent import ShadowAgent
from agents.portfolio_manager import PortfolioManager

__all__ = [
    "BaseAgent",
    "EvidencePacket",
    "TrendAnalyst",
    "MeanReversionAnalyst",
    "ScalperAnalyst",
    "StatArbAnalyst",
    "DiscretionaryAnalyst",
    "MarketMakerAnalyst",
    "ShadowAgent",
    "PortfolioManager",
]
