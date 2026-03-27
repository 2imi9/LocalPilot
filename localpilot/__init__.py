"""
LocalPilot — Web-Enhanced Autonomous ML Research
=================================================
Autonomous hyperparameter search grounded in recent arXiv literature.

Quick start:
    from localpilot.config import LocalPilotConfig
    from localpilot.browse import MolmoWebAgent, extract_ideas
    from localpilot.analyze import main as run_analysis
"""

from localpilot.config import LocalPilotConfig, WEB_AGENT_MODELS, CODE_AGENT_MODELS

__all__ = ["LocalPilotConfig", "WEB_AGENT_MODELS", "CODE_AGENT_MODELS"]
__version__ = "0.2.0"
