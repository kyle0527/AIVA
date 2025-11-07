"""
AIVA IDOR Function Module
========================

Enhanced IDOR detection capabilities with horizontal and vertical testing.
Integrated with AIVA five-module architecture.
"""

from .detector.idor_detector import IDORDetector
from .engine.idor_engine import IDOREngine, IDORIssue
from .config.idor_config import IdorConfig

__all__ = ["IDORDetector", "IDOREngine", "IDORIssue", "IdorConfig"]