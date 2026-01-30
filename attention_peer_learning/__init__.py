"""
Attention-Enhanced Peer Learning Framework

Clean implementation based on the paper:
"Attention-Enhanced Peer Learning for Multi-Agent Reinforcement Learning"

This package extends the original Peer Learning framework with transformer-based
attention mechanisms for dynamic trust evaluation.
"""

__version__ = "1.0.0"
__author__ = "Research Team"

from .configs.env_config import EnvConfig, get_env_config
from .core.attention_peer import AttentionSACPeer, AttentionDQNPeer

__all__ = [
    'EnvConfig',
    'get_env_config',
    'AttentionSACPeer',
    'AttentionDQNPeer',
]
