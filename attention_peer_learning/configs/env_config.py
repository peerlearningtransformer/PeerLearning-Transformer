"""
Environment Configuration System

Paper Reference: Section 3.3.1 - Environment-Adaptive Feature Processing
Defines environment-specific parameters for feature extraction and transformer architecture.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EnvConfig:
    """
    Configuration for environment-specific attention peer learning.
    
    Attributes:
        name: Environment name (e.g., 'LunarLander-v3')
        obs_dim: Observation space dimension
        action_dim: Action space dimension
        feature_dim: Extracted feature dimension for transformer input
        phases: List of behavioral phases for phase-aware attention
        is_continuous: Whether action space is continuous
        algorithm: RL algorithm to use ('SAC' or 'DQN')
    """
    name: str
    obs_dim: int
    action_dim: int
    feature_dim: int
    phases: List[str]
    is_continuous: bool
    algorithm: str
    
    # Transformer architecture parameters
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    max_seq_len: int = 200
    
    # Training parameters
    attention_lr: float = 1e-5
    attention_update_freq: int = 20
    sequence_length: int = 50
    attention_weight: float = 0.3  # ω_t coefficient from paper Eq. 4
    trust_scale: float = 100.0  # τ from paper Eq. 4
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.obs_dim > 0, "Observation dimension must be positive"
        assert self.action_dim > 0, "Action dimension must be positive"
        assert self.feature_dim > 0, "Feature dimension must be positive"
        assert self.d_model % self.nhead == 0, f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})"
        assert self.algorithm in ['SAC', 'DQN'], f"Algorithm must be SAC or DQN, got {self.algorithm}"


# Environment-specific configurations following paper Section 3.3.1
LUNAR_LANDER_CONFIG = EnvConfig(
    name='LunarLander-v3',
    obs_dim=8,
    action_dim=4,
    feature_dim=35,  # Paper: 35-dimensional features for LunarLander
    phases=['cruise', 'approach', 'landing', 'touchdown'],
    is_continuous=False,
    algorithm='DQN',
    trust_scale=150.0,  # Environment-adaptive trust scale
)

HOPPER_CONFIG = EnvConfig(
    name='Hopper-v4',
    obs_dim=11,
    action_dim=3,
    feature_dim=38,  # Paper: 38-dimensional features for Hopper
    phases=['stance', 'swing', 'flight', 'landing'],
    is_continuous=True,
    algorithm='SAC',
    trust_scale=300.0,  # Environment-adaptive trust scale
)

WALKER2D_CONFIG = EnvConfig(
    name='Walker2d-v4',
    obs_dim=17,
    action_dim=6,
    feature_dim=42,  # Paper: 42-dimensional features for Walker2D
    phases=['left_stance', 'right_stance', 'double_support', 'flight'],
    is_continuous=True,
    algorithm='SAC',
    trust_scale=300.0,  # Environment-adaptive trust scale
)

ANT_CONFIG = EnvConfig(
    name='Ant-v4',
    obs_dim=27,
    action_dim=8,
    feature_dim=52,  # 52-dimensional features for quadrupedal Ant locomotion
    phases=['trot', 'pace', 'gallop', 'stand'],
    is_continuous=True,
    algorithm='SAC',
    trust_scale=300.0,  # Environment-adaptive trust scale
)

# Room environment configurations
ROOM_CONFIG_BASE = EnvConfig(
    name='Room-v*',
    obs_dim=4,  # Will be adjusted per version (2 for 1D, 4 for 2D)
    action_dim=2,  # Will be adjusted per version (2 for 1D, 4 for 2D)
    feature_dim=22,  # 2 + 2 + 2 + 5 + 4 + 4 + 3 (from RoomFeatureExtractor)
    phases=['exploration', 'navigation', 'approaching', 'goal_reached'],
    is_continuous=False,
    algorithm='DQN',
    trust_scale=100.0,  # Environment-adaptive trust scale
)

ROOM_V3_CONFIG = EnvConfig(
    name='Room-v3',
    obs_dim=4,
    action_dim=4,
    feature_dim=22,
    phases=['exploration', 'navigation', 'approaching', 'goal_reached'],
    is_continuous=False,
    algorithm='DQN',
    trust_scale=100.0,
)

ROOM_V6_CONFIG = EnvConfig(
    name='Room-v6',
    obs_dim=4,
    action_dim=4,
    feature_dim=22,
    phases=['exploration', 'navigation', 'approaching', 'goal_reached'],
    is_continuous=False,
    algorithm='DQN',
    trust_scale=120.0,
)

ROOM_V21_CONFIG = EnvConfig(
    name='Room-v21',
    obs_dim=4,
    action_dim=4,
    feature_dim=22,
    phases=['exploration', 'navigation', 'approaching', 'goal_reached'],
    is_continuous=False,
    algorithm='DQN',
    trust_scale=150.0,
)

ROOM_V27_CONFIG = EnvConfig(
    name='Room-v27',
    obs_dim=4,
    action_dim=4,
    feature_dim=22,
    phases=['exploration', 'navigation', 'approaching', 'goal_reached'],
    is_continuous=False,
    algorithm='DQN',
    trust_scale=150.0,
)

ROOM_V33_CONFIG = EnvConfig(
    name='Room-v33',
    obs_dim=4,
    action_dim=4,
    feature_dim=22,
    phases=['exploration', 'navigation', 'approaching', 'goal_reached'],
    is_continuous=False,
    algorithm='DQN',
    trust_scale=200.0,
)


def get_env_config(env_name: str) -> EnvConfig:
    """
    Get environment configuration by name.
    
    Args:
        env_name: Environment name (e.g., 'LunarLander-v3', 'Hopper-v4', 'Walker2d-v4', 'Ant-v4', 'Room-v*')
        
    Returns:
        EnvConfig: Environment configuration
        
    Raises:
        ValueError: If environment is not supported
    """
    env_name_lower = env_name.lower()
    
    if 'lunar' in env_name_lower:
        return LUNAR_LANDER_CONFIG
    elif 'hopper' in env_name_lower:
        return HOPPER_CONFIG
    elif 'walker2d' in env_name_lower or 'walker' in env_name_lower:
        return WALKER2D_CONFIG
    elif 'ant' in env_name_lower:
        return ANT_CONFIG
    elif 'room' in env_name_lower:
        # Route to specific room config based on version (check larger versions first)
        if 'room-v33' in env_name_lower or 'room_v33' in env_name_lower:
            return ROOM_V33_CONFIG
        elif 'room-v27' in env_name_lower or 'room_v27' in env_name_lower:
            return ROOM_V27_CONFIG
        elif 'room-v21' in env_name_lower or 'room_v21' in env_name_lower:
            return ROOM_V21_CONFIG
        elif 'room-v6' in env_name_lower or 'room_v6' in env_name_lower:
            return ROOM_V6_CONFIG
        elif 'room-v3' in env_name_lower or 'room_v3' in env_name_lower:
            return ROOM_V3_CONFIG
        else:
            # Default to base room config
            return ROOM_CONFIG_BASE
    else:
        raise ValueError(Ant-v4, 
            f"Unsupported environment: {env_name}. "
            f"Supported: LunarLander-v3, Hopper-v4, Walker2d-v4, Room-v3/v6/v21/v27/v33"
        )


def print_env_config(config: EnvConfig) -> None:
    """Print environment configuration in readable format."""
    print(f"\n{'='*60}")
    print(f"Environment Configuration: {config.name}")
    print(f"{'='*60}")
    print(f"  Observation Dimension:  {config.obs_dim}")
    print(f"  Action Dimension:       {config.action_dim}")
    print(f"  Feature Dimension:      {config.feature_dim}")
    print(f"  Behavioral Phases:      {', '.join(config.phases)}")
    print(f"  Action Space:           {'Continuous' if config.is_continuous else 'Discrete'}")
    print(f"  Algorithm:              {config.algorithm}")
    print(f"\n  Transformer Architecture:")
    print(f"    Model Dimension:      {config.d_model}")
    print(f"    Attention Heads:      {config.nhead}")
    print(f"    Encoder Layers:       {config.num_layers}")
    print(f"    Max Sequence Length:  {config.max_seq_len}")
    print(f"\n  Training Parameters:")
    print(f"    Attention LR:         {config.attention_lr}")
    print(f"    Update Frequency:     {config.attention_update_freq}")
    print(f"    Sequence Length:      {config.sequence_length}")
    print(f"    Attention Weight (ω): {config.attention_weight}")
    print(f"    Trust Scale (τ):      {config.trust_scale}")
    print(f"{'='*60}\n")
