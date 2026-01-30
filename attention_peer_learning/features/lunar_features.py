"""
LunarLander-v2 Feature Extractor
Implements 35-dimensional feature extraction as per paper Section 3.3.1

Feature Breakdown:
- Position Features (6): x, y, distance_to_center, horizontal_danger, altitude_critical, in_landing_zone
- Velocity Features (7): vx, vy, speed, dangerous_descent, lateral_drift, velocity_alignment, controlled_speed
- Orientation Features (5): angle, angular_velocity, unstable_angle, rotation_danger, upright_bonus
- Landing Features (4): left_leg_contact, right_leg_contact, both_legs_contact, ground_clearance
- Action Context (6): action_onehot (4) + action_change + action_consistency
- Peer Context (4): peer_onehot (4)
- Reward Context (3): current_reward, reward_trend, cumulative_performance
"""

import numpy as np
from collections import deque
from typing import Dict, Tuple


class LunarLanderFeatureExtractor:
    """Extract 35-dimensional features from LunarLander environment"""
    
    def __init__(self, n_peers: int = 4):
        """
        Initialize feature extractor
        
        Args:
            n_peers: Number of peers in the group
        """
        self.n_peers = n_peers
        self.landing_zone_x = 0.0  # Landing pad center
        
        # Tracking histories
        self.peer_action_history: Dict[int, deque] = {i: deque(maxlen=50) for i in range(n_peers)}
        self.reward_history: deque = deque(maxlen=20)
        self.last_action: int = 0
        
    def extract_features(
        self, 
        obs: np.ndarray, 
        action: int, 
        peer_id: int, 
        reward: float, 
        next_obs: np.ndarray = None
    ) -> np.ndarray:
        """
        Extract 35-dimensional feature vector
        
        Args:
            obs: LunarLander observation (8-dim: x, y, vx, vy, angle, angular_vel, left_leg, right_leg)
            action: Discrete action (0-3: noop, left_engine, main_engine, right_engine)
            peer_id: ID of suggesting peer (0 to n_peers-1)
            reward: Current step reward
            next_obs: Next observation (optional, for reward trend)
            
        Returns:
            Feature vector of shape (35,)
        """
        if obs is None or len(obs) < 8:
            return self._get_default_features()
        
        # Parse observation
        x, y, vx, vy, angle, angular_vel, left_leg, right_leg = obs[:8]
        
        # Extract feature components
        position_features = self._extract_position_features(x, y)               # 6-dim
        velocity_features = self._extract_velocity_features(vx, vy, y)          # 7-dim
        orientation_features = self._extract_orientation_features(angle, angular_vel)  # 5-dim
        landing_features = self._extract_landing_features(left_leg, right_leg, y)      # 4-dim
        action_features = self._extract_action_features(action)                 # 6-dim
        peer_features = self._extract_peer_features(peer_id)                    # 4-dim
        reward_features = self._extract_reward_features(reward)                 # 3-dim
        
        # Update tracking
        self.peer_action_history[peer_id].append(action)
        self.reward_history.append(reward)
        self.last_action = action
        
        # Concatenate all features
        features = np.concatenate([
            position_features,     # 6
            velocity_features,     # 7
            orientation_features,  # 5
            landing_features,      # 4
            action_features,       # 6
            peer_features,         # 4
            reward_features        # 3
        ])
        
        return features.astype(np.float32)
    
    def _extract_position_features(self, x: float, y: float) -> np.ndarray:
        """Position analysis (6 dimensions)"""
        distance_to_center = abs(x - self.landing_zone_x)
        horizontal_danger = 1.0 if abs(x) > 0.4 else 0.0
        altitude_critical = 1.0 if y < 0.1 else 0.0
        in_landing_zone = 1.0 if abs(x) < 0.2 else 0.0
        
        return np.array([x, y, distance_to_center, horizontal_danger, altitude_critical, in_landing_zone])
    
    def _extract_velocity_features(self, vx: float, vy: float, y: float) -> np.ndarray:
        """Velocity analysis (7 dimensions)"""
        speed = np.sqrt(vx**2 + vy**2)
        dangerous_descent = 1.0 if (vy < -0.5 and y < 0.3) else 0.0
        lateral_drift = 1.0 if abs(vx) > 0.3 else 0.0
        
        # Velocity alignment: moving towards center
        velocity_alignment = -vx * np.sign(self.landing_zone_x - 0.0) if abs(vx) > 0.01 else 0.0
        controlled_speed = 1.0 if speed < 0.5 else 0.0
        
        return np.array([vx, vy, speed, dangerous_descent, lateral_drift, velocity_alignment, controlled_speed])
    
    def _extract_orientation_features(self, angle: float, angular_vel: float) -> np.ndarray:
        """Orientation analysis (5 dimensions)"""
        unstable_angle = 1.0 if abs(angle) > 0.4 else 0.0
        rotation_danger = 1.0 if abs(angular_vel) > 0.5 else 0.0
        upright_bonus = 1.0 if abs(angle) < 0.1 else 0.0
        
        return np.array([angle, angular_vel, unstable_angle, rotation_danger, upright_bonus])
    
    def _extract_landing_features(self, left_leg: float, right_leg: float, y: float) -> np.ndarray:
        """Landing status (4 dimensions)"""
        both_legs_contact = 1.0 if (left_leg > 0.5 and right_leg > 0.5) else 0.0
        ground_clearance = max(0.0, y - 0.05)  # Height above ground
        
        return np.array([left_leg, right_leg, both_legs_contact, ground_clearance])
    
    def _extract_action_features(self, action: int) -> np.ndarray:
        """Action context (6 dimensions)"""
        # One-hot encoding of action (4 actions)
        action_onehot = np.zeros(4)
        action_onehot[action] = 1.0
        
        # Action change indicator
        action_change = 1.0 if action != self.last_action else 0.0
        
        # Action consistency (how often peer repeats same action)
        action_consistency = 0.0
        if len(self.peer_action_history.get(0, [])) > 5:
            recent_actions = list(self.peer_action_history[0])[-5:]
            action_consistency = recent_actions.count(action) / 5.0
        
        return np.array([*action_onehot, action_change, action_consistency])
    
    def _extract_peer_features(self, peer_id: int) -> np.ndarray:
        """Peer context (4 dimensions)"""
        # One-hot encoding of peer ID (assuming max 4 peers)
        peer_onehot = np.zeros(4)
        if peer_id < 4:
            peer_onehot[peer_id] = 1.0
        
        return peer_onehot
    
    def _extract_reward_features(self, reward: float) -> np.ndarray:
        """Reward context (3 dimensions)"""
        # Current reward
        current_reward = reward
        
        # Reward trend (positive/negative change)
        reward_trend = 0.0
        if len(self.reward_history) > 0:
            reward_trend = reward - self.reward_history[-1]
        
        # Cumulative performance estimate
        cumulative_performance = np.mean(self.reward_history) if len(self.reward_history) > 0 else 0.0
        
        return np.array([current_reward, reward_trend, cumulative_performance])
    
    def _get_default_features(self) -> np.ndarray:
        """Return default zero features if extraction fails"""
        return np.zeros(35, dtype=np.float32)
    
    def reset(self):
        """Reset tracking histories"""
        self.peer_action_history = {i: deque(maxlen=50) for i in range(self.n_peers)}
        self.reward_history = deque(maxlen=20)
        self.last_action = 0
