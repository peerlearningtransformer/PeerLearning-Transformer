"""
Room-v* Feature Extractor
Implements feature extraction for Room navigation environments

Feature Breakdown:
- Position Features (4): agent_position (normalized for each dimension), distance_to_goal
- Goal Features (4): goal_position (normalized for each dimension), goal_distance_ratio
- Movement Features (5): velocity_estimate, movement_efficiency, exploration_coverage, path_efficiency, stuck_indicator
- Action Context (4): action_onehot (max 6 dimensions), action_magnitude
- Peer Context (4): peer_onehot (4)
- Performance Features (3): current_reward, reward_trend, cumulative_performance
"""

import numpy as np
from collections import deque
from typing import Dict, Tuple


class RoomFeatureExtractor:
    """Extract features from Room navigation environment"""
    
    def __init__(self, n_peers: int = 4, max_dims: int = 2):
        """
        Initialize feature extractor
        
        Args:
            n_peers: Number of peers in the group
            max_dims: Maximum number of dimensions (1D or 2D room)
        """
        self.n_peers = n_peers
        self.max_dims = max_dims
        
        # Feature dimension calculation:
        # Position (max_dims) + Goal (max_dims) + Distance metrics (2)
        # + Movement (5) + Action (max_dims + 2) + Peer (4) + Performance (3)
        self.feature_dim = max_dims + max_dims + 2 + 5 + (max_dims + 2) + 4 + 3
        
        # Tracking histories
        self.peer_action_history: Dict[int, deque] = {i: deque(maxlen=50) for i in range(n_peers)}
        self.reward_history: deque = deque(maxlen=20)
        self.position_history: deque = deque(maxlen=30)
        self.last_action: np.ndarray = np.zeros(max_dims)
        self.last_position: np.ndarray = None
        
    def extract_features(
        self, 
        obs: np.ndarray, 
        action: np.ndarray, 
        peer_id: int, 
        reward: float, 
        next_obs: np.ndarray = None
    ) -> np.ndarray:
        """
        Extract feature vector from Room environment
        
        Args:
            obs: Room observation (position and goal concatenated)
                 For 1D: [agent_pos, goal_pos] (2-dim)
                 For 2D: [agent_x, agent_y, goal_x, goal_y] (4-dim)
            action: Discrete action (0-5 depending on dims)
            peer_id: ID of suggesting peer (0 to n_peers-1)
            reward: Current step reward
            next_obs: Next observation (optional)
            
        Returns:
            Feature vector
        """
        if obs is None or len(obs) < 2 * self.max_dims:
            return self._get_default_features()
        
        # Parse observation
        agent_pos = obs[:self.max_dims]
        goal_pos = obs[self.max_dims:2*self.max_dims]
        
        # Extract feature components
        position_features = self._extract_position_features(agent_pos)          # max_dims
        goal_features = self._extract_goal_features(agent_pos, goal_pos)        # max_dims + 2
        movement_features = self._extract_movement_features(agent_pos)          # 5
        action_features = self._extract_action_features(action)                 # max_dims + 2
        peer_features = self._extract_peer_features(peer_id)                    # 4
        performance_features = self._extract_performance_features(reward)       # 3
        
        # Update tracking
        self.peer_action_history[peer_id].append(action.copy() if isinstance(action, np.ndarray) else np.array([action]))
        self.reward_history.append(reward)
        self.position_history.append(agent_pos.copy())
        self.last_action = action.copy() if isinstance(action, np.ndarray) else np.array([action])
        self.last_position = agent_pos.copy()
        
        # Concatenate all features
        features = np.concatenate([
            position_features,      # max_dims
            goal_features,          # max_dims + 2
            movement_features,      # 5
            action_features,        # max_dims + 2
            peer_features,          # 4
            performance_features    # 3
        ])
        
        # Pad to fixed size if needed
        if len(features) < self.feature_dim:
            features = np.concatenate([features, np.zeros(self.feature_dim - len(features))])
        
        return features[:self.feature_dim].astype(np.float32)
    
    def _extract_position_features(self, agent_pos: np.ndarray) -> np.ndarray:
        """Position analysis (max_dims dimensions)"""
        # Normalize position to [0, 1]
        normalized_pos = np.clip(agent_pos, 0.0, 1.0)
        return normalized_pos.copy()
    
    def _extract_goal_features(self, agent_pos: np.ndarray, goal_pos: np.ndarray) -> np.ndarray:
        """Goal and distance analysis (max_dims + 2 dimensions)"""
        # Normalized goal position
        normalized_goal = np.clip(goal_pos, 0.0, 1.0)
        
        # Distance to goal (Euclidean)
        distance = np.linalg.norm(agent_pos - goal_pos)
        
        # Direction towards goal (normalized)
        goal_direction = (goal_pos - agent_pos) / (distance + 1e-6)
        
        # Distance ratio (0-1 scale assuming max possible distance is sqrt(max_dims))
        max_distance = np.sqrt(self.max_dims)
        distance_ratio = np.clip(distance / max_distance, 0.0, 1.0)
        
        features = list(normalized_goal) + [distance, distance_ratio]
        return np.array(features)
    
    def _extract_movement_features(self, agent_pos: np.ndarray) -> np.ndarray:
        """Movement and exploration analysis (5 dimensions)"""
        # Velocity estimate (from position history)
        velocity = 0.0
        if self.last_position is not None:
            velocity = np.linalg.norm(agent_pos - self.last_position)
        
        # Movement efficiency (velocity with direction towards goal consistency)
        movement_efficiency = 0.0
        if len(self.position_history) > 1:
            displacements = [np.linalg.norm(self.position_history[i] - self.position_history[i-1]) 
                           for i in range(1, len(self.position_history))]
            if displacements:
                movement_efficiency = np.mean(displacements)
        
        # Exploration coverage (how spread out positions are)
        exploration_coverage = 0.0
        if len(self.position_history) > 2:
            positions_array = np.array(list(self.position_history))
            exploration_coverage = np.mean(np.std(positions_array, axis=0))
        
        # Path efficiency (direct distance / actual path traveled)
        path_efficiency = 0.0
        if len(self.position_history) > 2:
            first_pos = self.position_history[0]
            last_pos = self.position_history[-1]
            direct_distance = np.linalg.norm(last_pos - first_pos)
            path_length = sum(np.linalg.norm(self.position_history[i] - self.position_history[i-1]) 
                            for i in range(1, len(self.position_history)))
            path_efficiency = direct_distance / (path_length + 1e-6)
        
        # Stuck indicator (very small movement over time)
        stuck_indicator = 1.0 if movement_efficiency < 0.01 else 0.0
        
        return np.array([
            velocity,
            movement_efficiency,
            exploration_coverage,
            path_efficiency,
            stuck_indicator
        ])
    
    def _extract_action_features(self, action: np.ndarray) -> np.ndarray:
        """Action context (max_dims + 2 dimensions)"""
        # Convert action to array if needed
        if isinstance(action, (int, np.integer)):
            action_array = np.zeros(self.max_dims)
            action_array[action // 2 if self.max_dims > 1 else 0] = 1 if action % 2 == 0 else -1
        else:
            action_array = action.copy()
        
        # Pad to max_dims if needed
        if len(action_array) < self.max_dims:
            action_array = np.concatenate([action_array, np.zeros(self.max_dims - len(action_array))])
        
        # Action magnitude
        action_magnitude = np.linalg.norm(action_array[:self.max_dims])
        
        # Action smoothness (change from last action)
        action_smoothness = 1.0 / (1.0 + np.linalg.norm(action_array[:self.max_dims] - self.last_action[:self.max_dims]))
        
        features = list(action_array[:self.max_dims]) + [action_magnitude, action_smoothness]
        return np.array(features)
    
    def _extract_peer_features(self, peer_id: int) -> np.ndarray:
        """Peer context (4 dimensions)"""
        peer_onehot = np.zeros(4)
        if peer_id < 4:
            peer_onehot[peer_id] = 1.0
        return peer_onehot
    
    def _extract_performance_features(self, reward: float) -> np.ndarray:
        """Performance tracking (3 dimensions)"""
        # Current reward
        current_reward = np.clip(reward / 1.0, -1.0, 1.0)
        
        # Reward trend
        reward_trend = 0.0
        if len(self.reward_history) > 0:
            reward_trend = reward - self.reward_history[-1]
        
        # Cumulative performance
        cumulative_performance = np.mean(self.reward_history) if len(self.reward_history) > 0 else 0.0
        
        return np.array([current_reward, reward_trend, cumulative_performance])
    
    def _get_default_features(self) -> np.ndarray:
        """Return default features if extraction fails"""
        return np.zeros(self.feature_dim, dtype=np.float32)
    
    def reset(self):
        """Reset tracking histories"""
        self.peer_action_history = {i: deque(maxlen=50) for i in range(self.n_peers)}
        self.reward_history = deque(maxlen=20)
        self.position_history = deque(maxlen=30)
        self.last_action = np.zeros(self.max_dims)
        self.last_position = None
