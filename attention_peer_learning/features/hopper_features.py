"""
Hopper-v4 Feature Extractor
Implements 38-dimensional feature extraction as per paper Section 3.3.1

Feature Breakdown:
- Position Features (8): height, angle, joint positions (3), body_tilt, height_critical, optimal_height_indicator
- Joint Features (10): joint_angles (3), joint_velocities (3), joint_power (3), joint_coordination
- Gait Features (8): gait_phase_onehot (4), stance_ratio, flight_time, gait_stability, gait_efficiency
- Action Context (6): action (3), action_magnitude, action_smoothness, action_efficiency
- Peer Context (4): peer_onehot (4)
- Performance Features (2): reward_normalized, performance_trend
"""

import numpy as np
from collections import deque
from typing import Dict, Tuple


class HopperFeatureExtractor:
    """Extract 38-dimensional features from Hopper environment"""
    
    def __init__(self, n_peers: int = 4):
        """
        Initialize feature extractor
        
        Args:
            n_peers: Number of peers in the group
        """
        self.n_peers = n_peers
        self.optimal_height = 1.25  # Target hopping height
        
        # Tracking histories
        self.peer_action_history: Dict[int, deque] = {i: deque(maxlen=50) for i in range(n_peers)}
        self.reward_history: deque = deque(maxlen=20)
        self.gait_history: deque = deque(maxlen=30)
        self.last_action: np.ndarray = np.zeros(3)
        self.stance_history: deque = deque(maxlen=10)
        
    def extract_features(
        self, 
        obs: np.ndarray, 
        action: np.ndarray, 
        peer_id: int, 
        reward: float, 
        next_obs: np.ndarray = None
    ) -> np.ndarray:
        """
        Extract 38-dimensional feature vector
        
        Args:
            obs: Hopper observation (11-dim)
            action: Continuous action (3-dim: thigh, leg, foot torques)
            peer_id: ID of suggesting peer (0 to n_peers-1)
            reward: Current step reward
            next_obs: Next observation (optional)
            
        Returns:
            Feature vector of shape (38,)
        """
        if obs is None or len(obs) < 11:
            return self._get_default_features()
        
        # Parse observation (11-dim)
        # [0]: z (height), [1]: angle, [2-4]: joint angles (thigh, leg, foot)
        # [5-7]: joint velocities, [8-10]: body velocities
        height = obs[0]
        angle = obs[1]
        joint_angles = obs[2:5]
        joint_velocities = obs[5:8]
        body_velocities = obs[8:11]
        
        # Extract feature components
        position_features = self._extract_position_features(height, angle, joint_angles)     # 8-dim
        joint_features = self._extract_joint_features(joint_angles, joint_velocities, action)  # 10-dim
        gait_features = self._extract_gait_features(height, joint_angles, body_velocities)    # 8-dim
        action_features = self._extract_action_features(action)                               # 6-dim
        peer_features = self._extract_peer_features(peer_id)                                  # 4-dim
        performance_features = self._extract_performance_features(reward)                     # 2-dim
        
        # Update tracking
        self.peer_action_history[peer_id].append(action.copy())
        self.reward_history.append(reward)
        self.last_action = action.copy()
        
        # Concatenate all features
        features = np.concatenate([
            position_features,      # 8
            joint_features,         # 10
            gait_features,          # 8
            action_features,        # 6
            peer_features,          # 4
            performance_features    # 2
        ])
        
        return features.astype(np.float32)
    
    def _extract_position_features(self, height: float, angle: float, joint_angles: np.ndarray) -> np.ndarray:
        """Position and body state analysis (8 dimensions)"""
        # Height analysis
        height_critical = 1.0 if height < 0.8 else 0.0
        optimal_height_indicator = 1.0 if 1.0 < height < 1.5 else 0.0
        
        # Body orientation
        body_tilt = abs(angle)
        
        return np.array([
            height,
            angle,
            joint_angles[0],  # thigh angle
            joint_angles[1],  # leg angle
            joint_angles[2],  # foot angle
            body_tilt,
            height_critical,
            optimal_height_indicator
        ])
    
    def _extract_joint_features(self, joint_angles: np.ndarray, joint_velocities: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Joint state and power analysis (10 dimensions)"""
        # Joint power (torque * velocity)
        joint_power = action * joint_velocities
        
        # Joint coordination (variance across joints)
        joint_coordination = 1.0 / (1.0 + np.std(joint_velocities))
        
        return np.array([
            joint_angles[0], joint_angles[1], joint_angles[2],
            joint_velocities[0], joint_velocities[1], joint_velocities[2],
            joint_power[0], joint_power[1], joint_power[2],
            joint_coordination
        ])
    
    def _extract_gait_features(self, height: float, joint_angles: np.ndarray, body_velocities: np.ndarray) -> np.ndarray:
        """Gait phase and locomotion analysis (8 dimensions)"""
        # Detect gait phase based on height and joint configuration
        gait_phase = self._detect_gait_phase(height, joint_angles, body_velocities)
        self.gait_history.append(gait_phase)
        
        # Gait phase one-hot (4 phases: stance, swing, flight, landing)
        gait_onehot = np.zeros(4)
        gait_onehot[gait_phase] = 1.0
        
        # Stance ratio (time in stance phase)
        stance_ratio = 0.0
        if len(self.gait_history) > 0:
            stance_ratio = sum(1 for p in self.gait_history if p == 0) / len(self.gait_history)
        
        # Flight time estimate
        flight_time = sum(1 for p in list(self.gait_history)[-5:] if p == 2) / 5.0 if len(self.gait_history) >= 5 else 0.0
        
        # Gait stability (consistency of phase transitions)
        gait_stability = 0.0
        if len(self.gait_history) > 5:
            phase_changes = sum(1 for i in range(len(self.gait_history)-1) if self.gait_history[i] != self.gait_history[i+1])
            gait_stability = 1.0 / (1.0 + phase_changes)
        
        # Gait efficiency (height change per cycle)
        gait_efficiency = height / max(1.0, abs(body_velocities[2]) + 0.1)
        
        return np.array([
            *gait_onehot,       # 4
            stance_ratio,       # 1
            flight_time,        # 1
            gait_stability,     # 1
            gait_efficiency     # 1
        ])
    
    def _detect_gait_phase(self, height: float, joint_angles: np.ndarray, body_velocities: np.ndarray) -> int:
        """
        Detect current gait phase
        
        Returns:
            0: stance (grounded, supporting)
            1: swing (leg swinging forward)
            2: flight (airborne)
            3: landing (descending to ground)
        """
        vertical_velocity = body_velocities[2]
        leg_extension = joint_angles[1]  # Leg joint angle
        
        # Flight phase (high height, positive vertical velocity)
        if height > 1.3 and vertical_velocity > 0:
            return 2  # flight
        
        # Landing phase (descending)
        if vertical_velocity < -0.2:
            return 3  # landing
        
        # Swing phase (leg flexed, moving)
        if abs(leg_extension) > 0.3 and height > 1.0:
            return 1  # swing
        
        # Default: stance
        return 0  # stance
    
    def _extract_action_features(self, action: np.ndarray) -> np.ndarray:
        """Action context analysis (6 dimensions)"""
        # Action magnitude
        action_magnitude = np.linalg.norm(action)
        
        # Action smoothness (change from last action)
        action_smoothness = 1.0 / (1.0 + np.linalg.norm(action - self.last_action))
        
        # Action efficiency (low magnitude, high coordination)
        action_efficiency = 1.0 / (1.0 + action_magnitude * np.std(action))
        
        return np.array([
            action[0],           # thigh torque
            action[1],           # leg torque
            action[2],           # foot torque
            action_magnitude,
            action_smoothness,
            action_efficiency
        ])
    
    def _extract_peer_features(self, peer_id: int) -> np.ndarray:
        """Peer context (4 dimensions)"""
        peer_onehot = np.zeros(4)
        if peer_id < 4:
            peer_onehot[peer_id] = 1.0
        return peer_onehot
    
    def _extract_performance_features(self, reward: float) -> np.ndarray:
        """Performance tracking (2 dimensions)"""
        reward_normalized = np.clip(reward / 10.0, -1.0, 1.0)
        
        performance_trend = 0.0
        if len(self.reward_history) > 0:
            recent_avg = np.mean(list(self.reward_history)[-5:])
            performance_trend = reward - recent_avg
        
        return np.array([reward_normalized, performance_trend])
    
    def _get_default_features(self) -> np.ndarray:
        """Return default zero features if extraction fails"""
        return np.zeros(38, dtype=np.float32)
    
    def reset(self):
        """Reset tracking histories"""
        self.peer_action_history = {i: deque(maxlen=50) for i in range(self.n_peers)}
        self.reward_history = deque(maxlen=20)
        self.gait_history = deque(maxlen=30)
        self.last_action = np.zeros(3)
        self.stance_history = deque(maxlen=10)
