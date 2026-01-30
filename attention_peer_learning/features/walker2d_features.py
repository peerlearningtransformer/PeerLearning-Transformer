"""
Walker2D-v4 Feature Extractor
Implements 42-dimensional feature extraction as per paper Section 3.3.1

Feature Breakdown:
- Position Features (10): height, angle, joint positions (6), body_stability, optimal_height_indicator
- Joint Features (12): joint_angles (6), joint_velocities (6)
- Gait Features (8): gait_phase_onehot (4), left_right_balance, double_support_ratio, gait_symmetry, stride_efficiency
- Action Context (6): action (6)
- Peer Context (4): peer_onehot (4)
- Performance Features (2): reward_normalized, velocity_efficiency
"""

import numpy as np
from collections import deque
from typing import Dict, Tuple


class Walker2DFeatureExtractor:
    """Extract 42-dimensional features from Walker2D environment"""
    
    def __init__(self, n_peers: int = 4):
        """
        Initialize feature extractor
        
        Args:
            n_peers: Number of peers in the group
        """
        self.n_peers = n_peers
        self.optimal_height = 1.3  # Target walking height
        self.optimal_speed = 1.0   # Target forward velocity
        
        # Tracking histories
        self.peer_action_history: Dict[int, deque] = {i: deque(maxlen=50) for i in range(n_peers)}
        self.reward_history: deque = deque(maxlen=20)
        self.gait_history: deque = deque(maxlen=30)
        self.last_action: np.ndarray = np.zeros(6)
        self.foot_contact_history: deque = deque(maxlen=10)
        
    def extract_features(
        self, 
        obs: np.ndarray, 
        action: np.ndarray, 
        peer_id: int, 
        reward: float, 
        next_obs: np.ndarray = None
    ) -> np.ndarray:
        """
        Extract 42-dimensional feature vector
        
        Args:
            obs: Walker2D observation (17-dim)
            action: Continuous action (6-dim: torques for 6 joints)
            peer_id: ID of suggesting peer (0 to n_peers-1)
            reward: Current step reward
            next_obs: Next observation (optional)
            
        Returns:
            Feature vector of shape (42,)
        """
        if obs is None or len(obs) < 17:
            return self._get_default_features()
        
        # Parse observation (17-dim)
        # [0]: height (z), [1]: angle, [2-7]: joint angles (6 joints)
        # [8]: angular velocity, [9-14]: joint velocities, [15-16]: foot contacts
        height = obs[0]
        angle = obs[1]
        joint_angles = obs[2:8]      # 6 joints
        angular_velocity = obs[8]
        joint_velocities = obs[9:15]  # 6 velocities
        foot_contacts = obs[15:17] if len(obs) >= 17 else np.array([0.0, 0.0])
        
        # Extract feature components
        position_features = self._extract_position_features(height, angle, joint_angles)        # 10-dim
        joint_features = self._extract_joint_features(joint_angles, joint_velocities)           # 12-dim
        gait_features = self._extract_gait_features(foot_contacts, joint_angles, height)        # 8-dim
        action_features = self._extract_action_features(action)                                 # 6-dim
        peer_features = self._extract_peer_features(peer_id)                                    # 4-dim
        performance_features = self._extract_performance_features(reward, joint_velocities)     # 2-dim
        
        # Update tracking
        self.peer_action_history[peer_id].append(action.copy())
        self.reward_history.append(reward)
        self.last_action = action.copy()
        self.foot_contact_history.append(foot_contacts.copy())
        
        # Concatenate all features
        features = np.concatenate([
            position_features,      # 10
            joint_features,         # 12
            gait_features,          # 8
            action_features,        # 6
            peer_features,          # 4
            performance_features    # 2
        ])
        
        return features.astype(np.float32)
    
    def _extract_position_features(self, height: float, angle: float, joint_angles: np.ndarray) -> np.ndarray:
        """Position and stability analysis (10 dimensions)"""
        # Height analysis
        optimal_height_indicator = 1.0 if 1.0 < height < 1.6 else 0.0
        
        # Body stability
        body_stability = 1.0 / (1.0 + abs(angle))
        
        return np.array([
            height,
            angle,
            joint_angles[0],  # thigh (right)
            joint_angles[1],  # leg (right)
            joint_angles[2],  # foot (right)
            joint_angles[3],  # thigh (left)
            joint_angles[4],  # leg (left)
            joint_angles[5],  # foot (left)
            body_stability,
            optimal_height_indicator
        ])
    
    def _extract_joint_features(self, joint_angles: np.ndarray, joint_velocities: np.ndarray) -> np.ndarray:
        """Joint state analysis (12 dimensions)"""
        return np.concatenate([
            joint_angles,       # 6 angles
            joint_velocities    # 6 velocities
        ])
    
    def _extract_gait_features(self, foot_contacts: np.ndarray, joint_angles: np.ndarray, height: float) -> np.ndarray:
        """Gait phase and bipedal locomotion analysis (8 dimensions)"""
        # Detect bipedal gait phase
        gait_phase = self._detect_walker_gait_phase(foot_contacts, joint_angles)
        self.gait_history.append(gait_phase)
        
        # Gait phase one-hot (4 phases: left_stance, right_stance, double_support, flight)
        gait_onehot = np.zeros(4)
        gait_onehot[gait_phase] = 1.0
        
        # Left-right balance (foot contact symmetry)
        left_right_balance = abs(foot_contacts[0] - foot_contacts[1])
        
        # Double support ratio (both feet on ground)
        double_support_ratio = 0.0
        if len(self.foot_contact_history) > 0:
            double_support_count = sum(1 for fc in self.foot_contact_history if fc[0] > 0.5 and fc[1] > 0.5)
            double_support_ratio = double_support_count / len(self.foot_contact_history)
        
        # Gait symmetry (left vs right leg movement similarity)
        gait_symmetry = 1.0 / (1.0 + abs(joint_angles[0] - joint_angles[3]))  # Compare left/right thighs
        
        # Stride efficiency (height maintained during walking)
        stride_efficiency = height / max(1.0, abs(height - self.optimal_height) + 0.1)
        
        return np.array([
            *gait_onehot,           # 4
            left_right_balance,     # 1
            double_support_ratio,   # 1
            gait_symmetry,          # 1
            stride_efficiency       # 1
        ])
    
    def _detect_walker_gait_phase(self, foot_contacts: np.ndarray, joint_angles: np.ndarray) -> int:
        """
        Detect bipedal walking gait phase
        
        Returns:
            0: left_stance (left foot on ground, right swinging)
            1: right_stance (right foot on ground, left swinging)
            2: double_support (both feet on ground)
            3: flight (neither foot on ground)
        """
        left_contact = foot_contacts[0] > 0.5
        right_contact = foot_contacts[1] > 0.5
        
        # Double support (both feet down)
        if left_contact and right_contact:
            return 2
        
        # Flight phase (neither foot down)
        if not left_contact and not right_contact:
            return 3
        
        # Left stance (only left foot down)
        if left_contact and not right_contact:
            return 0
        
        # Right stance (only right foot down)
        if right_contact and not left_contact:
            return 1
        
        # Default: double support
        return 2
    
    def _extract_action_features(self, action: np.ndarray) -> np.ndarray:
        """Action context (6 dimensions)"""
        # Return action torques directly
        return action.copy()
    
    def _extract_peer_features(self, peer_id: int) -> np.ndarray:
        """Peer context (4 dimensions)"""
        peer_onehot = np.zeros(4)
        if peer_id < 4:
            peer_onehot[peer_id] = 1.0
        return peer_onehot
    
    def _extract_performance_features(self, reward: float, joint_velocities: np.ndarray) -> np.ndarray:
        """Performance and efficiency tracking (2 dimensions)"""
        reward_normalized = np.clip(reward / 10.0, -1.0, 1.0)
        
        # Velocity efficiency (reward per unit of joint velocity)
        total_joint_velocity = np.sum(np.abs(joint_velocities))
        velocity_efficiency = reward / max(1.0, total_joint_velocity)
        
        return np.array([reward_normalized, velocity_efficiency])
    
    def _get_default_features(self) -> np.ndarray:
        """Return default zero features if extraction fails"""
        return np.zeros(42, dtype=np.float32)
    
    def reset(self):
        """Reset tracking histories"""
        self.peer_action_history = {i: deque(maxlen=50) for i in range(self.n_peers)}
        self.reward_history = deque(maxlen=20)
        self.gait_history = deque(maxlen=30)
        self.last_action = np.zeros(6)
        self.foot_contact_history = deque(maxlen=10)
