"""
Ant-v4 Feature Extractor
Implements 52-dimensional feature extraction for quadrupedal locomotion

Feature Breakdown:
- Position Features (8): height, orientation (3), torso_stability, optimal_height_indicator, forward_progress, balance_score
- Joint Features (16): joint_angles (8), joint_velocities (8)
- Locomotion Features (10): gait_phase_onehot (4), leg_coordination, diagonal_symmetry, stride_efficiency, contact_ratio, body_alignment, movement_smoothness
- Action Context (8): action (8 torques)
- Peer Context (4): peer_onehot (4)
- Performance Features (6): reward_normalized, velocity_efficiency, energy_efficiency, stability_score, progress_rate, action_consistency
"""

import numpy as np
from collections import deque
from typing import Dict, Tuple


class AntFeatureExtractor:
    """Extract 52-dimensional features from Ant-v4 environment"""
    
    def __init__(self, n_peers: int = 4):
        """
        Initialize feature extractor
        
        Args:
            n_peers: Number of peers in the group
        """
        self.n_peers = n_peers
        self.optimal_height = 0.55  # Target torso height for Ant
        self.optimal_speed = 1.0    # Target forward velocity
        
        # Tracking histories
        self.peer_action_history: Dict[int, deque] = {i: deque(maxlen=50) for i in range(n_peers)}
        self.reward_history: deque = deque(maxlen=20)
        self.gait_history: deque = deque(maxlen=30)
        self.last_action: np.ndarray = np.zeros(8)
        self.position_history: deque = deque(maxlen=10)
        self.velocity_history: deque = deque(maxlen=10)
        
    def extract_features(
        self, 
        obs: np.ndarray, 
        action: np.ndarray, 
        peer_id: int, 
        reward: float, 
        next_obs: np.ndarray = None
    ) -> np.ndarray:
        """
        Extract 52-dimensional feature vector
        
        Args:
            obs: Ant observation (27-dim or 29-dim with contacts)
            action: Continuous action (8-dim: torques for 8 hip/ankle joints)
            peer_id: ID of suggesting peer (0 to n_peers-1)
            reward: Current step reward
            next_obs: Next observation (optional)
            
        Returns:
            Feature vector of shape (52,)
        """
        if obs is None or len(obs) < 27:
            return self._get_default_features()
        
        # Parse observation (27-dim standard Ant-v4)
        # [0-12]: joint angles and velocities (some versions include xyz)
        # Standard Ant-v4 observation:
        # [0-1]: x, y position (sometimes excluded in some versions)
        # [0-7 or 2-9]: joint angles (8 joints: 4 hip + 4 ankle)
        # [8-14 or 10-16]: joint velocities
        # [15-26 or 17-28]: additional state info (orientation, angular velocities, contact forces)
        
        # Robust parsing for different Ant observation formats
        if len(obs) == 27:
            # Standard format without explicit xy coordinates exposed
            joint_angles = obs[0:8]
            joint_velocities = obs[8:16]
            additional_state = obs[16:27]
        elif len(obs) == 29:
            # Format with contact forces
            joint_angles = obs[2:10]
            joint_velocities = obs[10:18]
            additional_state = obs[18:29]
        else:
            # Fallback
            joint_angles = obs[:8] if len(obs) >= 8 else np.zeros(8)
            joint_velocities = obs[8:16] if len(obs) >= 16 else np.zeros(8)
            additional_state = obs[16:] if len(obs) > 16 else np.zeros(11)
        
        # Try to extract height and orientation
        # Height is typically around index 0 or derived from z-coordinate
        height = additional_state[0] if len(additional_state) > 0 else 0.55
        orientation = additional_state[1:4] if len(additional_state) >= 4 else np.array([1.0, 0.0, 0.0])
        body_velocities = additional_state[4:7] if len(additional_state) >= 7 else np.zeros(3)
        angular_velocities = additional_state[7:10] if len(additional_state) >= 10 else np.zeros(3)
        
        # Extract feature components
        position_features = self._extract_position_features(
            height, orientation, joint_angles, body_velocities
        )  # 8-dim
        
        joint_features = self._extract_joint_features(
            joint_angles, joint_velocities
        )  # 16-dim
        
        locomotion_features = self._extract_locomotion_features(
            joint_angles, joint_velocities, orientation, angular_velocities
        )  # 10-dim
        
        action_features = self._extract_action_features(action)  # 8-dim
        
        peer_features = self._extract_peer_features(peer_id)  # 4-dim
        
        performance_features = self._extract_performance_features(
            reward, joint_velocities, action, body_velocities
        )  # 6-dim
        
        # Update tracking
        self.peer_action_history[peer_id].append(action.copy())
        self.reward_history.append(reward)
        self.last_action = action.copy()
        self.position_history.append(height)
        if len(body_velocities) > 0:
            self.velocity_history.append(np.linalg.norm(body_velocities[:2]))  # xy velocity
        
        # Concatenate all features
        features = np.concatenate([
            position_features,      # 8
            joint_features,         # 16
            locomotion_features,    # 10
            action_features,        # 8
            peer_features,          # 4
            performance_features    # 6
        ])
        
        return features.astype(np.float32)
    
    def _extract_position_features(
        self, 
        height: float, 
        orientation: np.ndarray, 
        joint_angles: np.ndarray,
        body_velocities: np.ndarray
    ) -> np.ndarray:
        """Position and body state analysis (8 dimensions)"""
        # Torso stability (upright orientation)
        # orientation is typically [w, x, y, z] quaternion or euler angles
        torso_stability = 1.0 / (1.0 + np.linalg.norm(orientation[1:]))  # Penalize tilt
        
        # Height analysis
        optimal_height_indicator = 1.0 if 0.4 < height < 0.7 else 0.0
        
        # Forward progress (x-velocity if available)
        forward_progress = body_velocities[0] if len(body_velocities) > 0 else 0.0
        
        # Balance score (center of mass consideration via joint symmetry)
        left_joints = joint_angles[0:4:2]  # e.g., front-left, back-left hips
        right_joints = joint_angles[1:4:2]  # e.g., front-right, back-right hips
        balance_score = 1.0 / (1.0 + abs(np.mean(left_joints) - np.mean(right_joints)))
        
        return np.array([
            height,
            orientation[0] if len(orientation) > 0 else 1.0,  # Primary orientation component
            orientation[1] if len(orientation) > 1 else 0.0,
            orientation[2] if len(orientation) > 2 else 0.0,
            torso_stability,
            optimal_height_indicator,
            forward_progress,
            balance_score
        ])
    
    def _extract_joint_features(
        self, 
        joint_angles: np.ndarray, 
        joint_velocities: np.ndarray
    ) -> np.ndarray:
        """Joint state analysis (16 dimensions)"""
        # 8 joint angles + 8 joint velocities
        return np.concatenate([
            joint_angles,       # 8 angles (4 hip + 4 ankle)
            joint_velocities    # 8 velocities
        ])
    
    def _extract_locomotion_features(
        self, 
        joint_angles: np.ndarray, 
        joint_velocities: np.ndarray,
        orientation: np.ndarray,
        angular_velocities: np.ndarray
    ) -> np.ndarray:
        """Quadrupedal gait and locomotion analysis (10 dimensions)"""
        # Detect gait phase for quadrupedal locomotion
        gait_phase = self._detect_ant_gait_phase(joint_angles, joint_velocities)
        self.gait_history.append(gait_phase)
        
        # Gait phase one-hot (4 phases: trot, pace, gallop, stand)
        gait_onehot = np.zeros(4)
        gait_onehot[gait_phase] = 1.0
        
        # Leg coordination (diagonal legs should move together in trot)
        front_left_hip = joint_angles[0] if len(joint_angles) > 0 else 0.0
        back_right_hip = joint_angles[3] if len(joint_angles) > 3 else 0.0
        front_right_hip = joint_angles[1] if len(joint_angles) > 1 else 0.0
        back_left_hip = joint_angles[2] if len(joint_angles) > 2 else 0.0
        
        diagonal_1 = abs(front_left_hip - back_right_hip)
        diagonal_2 = abs(front_right_hip - back_left_hip)
        leg_coordination = 1.0 / (1.0 + min(diagonal_1, diagonal_2))
        
        # Diagonal symmetry (left-right balance)
        diagonal_symmetry = 1.0 / (1.0 + abs(diagonal_1 - diagonal_2))
        
        # Stride efficiency (velocity per joint movement)
        total_joint_movement = np.sum(np.abs(joint_velocities))
        stride_efficiency = 1.0 / (1.0 + total_joint_movement) if total_joint_movement > 0 else 0.0
        
        # Contact ratio (estimate from joint configuration)
        # Low joint velocities suggest stance phase
        contact_ratio = sum(1 for v in joint_velocities if abs(v) < 0.5) / len(joint_velocities)
        
        # Body alignment (minimal angular velocity = stable movement)
        body_alignment = 1.0 / (1.0 + np.linalg.norm(angular_velocities))
        
        # Movement smoothness (gait consistency)
        movement_smoothness = 0.0
        if len(self.gait_history) > 5:
            phase_changes = sum(1 for i in range(len(self.gait_history)-1) 
                              if self.gait_history[i] != self.gait_history[i+1])
            movement_smoothness = 1.0 / (1.0 + phase_changes)
        
        return np.array([
            *gait_onehot,           # 4
            leg_coordination,       # 1
            diagonal_symmetry,      # 1
            stride_efficiency,      # 1
            contact_ratio,          # 1
            body_alignment,         # 1
            movement_smoothness     # 1
        ])
    
    def _detect_ant_gait_phase(
        self, 
        joint_angles: np.ndarray, 
        joint_velocities: np.ndarray
    ) -> int:
        """
        Detect quadrupedal gait phase
        
        Returns:
            0: trot (diagonal legs move together)
            1: pace (lateral legs move together)
            2: gallop (all legs together)
            3: stand (minimal movement)
        """
        # Calculate movement magnitude
        velocity_magnitude = np.sum(np.abs(joint_velocities))
        
        # Stand phase (very little movement)
        if velocity_magnitude < 1.0:
            return 3  # stand
        
        # Detect diagonal coordination (trot)
        # Front-left & back-right vs front-right & back-left
        if len(joint_velocities) >= 4:
            diag1_vel = abs(joint_velocities[0]) + abs(joint_velocities[3])  # FL + BR
            diag2_vel = abs(joint_velocities[1]) + abs(joint_velocities[2])  # FR + BL
            lateral_vel = abs(joint_velocities[0] + joint_velocities[1]) + abs(joint_velocities[2] + joint_velocities[3])
            
            # Trot: diagonal legs synchronized
            if abs(diag1_vel - diag2_vel) < 2.0:
                return 0  # trot
            
            # Pace: lateral legs synchronized
            if lateral_vel > diag1_vel + diag2_vel:
                return 1  # pace
        
        # Default: gallop
        return 2  # gallop
    
    def _extract_action_features(self, action: np.ndarray) -> np.ndarray:
        """Action context (8 dimensions)"""
        # Return 8 torque values directly
        return action.copy()
    
    def _extract_peer_features(self, peer_id: int) -> np.ndarray:
        """Peer context (4 dimensions)"""
        peer_onehot = np.zeros(4)
        if peer_id < 4:
            peer_onehot[peer_id] = 1.0
        return peer_onehot
    
    def _extract_performance_features(
        self, 
        reward: float, 
        joint_velocities: np.ndarray, 
        action: np.ndarray,
        body_velocities: np.ndarray
    ) -> np.ndarray:
        """Performance and efficiency tracking (6 dimensions)"""
        # Reward normalized
        reward_normalized = np.clip(reward / 10.0, -1.0, 1.0)
        
        # Velocity efficiency (reward per joint movement)
        total_joint_velocity = np.sum(np.abs(joint_velocities))
        velocity_efficiency = reward / max(1.0, total_joint_velocity)
        
        # Energy efficiency (reward per action magnitude)
        action_magnitude = np.linalg.norm(action)
        energy_efficiency = reward / max(1.0, action_magnitude)
        
        # Stability score (consistent height)
        stability_score = 0.0
        if len(self.position_history) > 5:
            height_variance = np.var(list(self.position_history))
            stability_score = 1.0 / (1.0 + height_variance)
        
        # Progress rate (forward velocity consistency)
        progress_rate = 0.0
        if len(self.velocity_history) > 5:
            recent_velocities = list(self.velocity_history)[-5:]
            progress_rate = np.mean(recent_velocities)
        
        # Action consistency (smooth action changes)
        action_consistency = 1.0 / (1.0 + np.linalg.norm(action - self.last_action))
        
        return np.array([
            reward_normalized,
            velocity_efficiency,
            energy_efficiency,
            stability_score,
            progress_rate,
            action_consistency
        ])
    
    def _get_default_features(self) -> np.ndarray:
        """Return default zero features if extraction fails"""
        return np.zeros(52, dtype=np.float32)
    
    def reset(self):
        """Reset tracking histories"""
        self.peer_action_history = {i: deque(maxlen=50) for i in range(self.n_peers)}
        self.reward_history = deque(maxlen=20)
        self.gait_history = deque(maxlen=30)
        self.last_action = np.zeros(8)
        self.position_history = deque(maxlen=10)
        self.velocity_history = deque(maxlen=10)
