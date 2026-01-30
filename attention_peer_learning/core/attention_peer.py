"""
Attention-Enhanced Peer Learning
Extends base Peer class with transformer-based trust evaluation

Main Components:
1. Feature Extraction: Environment-specific feature extractors
2. Attention Transformer: Neural trust evaluation
3. Hybrid Trust: Combines traditional + attention-based trust
4. History Buffer: Stores sequences for transformer training

Classes:
- AttentionPeerMixin: Base mixin for attention functionality
- AttentionSACPeer: SAC + Attention for continuous control
- AttentionDQNPeer: DQN + Attention for discrete control
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Tuple, List
from collections import deque

from stable_baselines3 import SAC, DQN

from peer import make_peer_class

from ..configs.env_config import get_env_config
from ..features.lunar_features import LunarLanderFeatureExtractor
from ..features.hopper_features import HopperFeatureExtractor
from ..features.walker2d_features import Walker2DFeatureExtractor
from ..features.ant_features import AntFeatureExtractor
from ..features.room_features import RoomFeatureExtractor
from ..transformers.attention_transformer import AttentionTransformer, CounterfactualEvaluator
from ..transformers.phase_aware_transformer import PhaseAwareTransformer, PhaseDetector
from ..trust.hybrid_trust import HybridTrustIntegrator


class AttentionHistoryBuffer:
    """
    Circular buffer for storing feature sequences for transformer training
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        sequence_length: int = 50
    ):
        """
        Args:
            max_size: Maximum number of transitions to store
            sequence_length: Length of sequences to sample
        """
        self.max_size = max_size
        self.sequence_length = sequence_length
        
        # Storage
        self.features: deque = deque(maxlen=max_size)
        self.phases: deque = deque(maxlen=max_size)
        self.rewards: deque = deque(maxlen=max_size)
        self.peer_ids: deque = deque(maxlen=max_size)
        
        self.current_size = 0
    
    def add(
        self,
        feature: np.ndarray,
        phase: int,
        reward: float,
        peer_id: int
    ):
        """Add single transition to buffer"""
        self.features.append(feature.copy())
        self.phases.append(phase)
        self.rewards.append(reward)
        self.peer_ids.append(peer_id)
        
        self.current_size = min(self.current_size + 1, self.max_size)
    
    def sample_sequence(self, batch_size: int = 32) -> Optional[Dict[str, torch.Tensor]]:
        """
        Sample batch of sequences
        
        Returns:
            Dictionary with:
                - features: [batch, seq_len, feature_dim]
                - phases: [batch, seq_len]
                - rewards: [batch, seq_len]
                - peer_ids: [batch, seq_len]
                - mask: [batch, seq_len] (1 = valid, 0 = padding)
        """
        if self.current_size < self.sequence_length:
            return None
        
        sequences_features = []
        sequences_phases = []
        sequences_rewards = []
        sequences_peer_ids = []
        masks = []
        
        for _ in range(batch_size):
            # Sample start index
            max_start = self.current_size - self.sequence_length
            start_idx = np.random.randint(0, max_start + 1)
            
            # Extract sequence
            seq_features = [self.features[start_idx + i] for i in range(self.sequence_length)]
            seq_phases = [self.phases[start_idx + i] for i in range(self.sequence_length)]
            seq_rewards = [self.rewards[start_idx + i] for i in range(self.sequence_length)]
            seq_peer_ids = [self.peer_ids[start_idx + i] for i in range(self.sequence_length)]
            
            sequences_features.append(np.stack(seq_features))
            sequences_phases.append(seq_phases)
            sequences_rewards.append(seq_rewards)
            sequences_peer_ids.append(seq_peer_ids)
            masks.append(np.ones(self.sequence_length))
        
        return {
            'features': torch.FloatTensor(np.stack(sequences_features)),
            'phases': torch.LongTensor(sequences_phases),
            'rewards': torch.FloatTensor(sequences_rewards),
            'peer_ids': torch.LongTensor(sequences_peer_ids),
            'mask': torch.FloatTensor(np.array(masks))
        }
    
    def clear(self):
        """Clear buffer"""
        self.features.clear()
        self.phases.clear()
        self.rewards.clear()
        self.peer_ids.clear()
        self.current_size = 0


class AttentionPeerMixin:
    """
    Mixin class adding attention-based trust evaluation to base Peer
    
    This mixin adds:
    - Environment-specific feature extraction
    - Attention transformer for trust prediction
    - Hybrid trust integration
    - Transformer training loop
    """
    
    def __init__(
        self,
        env_name: str,
        use_phase_aware: bool = True,
        attention_lr: float = 1e-5,
        attention_update_freq: int = 20,
        sequence_length: int = 50,
        history_buffer_size: int = 10000,
        warmup_steps: int = 500000,
        **kwargs
    ):
        """
        Initialize attention peer
        
        Args:
            env_name: Environment name ('LunarLander-v2', 'Hopper-v4', 'Walker2d-v4')
            use_phase_aware: Use phase-aware transformer (default: True)
            attention_lr: Learning rate for transformer (default: 1e-5)
            attention_update_freq: Update transformer every N steps (default: 20)
            sequence_length: Length of sequences for training (default: 50)
            history_buffer_size: Size of history buffer (default: 10000)
            warmup_steps: Warmup steps for attention mechanism (default: 500000)
            **kwargs: Passed to base Peer class
        """
        # Initialize base peer (will be called after mixin in MRO)
        super().__init__(**kwargs)
        
        # Get environment configuration
        self.env_config = get_env_config(env_name)
        self.env_name = env_name
        self.use_phase_aware = use_phase_aware
        self.attention_update_freq = attention_update_freq
        self.sequence_length = sequence_length
        
        self.feature_extractor = self._create_feature_extractor()
        
        self.transformer = self._create_transformer()
        if hasattr(self, 'device'):
            self.transformer = self.transformer.to(self.device)
        self.transformer_optimizer = optim.Adam(
            self.transformer.parameters(),
            lr=attention_lr
        )
        
        init_trust = kwargs.get('init_trust_values', 200.0)
        self.hybrid_trust = HybridTrustIntegrator(
            n_peers=4,  # Will be updated when n_peers is set
            trust_scale=self.env_config.trust_scale,
            attention_weight_coef=self.env_config.attention_weight,
            initial_trust_value=init_trust,
            warmup_steps=warmup_steps
        )
        
        self.history_buffer = AttentionHistoryBuffer(
            max_size=history_buffer_size,
            sequence_length=sequence_length
        )
        
        self.counterfactual_eval = CounterfactualEvaluator(n_peers=4)
        
        self.phase_detector = self._get_phase_detector()
        
        self.attention_step_count = 0
        self.last_obs = None
        self.last_action = None
        self.last_peer_id = None
        
        print(f"✅ Attention Peer initialized for {env_name}")
        print(f"   Feature dim: {self.env_config.feature_dim}")
        print(f"   Phase-aware: {use_phase_aware}")
        print(f"   Phases: {self.env_config.phases}")
        
        # Verify peer learning integration
        if hasattr(self, '_predict_train'):
            print(f"   ✓ Peer learning integration: ACTIVE (_predict_train found)")
        else:
            print(f"   ✗ WARNING: Peer learning integration: MISSING (_predict_train not found!)")
    
    @property
    def n_peers(self):
        """Get number of peers"""
        return self._Peer__n_peers if hasattr(self, '_Peer__n_peers') else 4
    
    @n_peers.setter
    def n_peers(self, value):
        """Set number of peers and update attention components"""
        if hasattr(self.__class__.__bases__[1], 'n_peers'):
            parent_prop = self.__class__.__bases__[1].__dict__.get('n_peers')
            if parent_prop and hasattr(parent_prop, 'fset'):
                parent_prop.fset(self, value)
        else:
            self._Peer__n_peers = value
        
        if hasattr(self, 'hybrid_trust'):
            self.hybrid_trust.n_peers = value
        if hasattr(self, 'counterfactual_eval'):
            self.counterfactual_eval.n_peers = value
    
    def _create_feature_extractor(self):
        """Create environment-specific feature extractor"""
        if 'Lunar' in self.env_name:
            return LunarLanderFeatureExtractor(n_peers=4)
        elif 'Hopper' in self.env_name:
            return HopperFeatureExtractor(n_peers=4)
        elif 'Walker' in self.env_name:
            return Walker2DFeatureExtractor(n_peers=4)
        elif 'Ant' in self.env_name:
            return AntFeatureExtractor(n_peers=4)
        elif 'Room' in self.env_name:
            return RoomFeatureExtractor(n_peers=4)
        else:
            raise ValueError(f"Unknown environment: {self.env_name}")
    
    def _create_transformer(self):
        """Create transformer (phase-aware or basic)"""
        if self.use_phase_aware:
            return PhaseAwareTransformer(
                feature_dim=self.env_config.feature_dim,
                n_phases=len(self.env_config.phases),
                phase_names=self.env_config.phases,
                d_model=self.env_config.d_model,
                nhead=self.env_config.nhead,
                num_layers=self.env_config.num_layers,
                max_seq_len=self.env_config.max_seq_len
            )
        else:
            return AttentionTransformer(
                feature_dim=self.env_config.feature_dim,
                d_model=self.env_config.d_model,
                nhead=self.env_config.nhead,
                num_layers=self.env_config.num_layers,
                max_seq_len=self.env_config.max_seq_len
            )
    
    def _get_phase_detector(self):
        """Get rule-based phase detector for environment"""
        if 'Lunar' in self.env_name:
            return PhaseDetector.detect_lunar_phase
        elif 'Hopper' in self.env_name:
            return PhaseDetector.detect_hopper_phase
        elif 'Walker' in self.env_name:
            return PhaseDetector.detect_walker2d_phase
        elif 'Room' in self.env_name:
            return self._detect_room_phase
        else:
            return lambda obs: 0
    
    def _detect_room_phase(self, obs: np.ndarray) -> int:
        """
        Detect phase for Room navigation environment
        
        Phases:
        0: exploration - far from goal (distance > 0.5)
        1: navigation - moving towards goal (0.2 < distance <= 0.5)
        2: approaching - close to goal (0.05 < distance <= 0.2)
        3: goal_reached - at goal (distance <= 0.05)
        
        Args:
            obs: Room observation (agent_pos, goal_pos concatenated)
        
        Returns:
            Phase index (0-3)
        """
        if obs is None or len(obs) < 4:
            return 0  # Default to exploration
        
        # Parse observation
        n_dims = len(obs) // 2
        agent_pos = obs[:n_dims]
        goal_pos = obs[n_dims:]
        
        # Calculate distance to goal
        distance = np.linalg.norm(agent_pos - goal_pos)
        
        # Determine phase based on distance thresholds
        if distance <= 0.05:
            return 3  # goal_reached
        elif distance <= 0.2:
            return 2  # approaching
        elif distance <= 0.5:
            return 1  # navigation
        else:
            return 0  # exploration
    
    def extract_features_and_phase(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        peer_id: int,
        reward: float,
        next_obs: np.ndarray = None
    ) -> Tuple[np.ndarray, int]:
        """
        Extract features and detect phase
        
        Returns:
            features: Feature vector
            phase: Phase index
        """
        # Extract features
        features = self.feature_extractor.extract_features(
            obs, action, peer_id, reward, next_obs
        )
        
        # Detect phase
        phase = self.phase_detector(obs)
        
        return features, phase
    
    def update_attention_trust(self, obs: np.ndarray, action: np.ndarray, 
                              reward: float, next_obs: np.ndarray, peer_id: int):
        """
        Update attention-specific components after each step
        
        This method ONLY handles:
        1. Feature extraction and buffer filling
        2. Transformer training
        
        Traditional trust is updated separately in _update_trust() via buffer sampling (like baseline)
        
        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            peer_id: Peer who suggested the action
        """
        if not hasattr(self, '_attention_trust_update_count'):
            self._attention_trust_update_count = 0
        self._attention_trust_update_count += 1
        
        features, phase = self.extract_features_and_phase(
            obs, action, peer_id, reward, next_obs
        )
        
        self.history_buffer.add(features, phase, reward, peer_id)
        
        self.attention_step_count += 1
        
        if self.attention_step_count % self.attention_update_freq == 0:
            self._train_transformer()
    
    def _train_transformer(self):
        """Train transformer on sampled sequences"""
        batch = self.history_buffer.sample_sequence(batch_size=32)
        
        if batch is None:
            return
        
        device = next(self.transformer.parameters()).device
        features = batch['features'].to(device)
        phases = batch['phases'].to(device)
        rewards = batch['rewards'].to(device)
        mask = batch['mask'].to(device)
        
        if self.use_phase_aware:
            outcome_pred, phase_logits, _ = self.transformer(features, phases, mask)
            
            loss_dict = self.transformer.compute_loss(
                outcome_pred,
                rewards.unsqueeze(-1),
                phase_logits,
                phases,
                mask
            )
            loss = loss_dict['total_loss']
            
            if not hasattr(self, '_transformer_update_count'):
                self._transformer_update_count = 0
            self._transformer_update_count += 1
            
            self._last_transformer_loss = {
                'total': loss.item(),
                'outcome': loss_dict.get('outcome_loss', 0),
                'phase': loss_dict.get('phase_loss', 0),
                'pred_min': outcome_pred.min().item(),
                'pred_max': outcome_pred.max().item()
            }
        else:
            outcome_pred, _ = self.transformer(features, mask)
            loss = nn.functional.mse_loss(outcome_pred.squeeze(), rewards)
            
            if not hasattr(self, '_transformer_update_count'):
                self._transformer_update_count = 0
            self._transformer_update_count += 1
            
            self._last_transformer_loss = {
                'total': loss.item(),
                'pred_min': outcome_pred.min().item(),
                'pred_max': outcome_pred.max().item()
            }
        
        self.hybrid_trust.update_confidence(loss.item())
        
        self.transformer_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 1.0)
        self.transformer_optimizer.step()
    
    def _update_trust(self, batch_size=10):
        """
        Override base trust update to use hybrid trust values
        
        For attention peers (matching baseline flow):
        1. Sample from buffer and update traditional trust via TD learning (like baseline)
        2. Compute attention-based trust from transformer
        3. Combine using hybrid integration
        4. Sync to self.trust_values for compatibility
        
        This is called every step via _on_step() -> peer_value_functions["trust"]()
        """
        # Debug: Track trust update calls
        if not hasattr(self, '_trust_update_counter'):
            self._trust_update_counter = 0
        self._trust_update_counter += 1
        
        # === STEP 1: Update traditional trust via buffer sampling (LIKE BASELINE) ===
        if self.use_buffer_for_trust:
            batch = self.buffer.sample(batch_size)
        else:
            batch = self.buffer.latest()
            batch_size = 1
        
        if batch is not None:  # buffer sufficiently full
            # next observations
            obs = np.array([b[3] for b in batch]).reshape(batch_size, -1)
            v = self.value(obs)

            if self.group.use_advantage:
                # previous observations
                prev_obs = np.array([b[4] for b in batch]).reshape(batch_size, -1)
                prev_v = self.value(prev_obs)
            else:
                prev_v = np.zeros_like(v)

            # Compute TD targets for each peer (like baseline)
            for i in range(batch_size):
                # Skip if peer index is None
                if batch[i][2] is None:
                    continue
                    
                # Ensure scalar values
                v_value = v[i].item() if hasattr(v[i], 'item') else float(v[i])
                prev_v_value = prev_v[i].item() if hasattr(prev_v[i], 'item') else float(prev_v[i])
                reward = batch[i][0]
                peer_id = int(batch[i][2]) if not isinstance(batch[i][2], int) else batch[i][2]
                
                # TD target: (r + γ*V(s')) - V(s) if advantage, else r + γ*V(s')
                td_target = (reward + self.gamma * v_value) - prev_v_value
                
                # Update traditional trust for this peer
                self.hybrid_trust.update_traditional_trust(peer_id, td_target)
        
        # === STEP 2: Compute attention-based trust ===
        attention_values = self.get_attention_trust_values()
        
        # === STEP 3: Compute hybrid trust ===
        # Update shared step counter with actual timesteps for warmup calculation
        if hasattr(self, 'num_timesteps'):
            self.hybrid_trust.set_current_step(self.num_timesteps)
        # Pass update counter as increment_id to prevent duplicate step counting
        hybrid_trusts = self.hybrid_trust.compute_all_final_trusts(attention_values, increment_id=self._trust_update_counter)
        
        # === STEP 4: Sync to self.trust_values for compatibility ===
        for peer_id in range(self.n_peers):
            if peer_id in hybrid_trusts:
                trust_val = hybrid_trusts[peer_id]
                if hasattr(trust_val, 'item'):
                    trust_val = trust_val.item()
                self.trust_values[peer_id] = float(trust_val)
        
        # Also update peer_values dict
        if hasattr(self, 'peer_values') and 'trust' in self.peer_values:
            self.peer_values['trust'] = self.trust_values
        
        # Periodic debug logging - every 5000 steps
        # This prints once per group, not once per peer
        shared_step = self.num_timesteps if hasattr(self, 'num_timesteps') else 0
        if shared_step % 5000 == 0 and shared_step > 0:
            # Only print if this is the first time we hit this milestone
            if not hasattr(self, '_last_logged_step'):
                self._last_logged_step = -1
            
            if shared_step != self._last_logged_step:
                self._last_logged_step = shared_step
                trad_trusts = {i: self.hybrid_trust.traditional_trust.get_trust(i) for i in range(self.n_peers)}
                omega_t = self.hybrid_trust.get_current_attention_weight()
                warmup_progress = min(1.0, shared_step / self.hybrid_trust.warmup_steps)
                followed = getattr(self, 'followed_peer', None)
                
                print(f"\n📊 GROUP Trust Status [Step {shared_step}]:", flush=True)
                print(f"   Traditional trusts: {trad_trusts}", flush=True)
                print(f"   Attention trusts (unbounded): {attention_values}", flush=True)
                print(f"   Hybrid trusts: {hybrid_trusts}", flush=True)
                print(f"   omega_t: {omega_t:.4f}, warmup: {warmup_progress:.2%}", flush=True)
                print(f"   Followed peer: {followed}", flush=True)
                print(f"   History buffer: {self.history_buffer.current_size}, Transformer updates: {getattr(self, '_transformer_update_count', 0)}", flush=True)
                
                # Show transformer training metrics if available
                if hasattr(self, '_last_transformer_loss'):
                    loss_info = self._last_transformer_loss
                    if 'outcome' in loss_info:
                        # Phase-aware transformer
                        print(f"   🧠 Transformer - Total: {loss_info['total']:.4f}, Outcome: {loss_info['outcome']:.4f}, Phase: {loss_info['phase']:.4f}", flush=True)
                    else:
                        # Standard transformer
                        print(f"   🧠 Transformer Loss: {loss_info['total']:.4f}", flush=True)
                    print(f"   Pred range: [{loss_info['pred_min']:.3f}, {loss_info['pred_max']:.3f}]\n", flush=True)
                else:
                    print("", flush=True)
    
    def get_attention_trust_values(self) -> Dict[int, float]:
        """
        Get current attention-based trust values for all peers
        Uses counterfactual evaluation
        
        Returns:
            Dictionary mapping peer_id to attention trust value
        """
        # Get recent history
        if self.history_buffer.current_size < self.sequence_length:
            return {i: 0.0 for i in range(self.n_peers)}
        
        # Sample recent sequence
        recent_features = [
            self.history_buffer.features[-self.sequence_length + i] 
            for i in range(self.sequence_length)
        ]
        recent_features = np.stack(recent_features)
        
        # Get phase indices if using phase-aware transformer
        recent_phases = None
        if self.use_phase_aware and self.history_buffer.current_size >= self.sequence_length:
            recent_phases = [
                self.history_buffer.phases[-self.sequence_length + i]
                for i in range(self.sequence_length)
            ]
            recent_phases = np.array(recent_phases)
        
        # Evaluate each peer using counterfactual
        raw_predictions = []
        
        # Determine peer feature indices based on environment
        # Peer features are always one-hot encoded for n_peers dimensions (4 dims for 4 peers)
        # They are typically the last features in the extracted feature vector
        feature_dim = self.env_config.feature_dim
        peer_feature_dim = 4  # One-hot for 4 peers
        peer_start_idx = feature_dim - peer_feature_dim
        peer_indices = (peer_start_idx, feature_dim)
        
        for peer_id in range(self.n_peers):
            trust_value = self.counterfactual_eval.evaluate_counterfactual_trust(
                self.transformer,
                recent_features,
                peer_id,
                peer_indices,
                phase_indices=recent_phases
            )
            raw_predictions.append(trust_value)
        
        # Convert to dictionary WITHOUT clipping
        # Normalization happens at peer selection time (like baseline)
        # This allows negative values for bad peers and large values for good peers
        attention_values = {
            i: float(raw_predictions[i]) 
            for i in range(self.n_peers)
        }
        
        return attention_values
    
    def get_hybrid_trust_values(self) -> Dict[int, float]:
        """
        Get final hybrid trust values for all peers
        
        Returns:
            Dictionary of final trust values
        """
        # Get attention values
        attention_values = self.get_attention_trust_values()
        
        # Compute hybrid trust (no increment_id since this is just a getter)
        final_trusts = self.hybrid_trust.compute_all_final_trusts(attention_values, increment_id=None)
        
        return final_trusts
    
    def _predict_train(self, observation, state=None, episode_start=None, deterministic=False):
        """
        Action selection during training - uses peer suggestions
        
        This method is critical for peer learning to work. The baseline Peer.learn() method
        replaces self.predict with this method during training, so that SAC calls this
        instead of the default predict(), enabling peer-based action selection.
        
        Args:
            observation: Current observation
            state: RNN state (not used)
            episode_start: Episode start flag (not used)
            deterministic: Whether to use deterministic actions
        
        Returns:
            Action selected from peers, state (None)
        """
        if deterministic:
            # For deterministic evaluation, use own policy
            return self.policy.predict(observation, state=state,
                                      episode_start=episode_start,
                                      deterministic=deterministic)
        else:
            # During training, use peer suggestions via get_action
            # This will set self.followed_peer and enable trust updates
            action, state = self.get_action(observation)
            
            # Track predictions
            if not hasattr(self, '_predict_train_count'):
                self._predict_train_count = 0
            self._predict_train_count += 1
            
            return action, state
    
    def _sample_action(self, learning_starts: int, action_noise=None, n_envs: int = 1):
        """
        Override SAC's _sample_action to enable peer learning during warmup phase
        
        SAC normally uses random actions before learning_starts steps, but we want
        peer learning to work from the beginning (like the baseline implementation).
        
        Returns:
            action: Action to execute in environment
            buffer_action: Scaled action to store in replay buffer
        """
        # Always use peer suggestions (via predict method) unless in solo epoch
        # The baseline Peer.learn() swaps self.predict = self._predict_train during peer epochs
        if hasattr(self, 'group') and self.group is not None and not self.group.solo_epoch:
            # Use peer learning (calls self.predict which is swapped to _predict_train)
            assert self._last_obs is not None, "self._last_obs was not set"
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)
        else:
            # Solo epoch or no group: use parent's default behavior
            return super()._sample_action(learning_starts, action_noise, n_envs)
        
        # Handle action scaling (copied from SAC's _sample_action)
        from gymnasium import spaces
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)
            
            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)
            
            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        
        return action, buffer_action
    
    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos):
        """
        Override to add attention-specific features to history buffer
        
        This is called after every environment step during training.
        Note: Traditional trust is updated in _update_trust() via buffer sampling (like baseline)
        """
        # Debug: Track if this method is called
        if not hasattr(self, '_store_transition_count'):
            self._store_transition_count = 0
        self._store_transition_count += 1
        
        # Get old observation before calling parent
        old_obs = self._last_obs
        
        # Call parent to handle base transition storage AND buffer.add()
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)
        
        # Add attention-specific features to history buffer (for transformer training)
        # Only during peer learning epochs AND when following a peer
        if hasattr(self, 'group') and self.group is not None and not self.group.solo_epoch:
            if hasattr(self, 'followed_peer') and self.followed_peer is not None:
                # Ensure reward is scalar
                reward_scalar = float(reward[0]) if hasattr(reward, '__len__') else float(reward)
                
                # Update attention-specific components (features, transformer)
                self.update_attention_trust(
                    obs=old_obs[0] if len(old_obs.shape) > 1 else old_obs,
                    action=buffer_action[0] if len(buffer_action.shape) > 1 else buffer_action,
                    reward=reward_scalar,
                    next_obs=new_obs[0] if len(new_obs.shape) > 1 else new_obs,
                    peer_id=self.followed_peer
                )
    
    def get_trust_data(self) -> dict:
        """
        Get trust data in unified format for analysis
        
        Returns:
            Dictionary with all trust components for TrustAnalyzer
        """
        # Get attention values
        attention_values = self.get_attention_trust_values()
        
        # Compute hybrid trust (no increment - this is read-only)
        final_trusts = self.hybrid_trust.compute_all_final_trusts(attention_values, increment_id=None)
        
        # Get traditional trust values
        traditional_trusts = {i: self.hybrid_trust.traditional_trust.get_trust(i) 
                             for i in range(self.n_peers)}
        
        # Get confidence and omega_t
        confidence = self.hybrid_trust.get_current_confidence()
        omega_t = self.hybrid_trust.get_current_attention_weight()
        
        return {
            'final_trust': final_trusts,
            'attention_trust': attention_values,
            'traditional_trust': traditional_trusts,
            'confidence': confidence,
            'omega_t': omega_t
        }


# Create concrete classes

class AttentionSACPeer(AttentionPeerMixin, make_peer_class(SAC)):
    """
    SAC Peer with Attention-based Trust
    For continuous control (Hopper-v4, Walker2d-v4)
    """
    
    def __init__(self, env_name: str, **kwargs):
        super().__init__(env_name=env_name, **kwargs)


class AttentionDQNPeer(AttentionPeerMixin, make_peer_class(DQN)):
    """
    DQN Peer with Attention-based Trust
    For discrete control (LunarLander-v2)
    """
    
    def __init__(self, env_name: str, **kwargs):
        super().__init__(env_name=env_name, **kwargs)
