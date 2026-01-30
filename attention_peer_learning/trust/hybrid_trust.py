"""
Hybrid Trust System
Combines traditional TD-based trust with attention-based trust evaluation

Key Components:
1. Traditional Trust (v_trad): TD learning-based peer value estimation
2. Attention Trust (v_attn): Transformer-based peer quality prediction  
3. Confidence Metric (c_t): Adaptive weight based on model confidence
4. Hybrid Integration: Normalize each component THEN combine

Equations:
- Traditional Update: v_trad_new = v_trad + α*(TD_target - v_trad)
- Confidence: c_t = 0.7*sigmoid(-ΔL̄) + 0.3/(1+std(L_history))
- Hybrid Weight: ω_t = warmup_progress * max_weight * c_t
- Normalization: norm(x) = x / max(|x_all|)  (per component)
- Final Trust: v_final_j = ω_t*norm(v_attn_j) + (1-ω_t)*norm(v_trad_j)

Critical Design:
Each component is normalized SEPARATELY before combining, matching the baseline's
approach where trust, critic, and agent_values are each normalized individually
then summed. This ensures proper weighting proportional to ω_t.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, List
from collections import deque


class TraditionalTrust:
    """
    Traditional TD-based trust value tracking
    Uses temporal difference learning to update peer trust values
    """
    
    def __init__(
        self,
        n_peers: int = 4,
        learning_rate: float = 0.05,
        initial_value: float = 0.0
    ):
        """
        Initialize traditional trust tracker
        
        Args:
            n_peers: Number of peers
            learning_rate: TD learning rate (α)
            initial_value: Initial trust value for all peers
        """
        self.n_peers = n_peers
        self.learning_rate = learning_rate
        
        # Trust values for each peer
        self.trust_values: Dict[int, float] = {
            i: initial_value for i in range(n_peers)
        }
        
        # Track update history
        self.update_history: Dict[int, deque] = {
            i: deque(maxlen=100) for i in range(n_peers)
        }
    
    def update(
        self,
        peer_id: int,
        td_target: float,
        current_value: Optional[float] = None
    ) -> float:
        """
        Update trust value using TD learning
        
        v_trad_new = v_trad + α * (TD_target - v_trad)
        
        Args:
            peer_id: Peer whose trust to update
            td_target: TD target (reward + γ*V(s'))
            current_value: Current state value (if None, uses stored trust)
        
        Returns:
            Updated trust value
        """
        if peer_id not in self.trust_values:
            self.trust_values[peer_id] = 0.0
        
        # Get current trust value
        current_trust = current_value if current_value is not None else self.trust_values[peer_id]
        
        # Ensure td_target is scalar
        if hasattr(td_target, 'item'):
            td_target = float(td_target.item())
        elif hasattr(td_target, '__len__') and len(td_target) > 0:
            td_target = float(td_target[0])
        else:
            td_target = float(td_target)
        
        # TD error
        td_error = td_target - float(current_trust)
        
        # Update
        new_trust = float(current_trust) + self.learning_rate * td_error
        
        # Store
        self.trust_values[peer_id] = new_trust
        self.update_history[peer_id].append(td_error)
        
        return new_trust
    
    def get_trust(self, peer_id: int) -> float:
        """Get current trust value for peer"""
        return float(self.trust_values.get(peer_id, 0.0))
    
    def get_all_trusts(self) -> Dict[int, float]:
        """Get all peer trust values"""
        return self.trust_values.copy()
    
    def reset_peer(self, peer_id: int, value: float = 0.0):
        """Reset trust value for specific peer"""
        self.trust_values[peer_id] = value
        self.update_history[peer_id].clear()
    
    def reset_all(self, value: float = 0.0):
        """Reset all trust values"""
        for peer_id in range(self.n_peers):
            self.reset_peer(peer_id, value)


class ConfidenceMetric:
    """
    Confidence metric for adaptive weighting of attention vs traditional trust
    
    Equation: c_t = 0.7*sigmoid(-ΔL̄) + 0.3/(1+std(L_history))
    
    Components:
    - Loss gradient: sigmoid(-ΔL̄) - high when loss is decreasing
    - Loss stability: 1/(1+std(L)) - high when loss is stable
    """
    
    def __init__(
        self,
        history_length: int = 50,
        gradient_weight: float = 0.7,
        stability_weight: float = 0.3
    ):
        """
        Initialize confidence metric
        
        Args:
            history_length: Number of recent losses to track
            gradient_weight: Weight for gradient component (default: 0.7)
            stability_weight: Weight for stability component (default: 0.3)
        """
        self.history_length = history_length
        self.gradient_weight = gradient_weight
        self.stability_weight = stability_weight
        
        # Loss history
        self.loss_history: deque = deque(maxlen=history_length)
    
    def update(self, loss: float):
        """Update with new loss value"""
        self.loss_history.append(loss)
    
    def compute(self) -> float:
        """
        Compute confidence metric
        
        c_t = 0.7*sigmoid(-ΔL̄) + 0.3/(1+std(L_history))
        
        Returns:
            Confidence value in [0, 1]
        """
        if len(self.loss_history) < 2:
            return 0.5  # Default moderate confidence
        
        # Convert to numpy for computation
        losses = np.array(self.loss_history)
        
        # 1. Loss gradient component
        # Compute recent loss change (ΔL̄)
        recent_window = min(10, len(losses) // 2)
        recent_losses = losses[-recent_window:]
        older_losses = losses[-2*recent_window:-recent_window] if len(losses) >= 2*recent_window else losses[:recent_window]
        
        delta_loss = np.mean(recent_losses) - np.mean(older_losses)
        
        # Sigmoid of negative delta (high when loss decreasing)
        gradient_component = 1.0 / (1.0 + np.exp(delta_loss))  # sigmoid(-ΔL)
        
        # 2. Loss stability component
        loss_std = np.std(losses)
        stability_component = 1.0 / (1.0 + loss_std)
        
        # 3. Combined confidence
        confidence = (
            self.gradient_weight * gradient_component +
            self.stability_weight * stability_component
        )
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def get_gradient_component(self) -> float:
        """Get just the gradient component (for logging)"""
        if len(self.loss_history) < 2:
            return 0.5
        
        losses = np.array(self.loss_history)
        recent_window = min(10, len(losses) // 2)
        recent_losses = losses[-recent_window:]
        older_losses = losses[-2*recent_window:-recent_window] if len(losses) >= 2*recent_window else losses[:recent_window]
        delta_loss = np.mean(recent_losses) - np.mean(older_losses)
        
        return float(1.0 / (1.0 + np.exp(delta_loss)))
    
    def get_stability_component(self) -> float:
        """Get just the stability component (for logging)"""
        if len(self.loss_history) < 2:
            return 0.5
        
        loss_std = np.std(self.loss_history)
        return float(1.0 / (1.0 + loss_std))
    
    def reset(self):
        """Reset history"""
        self.loss_history.clear()


class HybridTrustIntegrator:
    """
    Integrates traditional and attention-based trust values
    
    Final Trust Equation:
    v_final_j = ω_t * norm(v_attn_j) + (1 - ω_t) * norm(v_trad_j)
    
    Where:
    - norm(x) = x / max(|x_all|) - normalization across all peers
    - v_attn_j: Attention-based trust for peer j (unbounded)
    - v_trad_j: Traditional TD-based trust for peer j (unbounded)
    - ω_t: Attention weight (scales with confidence and warmup)
    - c_t: Confidence metric
    
    Critical: Each component is normalized SEPARATELY before combining.
    This matches baseline's approach where each peer_values component is
    normalized individually then added together. Ensures proper weighting.
    """
    
    # Class variable - shared step counter across all peers
    _shared_step = 0
    _last_increment_id = -1  # Track last unique increment to prevent duplicates
    
    def __init__(
        self,
        n_peers: int = 4,
        trust_scale: float = 100.0,
        attention_weight_coef: float = 0.3,
        warmup_steps: int = 500000,
        max_attention_weight: float = 0.5,
        initial_trust_value: float = 0.0
    ):
        """
        Initialize hybrid trust integrator
        
        Args:
            n_peers: Number of peers
            trust_scale: DEPRECATED - kept for backward compatibility, not used in calculations
            attention_weight_coef: Coefficient for attention weight (default: 0.3)
            warmup_steps: Steps for curriculum warmup (default: 500000)
            max_attention_weight: Maximum attention weight after warmup (default: 0.5)
            initial_trust_value: Initial trust value for all peers (default: 0.0, matches baseline)
        
        Note:
            Trust values are NOT normalized here. Normalization happens at peer selection
            using values / np.max(np.abs(values)), matching baseline behavior.
            This allows trust values to be unbounded and properly represent bad (negative)
            and good (large positive) peers.
        """
        self.n_peers = n_peers
        self.trust_scale = trust_scale  # Kept for backward compatibility, not used in calculations
        self.attention_weight_coef = attention_weight_coef
        self.warmup_steps = warmup_steps
        self.max_attention_weight = max_attention_weight
        
        # Components
        self.traditional_trust = TraditionalTrust(n_peers, initial_value=initial_trust_value)
        self.confidence_metric = ConfidenceMetric()
        
        # Track final trust values (initialized with initial value)
        self.final_trust_values: Dict[int, float] = {i: initial_trust_value for i in range(n_peers)}
    
    def set_current_step(self, step: int):
        """Set current training step (used for warmup calculation)"""
        HybridTrustIntegrator._shared_step = step
    
    def get_attention_weight_coef(self) -> float:
        """
        Compute attention weight with curriculum learning
        
        Gradually increases from 0.1 to max_attention_weight over warmup_steps.
        Then scales by confidence.
        
        Returns:
            Attention weight coefficient with warmup and confidence scaling
        """
        # Calculate warmup progress using shared step counter
        if HybridTrustIntegrator._shared_step == 0:
            return 0.0  # Start with pure traditional trust (matches baseline init=200)
        
        progress = min(1.0, HybridTrustIntegrator._shared_step / self.warmup_steps)
        
        # Interpolate from 0.0 to max_attention_weight
        warmup_coef = self.max_attention_weight * progress
        
        # Scale by confidence
        confidence = self.confidence_metric.compute()
        
        return warmup_coef * confidence
    
    def compute_final_trust(
        self,
        peer_id: int,
        attention_value: float,
        traditional_value: Optional[float] = None,
        increment_step: bool = True
    ) -> float:
        """
        Compute final hybrid trust value for a single peer
        
        WARNING: This method does NOT normalize before combining.
        For proper normalization, use compute_all_final_trusts() instead.
        This method is kept for backward compatibility.
        
        Args:
            peer_id: Peer to compute trust for
            attention_value: Attention-based trust (v_attn)
            traditional_value: Traditional trust (if None, uses stored)
            increment_step: Whether to increment step counter (deprecated)
        
        Returns:
            Final hybrid trust value
        
        Note:
            This method combines raw values without normalization.
            Use compute_all_final_trusts() for proper normalized combination.
        """
        # Step counter is now shared at class level and incremented in compute_all_final_trusts
        # This parameter is kept for backward compatibility but not used
        
        # Get traditional trust
        if traditional_value is None:
            traditional_value = self.traditional_trust.get_trust(peer_id)
        
        # Compute attention weight with curriculum
        omega_t = self.get_attention_weight_coef()
        
        # Hybrid integration (both components at natural scale)
        # Note: No pre-scaling needed - normalization happens at peer selection
        # via values / np.max(np.abs(values)), which balances scales automatically
        final_trust = omega_t * attention_value + (1.0 - omega_t) * traditional_value
        
        # Ensure scalar conversion (handle numpy arrays and torch tensors)
        if hasattr(final_trust, 'item'):
            final_trust = final_trust.item()
        final_trust = float(final_trust)
        
        # Store
        self.final_trust_values[peer_id] = final_trust
        
        return final_trust
    
    def compute_all_final_trusts(
        self,
        attention_values: Dict[int, float],
        increment_id: int = None
    ) -> Dict[int, float]:
        """
        Compute final trust for all peers
        
        CRITICAL: Normalizes each component SEPARATELY before combining.
        This matches baseline approach and ensures proper weighting.
        
        Args:
            attention_values: Dictionary of attention-based trust values
            increment_id: Unique ID for this call (prevents duplicate increments)
        
        Returns:
            Dictionary of final hybrid trust values
        """
        # Increment shared step counter only if this is a new unique call
        if increment_id is not None and increment_id != HybridTrustIntegrator._last_increment_id:
            HybridTrustIntegrator._shared_step += 1
            HybridTrustIntegrator._last_increment_id = increment_id
        
        # Get all trust values as arrays
        trad_array = np.array([self.traditional_trust.get_trust(i) for i in range(self.n_peers)])
        attn_array = np.array([attention_values.get(i, 0.0) for i in range(self.n_peers)])
        
        # Normalize EACH component separately (matching baseline approach)
        # This ensures both components contribute proportionally to omega_t
        max_trad = np.max(np.abs(trad_array))
        max_attn = np.max(np.abs(attn_array))
        
        # Avoid division by zero
        trad_norm = trad_array / max_trad if max_trad > 0 else trad_array
        attn_norm = attn_array / max_attn if max_attn > 0 else attn_array
        
        # Get attention weight with curriculum
        omega_t = self.get_attention_weight_coef()
        
        # Hybrid combination: omega_t * norm(attention) + (1-omega_t) * norm(traditional)
        # Now attention contributes exactly omega_t fraction, traditional (1-omega_t) fraction
        hybrid_array = omega_t * attn_norm + (1.0 - omega_t) * trad_norm
        
        # Convert to dictionary and store
        final_trusts = {}
        for pid in range(self.n_peers):
            trust_val = float(hybrid_array[pid])
            self.final_trust_values[pid] = trust_val
            final_trusts[pid] = trust_val
        
        return final_trusts
    
    def update_traditional_trust(
        self,
        peer_id: int,
        td_target: float
    ) -> float:
        """Update traditional trust component"""
        return self.traditional_trust.update(peer_id, td_target)
    
    def update_confidence(self, loss: float):
        """Update confidence metric with new loss"""
        self.confidence_metric.update(loss)
    
    def get_current_confidence(self) -> float:
        """Get current confidence value"""
        return self.confidence_metric.compute()
    
    def get_current_attention_weight(self) -> float:
        """Get current attention weight ω_t"""
        return self.get_attention_weight_coef()
    
    def get_current_step(self) -> int:
        """Get current shared step counter"""
        return HybridTrustIntegrator._shared_step
    
    def get_trust_components(self, peer_id: int) -> Dict[str, float]:
        """
        Get all trust components for logging/debugging
        
        Returns:
            Dictionary with traditional, attention_weight, confidence, final
        """
        confidence = self.confidence_metric.compute()
        omega_t = self.attention_weight_coef * confidence
        
        return {
            'traditional': self.traditional_trust.get_trust(peer_id),
            'final': self.final_trust_values.get(peer_id, 0.0),
            'attention_weight': omega_t,
            'confidence': confidence,
            'gradient_component': self.confidence_metric.get_gradient_component(),
            'stability_component': self.confidence_metric.get_stability_component()
        }
    
    def reset(self):
        """Reset all components"""
        self.traditional_trust.reset_all()
        self.confidence_metric.reset()
        self.final_trust_values = {i: 0.0 for i in range(self.n_peers)}
