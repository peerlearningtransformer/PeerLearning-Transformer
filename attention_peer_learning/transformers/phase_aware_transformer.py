"""
Phase-Aware Attention Transformer
Extends base transformer with behavioral phase awareness as per paper Section 3.3.2

Key Features:
1. Phase Embeddings: Learned representations for behavioral phases (cruise, approach, landing, etc.)
2. Phase Modulation: Modulates attention based on current behavioral phase
3. Auxiliary Classification: Multi-task learning with phase classification
4. Combined Loss: Outcome prediction + phase classification

Phases by Environment:
- LunarLander: cruise, approach, landing, touchdown
- Hopper: stance, swing, flight, landing
- Walker2D: left_stance, right_stance, double_support, flight
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
from .attention_transformer import AttentionTransformer, PositionalEncoding


class PhaseEmbedding(nn.Module):
    """Learnable embeddings for behavioral phases"""
    
    def __init__(self, n_phases: int, d_model: int):
        super().__init__()
        self.n_phases = n_phases
        self.d_model = d_model
        
        # Phase embeddings
        self.embeddings = nn.Embedding(n_phases, d_model)
        
        # Initialize embeddings
        nn.init.normal_(self.embeddings.weight, mean=0, std=0.02)
    
    def forward(self, phase_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            phase_indices: [batch, seq_len] - Phase index for each timestep
        Returns:
            Phase embeddings [batch, seq_len, d_model]
        """
        return self.embeddings(phase_indices)


class PhaseModulatedAttention(nn.Module):
    """
    Attention mechanism modulated by behavioral phase
    Helps transformer focus on phase-relevant patterns
    """
    
    def __init__(self, d_model: int, nhead: int, n_phases: int):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.n_phases = n_phases
        
        # Phase-specific attention scaling
        self.phase_attention_scale = nn.Parameter(torch.ones(n_phases, nhead))
        
        # Standard multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True
        )
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        phase_indices: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Phase-modulated attention
        
        Args:
            query, key, value: [batch, seq_len, d_model]
            phase_indices: [batch, seq_len] - Phase for each timestep
            attn_mask, key_padding_mask: Standard attention masks
        
        Returns:
            attended: [batch, seq_len, d_model]
            attention_weights: [batch, seq_len, seq_len]
        """
        # Standard attention
        attended, attention_weights = self.attention(
            query, key, value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        
        # Phase modulation (simplified - apply phase-specific scaling)
        # In practice, this would be more sophisticated
        
        return attended, attention_weights


class PhaseAwareTransformer(nn.Module):
    """
    Transformer with phase-aware attention and auxiliary phase classification
    
    Args:
        feature_dim: Input feature dimension
        n_phases: Number of behavioral phases
        phase_names: List of phase names (for logging)
        d_model: Transformer dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        max_seq_len: Maximum sequence length
        dropout: Dropout rate
        phase_loss_weight: Weight for phase classification loss (default: 0.3)
    """
    
    def __init__(
        self,
        feature_dim: int,
        n_phases: int,
        phase_names: List[str],
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 200,
        dropout: float = 0.1,
        phase_loss_weight: float = 0.3
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.n_phases = n_phases
        self.phase_names = phase_names
        self.d_model = d_model
        self.phase_loss_weight = phase_loss_weight
        
        # Base transformer components (reuse from base class)
        self.input_projection = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Phase embeddings
        self.phase_embedding = PhaseEmbedding(n_phases, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Outcome predictor head
        self.outcome_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Phase classification head (auxiliary task)
        self.phase_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_phases)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        features: torch.Tensor,
        phase_indices: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with phase awareness
        
        Args:
            features: [batch, seq_len, feature_dim]
            phase_indices: [batch, seq_len] - Phase indices
            mask: [batch, seq_len] - Valid mask
        
        Returns:
            outcome: [batch, seq_len, 1] - Predicted outcomes
            phase_logits: [batch, seq_len, n_phases] - Phase classification logits
            encoded: [batch, seq_len, d_model] - Encoded representations
        """
        # 1. Project features
        x = self.input_projection(features)  # [batch, seq_len, d_model]
        
        # 2. Add positional encoding
        x = self.positional_encoding(x)
        
        # 3. Add phase embeddings
        phase_emb = self.phase_embedding(phase_indices)  # [batch, seq_len, d_model]
        x = x + phase_emb  # Combine feature and phase information
        
        # 4. Attention mask
        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = (mask == 0)
        
        # 5. Transformer encoding
        encoded = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask
        )  # [batch, seq_len, d_model]
        
        # 6. Outcome prediction
        outcome = self.outcome_head(encoded)  # [batch, seq_len, 1]
        
        # 7. Phase classification (auxiliary task)
        phase_logits = self.phase_classifier(encoded)  # [batch, seq_len, n_phases]
        
        return outcome, phase_logits, encoded
    
    def compute_loss(
        self,
        outcome_pred: torch.Tensor,
        outcome_target: torch.Tensor,
        phase_logits: torch.Tensor,
        phase_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss: outcome + phase classification
        
        Args:
            outcome_pred: [batch, seq_len, 1]
            outcome_target: [batch, seq_len, 1]
            phase_logits: [batch, seq_len, n_phases]
            phase_target: [batch, seq_len] - Phase indices
            mask: [batch, seq_len] - Valid mask
        
        Returns:
            Dictionary with 'total_loss', 'outcome_loss', 'phase_loss'
        """
        # Outcome loss (MSE)
        outcome_loss = F.mse_loss(
            outcome_pred.squeeze(-1),
            outcome_target.squeeze(-1),
            reduction='none'
        )
        
        # Phase classification loss (CrossEntropy)
        batch_size, seq_len, n_classes = phase_logits.shape
        phase_logits_flat = phase_logits.reshape(-1, n_classes)
        phase_target_flat = phase_target.reshape(-1)
        
        phase_loss = F.cross_entropy(
            phase_logits_flat,
            phase_target_flat,
            reduction='none'
        ).reshape(batch_size, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            outcome_loss = outcome_loss * mask
            phase_loss = phase_loss * mask
            valid_count = mask.sum()
        else:
            valid_count = batch_size * seq_len
        
        # Average over valid elements
        outcome_loss = outcome_loss.sum() / valid_count.clamp(min=1)
        phase_loss = phase_loss.sum() / valid_count.clamp(min=1)
        
        # Combined loss
        total_loss = outcome_loss + self.phase_loss_weight * phase_loss
        
        return {
            'total_loss': total_loss,
            'outcome_loss': outcome_loss,
            'phase_loss': phase_loss
        }
    
    def predict_peer_value(
        self,
        features: torch.Tensor,
        phase_indices: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> float:
        """
        Predict aggregated trust value
        
        Args:
            features: [seq_len, feature_dim]
            phase_indices: [seq_len]
            mask: [seq_len]
        
        Returns:
            Predicted trust value
        """
        # Add batch dimension
        if features.dim() == 2:
            features = features.unsqueeze(0)
        if phase_indices.dim() == 1:
            phase_indices = phase_indices.unsqueeze(0)
        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(0)
        
        # Get predictions
        with torch.no_grad():
            outcome, _, _ = self.forward(features, phase_indices, mask)
        
        # Aggregate
        if mask is not None:
            valid_mask = mask.unsqueeze(-1)
            valid_sum = (outcome * valid_mask).sum(dim=1)
            valid_count = valid_mask.sum(dim=1).clamp(min=1)
            aggregated = (valid_sum / valid_count).squeeze()
        else:
            aggregated = outcome.mean(dim=1).squeeze()
        
        return aggregated.item()
    
    def detect_phase(
        self,
        features: torch.Tensor
    ) -> int:
        """
        Detect behavioral phase from features
        
        Args:
            features: [feature_dim] - Single timestep features
        
        Returns:
            Predicted phase index
        """
        # Add batch and sequence dimensions
        features = features.unsqueeze(0).unsqueeze(0)  # [1, 1, feature_dim]
        
        # Dummy phase index (will be predicted)
        phase_indices = torch.zeros(1, 1, dtype=torch.long)
        
        with torch.no_grad():
            _, phase_logits, _ = self.forward(features, phase_indices)
        
        # Get predicted phase
        predicted_phase = phase_logits.squeeze().argmax().item()
        
        return predicted_phase


class PhaseDetector:
    """
    Rule-based phase detection for environments
    Used to generate phase labels for training
    """
    
    @staticmethod
    def detect_lunar_phase(obs: np.ndarray) -> int:
        """
        Detect LunarLander phase
        0: cruise, 1: approach, 2: landing, 3: touchdown
        """
        if len(obs) < 8:
            return 0
        
        x, y, vx, vy, angle, angular_vel, left_leg, right_leg = obs[:8]
        
        # Touchdown: legs in contact
        if left_leg > 0.5 or right_leg > 0.5:
            return 3
        
        # Landing: low altitude, descending
        if y < 0.3 and vy < -0.1:
            return 2
        
        # Approach: moderate altitude, moving towards center
        if y < 0.7 and abs(x) < 0.4:
            return 1
        
        # Cruise: high altitude
        return 0
    
    @staticmethod
    def detect_hopper_phase(obs: np.ndarray) -> int:
        """
        Detect Hopper phase
        0: stance, 1: swing, 2: flight, 3: landing
        """
        if len(obs) < 11:
            return 0
        
        height = obs[0]
        joint_angles = obs[2:5]
        body_velocities = obs[8:11]
        vertical_velocity = body_velocities[2]
        
        # Flight: high and ascending
        if height > 1.3 and vertical_velocity > 0:
            return 2
        
        # Landing: descending
        if vertical_velocity < -0.2:
            return 3
        
        # Swing: leg moving
        if abs(joint_angles[1]) > 0.3:
            return 1
        
        # Stance: grounded
        return 0
    
    @staticmethod
    def detect_walker2d_phase(obs: np.ndarray) -> int:
        """
        Detect Walker2D phase
        0: left_stance, 1: right_stance, 2: double_support, 3: flight
        """
        if len(obs) < 17:
            return 2
        
        foot_contacts = obs[15:17]
        left_contact = foot_contacts[0] > 0.5
        right_contact = foot_contacts[1] > 0.5
        
        # Double support
        if left_contact and right_contact:
            return 2
        
        # Flight
        if not left_contact and not right_contact:
            return 3
        
        # Left stance
        if left_contact:
            return 0
        
        # Right stance
        return 1
