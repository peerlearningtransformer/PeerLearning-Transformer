"""
Base Attention Transformer for Peer Trust Evaluation
Implements Section 3.3 of the paper - Transformer-based Attention Mechanism

Architecture:
1. Input Projection: Projects feature vectors to d_model dimension (Eq. 2)
2. Positional Encoding: Adds temporal position information
3. Multi-Head Attention: Captures relationships between sequence elements
4. Multi-Scale Pooling: Aggregates information at different temporal scales
5. Outcome Predictor: Predicts peer contribution quality

Key Equations:
- Input Projection (Eq. 2): X_proj = W_proj * X + b_proj
- Attention (Eq. 1): Attention(Q,K,V) = softmax(QK^T/√d_k)V
- Trust Value: v_attn computed from outcome prediction
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Tuple, Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal context"""
    
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            x with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class MultiScalePooling(nn.Module):
    """
    Global Average + Max Pooling as per Section 3.3.5
    
    "The architecture concatenates global average and max pooling outputs 
    to synthesize the sequence" - captures both general trends (avg) and 
    critical failure moments (max) for trust evaluation.
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Projection layer to combine concatenated pooling outputs (2 * d_model -> d_model)
        self.fusion = nn.Linear(d_model * 2, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model] - Encoded sequence
        Returns:
            Pooled features [batch, seq_len, d_model]
        """
        # Global Average Pooling: captures overall sequence behavior
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # [batch, 1, d_model]
        
        # Global Max Pooling: captures critical/extreme moments (failures, successes)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # [batch, 1, d_model]
        
        # Concatenate global statistics
        global_features = torch.cat([avg_pool, max_pool], dim=-1)  # [batch, 1, d_model * 2]
        
        # Broadcast to sequence length
        global_features = global_features.expand(-1, x.size(1), -1)  # [batch, seq_len, d_model * 2]
        
        # Fuse to original dimension
        fused = self.fusion(global_features)  # [batch, seq_len, d_model]
        
        return fused


class AttentionTransformer(nn.Module):
    """
    Base transformer for peer behavior analysis and trust evaluation
    
    Args:
        feature_dim: Dimension of input features (environment-specific)
        d_model: Transformer hidden dimension (default: 64)
        nhead: Number of attention heads (default: 4)
        num_layers: Number of transformer layers (default: 2)
        max_seq_len: Maximum sequence length (default: 200)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        feature_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 200,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        # 1. Input projection (Eq. 2)
        self.input_projection = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # 3. Transformer encoder layers
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
        
        # 4. Multi-scale pooling
        self.multi_scale_pooling = MultiScalePooling(d_model)
        
        # 5. Outcome predictor
        self.outcome_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)  # Predict single outcome value
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(
        self, 
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through transformer
        
        Args:
            features: [batch, seq_len, feature_dim] - Sequence of feature vectors
            mask: [batch, seq_len] - Optional attention mask (1 = valid, 0 = masked)
        
        Returns:
            outcome: [batch, seq_len, 1] - Predicted outcome for each timestep
            attention_weights: [batch, seq_len, d_model] - Encoded representations
        """
        batch_size, seq_len, _ = features.shape
        
        # 1. Project input features to d_model dimension
        x = self.input_projection(features)  # [batch, seq_len, d_model]
        
        # 2. Add positional encoding
        x = self.positional_encoding(x)
        
        # 3. Create attention mask if needed
        src_key_padding_mask = None
        if mask is not None:
            # Convert mask to padding mask (True = masked, False = valid)
            src_key_padding_mask = (mask == 0)
        
        # 4. Apply transformer encoder
        encoded = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask
        )  # [batch, seq_len, d_model]
        
        # 5. Apply multi-scale pooling
        pooled = self.multi_scale_pooling(encoded)  # [batch, seq_len, d_model]
        
        # 6. Predict outcome for each timestep
        outcome = self.outcome_predictor(pooled)  # [batch, seq_len, 1]
        
        return outcome, pooled
    
    def predict_peer_value(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> float:
        """
        Predict aggregated trust value for a peer based on sequence
        
        Args:
            features: [seq_len, feature_dim] - Single sequence
            mask: [seq_len] - Optional mask
        
        Returns:
            Predicted trust value (scalar)
        """
        # Add batch dimension
        if features.dim() == 2:
            features = features.unsqueeze(0)
        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(0)
        
        # Get predictions
        with torch.no_grad():
            outcome, _ = self.forward(features, mask)
        
        # Aggregate over sequence (mean of valid predictions)
        if mask is not None:
            valid_mask = mask.unsqueeze(-1)  # [batch, seq_len, 1]
            valid_sum = (outcome * valid_mask).sum(dim=1)
            valid_count = valid_mask.sum(dim=1).clamp(min=1)
            aggregated = (valid_sum / valid_count).squeeze()
        else:
            aggregated = outcome.mean(dim=1).squeeze()
        
        # Ensure scalar conversion
        if aggregated.dim() > 0:
            aggregated = aggregated[0] if aggregated.numel() > 0 else aggregated
        return float(aggregated.item())
    
    def compute_attention_weights(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract attention weights from transformer
        
        Args:
            features: [seq_len, feature_dim]
        
        Returns:
            Attention weights [seq_len, seq_len]
        """
        # Add batch dimension
        if features.dim() == 2:
            features = features.unsqueeze(0)
        
        # Project and encode
        x = self.input_projection(features)
        x = self.positional_encoding(x)
        
        # Get attention from first layer
        # Note: This is a simplified version - full implementation would extract
        # attention weights from transformer layers
        encoded = self.transformer_encoder(x)
        
        # Compute pairwise similarity as proxy for attention
        attention = torch.matmul(encoded, encoded.transpose(1, 2))
        attention = torch.softmax(attention / math.sqrt(self.d_model), dim=-1)
        
        return attention.squeeze(0)


class CounterfactualEvaluator:
    """
    Counterfactual evaluation to isolate peer behavioral signature
    As per paper Section 3.3.3
    
    Creates counterfactual sequences by replacing peer_id embeddings
    to evaluate what would have happened with different peer suggestions
    """
    
    def __init__(self, n_peers: int = 4):
        self.n_peers = n_peers
    
    def create_counterfactual_sequence(
        self,
        features: np.ndarray,
        original_peer_id: int,
        counterfactual_peer_id: int,
        peer_feature_indices: Tuple[int, int]
    ) -> np.ndarray:
        """
        Create counterfactual sequence by replacing peer ID
        
        Args:
            features: [seq_len, feature_dim] - Original feature sequence
            original_peer_id: Original peer who suggested actions
            counterfactual_peer_id: Hypothetical peer to evaluate
            peer_feature_indices: (start_idx, end_idx) of peer one-hot in features
        
        Returns:
            Counterfactual features with replaced peer ID
        """
        counterfactual = features.copy()
        
        start_idx, end_idx = peer_feature_indices
        
        # Replace peer one-hot encoding
        counterfactual[:, start_idx:end_idx] = 0
        if counterfactual_peer_id < (end_idx - start_idx):
            counterfactual[:, start_idx + counterfactual_peer_id] = 1.0
        
        return counterfactual
    
    def evaluate_counterfactual_trust(
        self,
        transformer: AttentionTransformer,
        features: np.ndarray,
        peer_id: int,
        peer_feature_indices: Tuple[int, int],
        phase_indices: Optional[np.ndarray] = None
    ) -> float:
        """
        Evaluate peer using counterfactual analysis
        
        Args:
            transformer: Trained attention transformer
            features: [seq_len, feature_dim] - Feature sequence
            peer_id: Peer to evaluate
            peer_feature_indices: Location of peer features
            phase_indices: [seq_len] - Phase indices (for phase-aware transformers)
        
        Returns:
            Counterfactual trust value
        """
        # Create counterfactual for this peer
        counterfactual = self.create_counterfactual_sequence(
            features, 
            original_peer_id=0,  # Doesn't matter for evaluation
            counterfactual_peer_id=peer_id,
            peer_feature_indices=peer_feature_indices
        )
        
        # Predict with transformer
        # Get device from transformer
        device = next(transformer.parameters()).device 
        features_tensor = torch.FloatTensor(counterfactual).to(device)
        
        # Check if transformer needs phase indices (PhaseAwareTransformer)
        if phase_indices is not None and hasattr(transformer, 'phase_embedding'):
            phase_tensor = torch.LongTensor(phase_indices).to(device)
            trust_value = transformer.predict_peer_value(features_tensor, phase_tensor)
        else:
            trust_value = transformer.predict_peer_value(features_tensor)
        
        return trust_value
