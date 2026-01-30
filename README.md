# Transformer-Based Trust Estimation for Peer Learning in Reinforcement Learning

Implementation of **Transformer-Based Trust Estimation for Peer Learning in Reinforcement Learning** research.

Multiple agents (*peers*) learning together from scratch with Transformer-based trust mechanisms for evaluating peer suggestions.

## Features

- **Attention Mechanisms**: Phase-aware transformers for context-dependent trust
- **Hybrid Trust**: Combines traditional trust with attention-based confidence
- **Environment Support**: LunarLander-v3 (DQN), Hopper-v4 (SAC), Walker2d-v4 (SAC), Ant-v4 (SAC), Room-v21(DQN)
- **Modern Stack**: Python 3.11+, PyTorch 2.0+, Gymnasium, Stable-Baselines3 2.2+
- **GPU Accelerated**: Full CUDA support for faster training

---

## Installation

- Python 3.11 or higher
- pip install -r requirements_attention.txt


## Architecture

### Project Structure
```
PeerLearning/
├── attention_peer_learning/       
│   ├── configs/                   # Environment configurations
│   │   └── env_config.py         # Single source of truth for env params
│   ├── features/                  # Feature extractors
│   │   ├── lunar_features.py     # 35-dim features for LunarLander
│   │   ├── hopper_features.py    # 38-dim features for Hopper
│   │   ├── walker2d_features.py  # 42-dim features for Walker2d
│   │   ├── ant_features.py       # 52-dim features for Ant
│   │   └── room_features.py      # 22-dim features for Room
│   ├── transformers/              # Attention mechanisms
│   │   ├── attention_transformer.py      # Base transformer
│   │   └── phase_aware_transformer.py    # Phase-aware transformer
│   ├── trust/                     # Trust mechanisms
│   │   └── hybrid_trust.py       # Baseline + Transformer-based trust
│   └── core/                      # Core peer classes
│       └── attention_peer.py     # AttentionSACPeer, AttentionDQNPeer
├── run_attention_peer.py          # Training script
├── peer.py                        # Base peer learning infrastructure
├── callbacks.py                   # Evaluation callbacks
└── utils.py                       # Utilities
```

### Hybrid Trust Formula
```
v_final = ω_t * norm(v_attn) + (1 - ω_t) * norm(v_trad)

where:
  ω_t = warmup_progress * max_weight * c_t
  c_t = 0.7*sigmoid(-ΔL̄) + 0.3/(1 + σ(L_history))
  max_weight = 0.5 (attention contribution cap)
```

---

## Technical Architecture Details

This section provides technical specifications for the transformer-based attention mechanism implemented in this work.

### Overview

The baseline peer learning framework uses scalar trust values updated via temporal difference learning:

```
v_trust(j) = v_trust(j) + α * (TD_target - v_trust(j))
```

This approach lacks temporal context and cannot model long-term behavioral patterns. Our Transformer-based framework addresses these limitations through a transformer architecture that processes sequential peer interaction histories.

### Transformer Architecture Pipeline

The attention mechanism follows a five-stage pipeline:

```
Input Features → Projection → Positional Encoding → 
Transformer Encoder → Multi-Scale Pooling → Outcome Predictor
```

Each component is detailed below with precise specifications.

### Layer 1: Input Projection

Environment-specific features are projected to a uniform hidden dimension:

```python
Input Projection:
  Linear(feature_dim → 64)
  LayerNorm(64)
  ReLU()
  Dropout(p=0.1)
```

**Specifications:**
- Input: `[batch_size, sequence_length, feature_dim]`
- Output: `[batch_size, sequence_length, 64]`
- Parameters: `feature_dim × 64 + 64` (e.g., 3,392 for Ant-v4)

### Layer 2: Positional Encoding

Sinusoidal encoding injects temporal information:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Specifications:**
- Maximum sequence length: 200
- Encoding dimension: 64
- Parameters: 0 (non-trainable, fixed)

### Layer 3: Transformer Encoder

Multi-head self-attention captures dependencies across sequence elements:

```python
TransformerEncoderLayer:
  d_model = 64
  nhead = 4 (each head processes 16 dimensions)
  dim_feedforward = 256 (d_model × 4)
  dropout = 0.1
  num_layers = 2
```

**Attention Mechanism:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) * V
where d_k = d_model / nhead = 16
```

**Feed-Forward Network (per layer):**
```
FFN(x) = ReLU(Linear(64 → 256)(x))
       → Dropout(0.1)
       → Linear(256 → 64)
```

**Parameters per layer:**
- Attention projections: 16,640
- Feed-forward: 33,216
- Layer normalization: 256
- Total per layer: ~50,112
- Total both layers: ~100,224

**Complexity:** O(n²d) for self-attention where n=sequence_length, d=d_model

### Layer 4: Multi-Scale Pooling

Dual pooling captures both general trends and critical events:

```python
avg_pool = mean(x, dim=1)  # General behavior
max_pool = max(x, dim=1)   # Critical moments
fused = Linear(128 → 64)(concat(avg_pool, max_pool))
```

**Specifications:**
- Input: `[batch, seq_len, 64]`
- Output: `[batch, seq_len, 64]` (broadcast after fusion)
- Parameters: 8,256

### Layer 5: Outcome Predictor

Multi-layer perceptron predicts peer contribution quality:

```python
Outcome Predictor:
  Linear(64 → 32)
  ReLU()
  Dropout(0.1)
  Linear(32 → 1)
```

**Specifications:**
- Output: Scalar trust score `v_attn` per timestep
- Parameters: 2,113
- Aggregation: Mean over valid sequence elements

### Phase-Aware Extension

The phase-aware transformer adds behavioral phase modeling:

**Phase Embeddings:**
```python
Embedding(n_phases, d_model)
Parameters: n_phases × 64 (e.g., 256 for 4 phases)
```

**Phase-Modulated Attention:**
- Learnable scaling parameters: `[n_phases, nhead]`
- Enables phase-specific attention patterns

**Phase Classifier:**
```python
Linear(64 → 32) → ReLU → Linear(32 → n_phases)
Combined Loss: L_total = L_outcome + 0.3 * L_phase
```

**Additional parameters:** ~2,500

### Total Parameter Counts

| Architecture | Parameters |
|--------------|-----------|
| Base Transformer | ~114,000 |
| Phase-Aware Transformer | ~116,500 |

### Environment-Specific Feature Dimensions

| Environment | obs_dim | action_dim | feature_dim | Behavioral Phases |
|-------------|---------|------------|-------------|-------------------|
| LunarLander-v3 | 8 | 4 | 35 | cruise, approach, landing, touchdown |
| Hopper-v4 | 11 | 3 | 38 | stance, swing, flight, landing |
| Walker2d-v4 | 17 | 6 | 42 | left_stance, right_stance, double_support, flight |
| Ant-v4 | 27 | 8 | 52 | trot, pace, gallop, stand |
| Room-v6/v21/v27/v33 | 4 | 4 | 22 | exploration, navigation, approaching, goal_reached |

**Ant-v4 Feature Breakdown (52 dimensions):**
1. Position Features (8): height, orientation, torso stability, balance
2. Joint Features (16): joint angles and velocities
3. Locomotion Features (10): gait phase, leg coordination, stride efficiency
4. Action Context (8): joint torques
5. Peer Context (4): peer one-hot encoding
6. Performance Features (6): reward, velocity efficiency, energy efficiency

### Hybrid Trust Integration

The final trust value combines traditional TD-learning with attention-based prediction:

```
v_final_j = ω_t * norm(v_attn_j) + (1 - ω_t) * norm(v_trad_j)
```

**Adaptive Weighting:**
```
c_t = 0.7 * sigmoid(-ΔL̄) + 0.3 / (1 + σ(L_history))
ω_t = min(step / warmup_steps, 1.0) * 0.5 * c_t
```

**Normalization (Critical Design):**
Each component is normalized independently before combination:
```
norm(x_j) = x_j / max(|x_0|, |x_1|, ..., |x_n|)
```

### Hyperparameter Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| d_model | 64 | Transformer hidden dimension |
| nhead | 4 | Number of attention heads |
| num_layers | 2 | Transformer encoder layers |
| dim_feedforward | 256 | FFN intermediate dimension |
| dropout | 0.1 | Dropout rate |
| max_seq_len | 200 | Maximum sequence for positional encoding |
| sequence_length | 50 | Actual training sequence length |
| batch_size | 32 | Transformer training batch |
| learning_rate | 1e-5 | Adam optimizer learning rate |
| update_freq | 20 | Transformer update interval (steps) |
| warmup_steps | 500,000 | Attention warmup period (full training horizon) |
| history_buffer | 10,000 | Transition buffer capacity |
| attention_weight | 0.5 | Maximum attention contribution (λ_max) |
| trust_scale | 300.0 | Environment-dependent normalization |

### Training Procedure

**Transformer Training Loop:**
1. Sample mini-batch from circular history buffer: `[batch=32, seq_len=50, feature_dim]`
2. Forward pass through transformer architecture
3. Compute loss:
   - Base: `MSE(predicted_outcome, actual_reward)`
   - Phase-aware: `MSE + 0.3 * CrossEntropy(predicted_phase, true_phase)`
4. Backward propagation and optimizer step
5. Update every 20 environment interactions

**Warmup Strategy:**
- Steps 0-500,000: Linear ramp-up of attention weight `ω_t` from 0 to maximum over full training horizon
- Purpose: Allow transformer to learn stable representations before deployment
- During warmup: Traditional trust dominates peer selection

### Computational Complexity

| Component | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Feature Extraction | O(d) | O(1) |
| Input Projection | O(n·d·m) | O(n·m) |
| Positional Encoding | O(1) | O(max_len·d) |
| Self-Attention | O(n²·d) | O(n²) |
| Feed-Forward | O(n·d²) | O(n·d) |
| Multi-Scale Pooling | O(n·d) | O(d) |
| Outcome Predictor | O(n·d) | O(n) |

where n=sequence_length, d=d_model, m=feature_dim

**Dominant term:** Self-attention O(n²·d) ≈ 160,000 operations for typical configuration (n=50, d=64)

### Comparison: Baseline vs Transformer-Based

| Aspect | Baseline | Transformer-Based |
|--------|----------|-------------------|
| Trust Mechanism | Scalar TD-learning | Transformer + TD hybrid |
| Trainable Parameters | 0 | ~114,000 |
| Temporal Context | 1 step (instant) | 50 steps (sequence) |
| Complexity | O(1) | O(n²·d) |
| Feature Processing | Raw observations | Structured extraction |
| Phase Awareness | None | Learnable phase embeddings |
| Long-term Dependencies | Not modeled | Multi-head self-attention |
| Adaptive Integration | Fixed weights | Confidence-based weighting |
| Memory Requirements | Minimal | 10,000 transition buffer |

---

## Command-Line Arguments

### Environment
- `--env`: Environment name (`LunarLander-v3`, `Hopper-v4`, `Walker2d-v4`, `Ant-v4`, `Room-v21`)
- `--n-peers`: Number of peer agents (default: 4)
- `--seed`: Random seed (optional)

### Training
- `--n-epochs`: Number of training epochs (default: 100)
- `--epoch-length`: Steps per epoch (default: 1000)
- `--total-timesteps`: Override total timesteps (optional)

### Attention
- `--use-phase-aware`: Enable phase-aware transformer (recommended)
- `--attention-lr`: Attention learning rate (default: 5e-5)
- `--attention-update-freq`: Update frequency in steps (default: 20)
- `--sequence-length`: Sequence length for transformer (default: 50)

### Peer Learning
- `--use-trust`: Enable trust mechanisms (default: True)
- `--use-agent-values`: Use agent value functions (default: True)
- `--temperature`: Sampling temperature (default: 1.0)
- `--lr`: Trust update learning rate (default: 0.95)

### Logging
- `--exp-name`: Experiment name (default: auto-generated)
- `--log-dir`: Log directory (default: `experiments`)
- `--use-wandb`: Enable W&B logging
- `--wandb-project`: W&B project name (default: `attention-peer-learning`)
- `--verbose`: Verbosity level (0, 1, 2)

### Implementation Notes

**Sequence Buffer Management:**
- Circular buffer stores last 10,000 transitions
- Each transition: `(feature, phase, reward, peer_id)`
- Sampling yields batches of 50-step sequences with padding masks

**Gradient Flow:**
- Transformer trained end-to-end with backpropagation through time
- Residual connections in encoder prevent gradient vanishing
- Layer normalization stabilizes training

---

#
## Experiment Commands

### LunarLander-v3 (500k Steps)

**Attention-Enhanced:**
```bash
python run_attention_peer.py \
  --env LunarLander-v3 \
  --n-peers 4 \
  --total-timesteps 500000 \
  --epoch-length 1000 \
  --use-trust \
  --use-agent-values \
  --temperature 1.0 \
  --lr 0.95 \
  --use-phase-aware \
  --attention-lr 1e-5 \
  --attention-update-freq 20 \
  --sequence-length 50 \
  --warmup-steps 500000 \
  --exp-name <project_name> \
  --log-dir <log_dir> \
  --use-wandb \
  --wandb-project <wandb_project_name> \
  --seed <seed> \
  --verbose 1
```

---

### Hopper-v4 (500k Steps)

**Attention-Enhanced:**
```bash
python run_attention_peer.py \
  --env Hopper-v4 \
  --n-peers 4 \
  --total-timesteps 500000 \
  --epoch-length 1000 \
  --use-trust \
  --use-agent-values \
  --temperature 1.0 \
  --lr 0.95 \
  --use-phase-aware \
  --attention-lr 1e-5 \
  --attention-update-freq 20 \
  --sequence-length 50 \
  --warmup-steps 500000 \
  --exp-name <project_name> \
  --log-dir <log_dir> \
  --use-wandb \
  --wandb-project <wandb_project_name> \
  --seed <seed> \
  --verbose 1
```

---

### Walker2d-v4 (500k Steps)

**Attention-Enhanced:**
```bash
python run_attention_peer.py \
  --env Walker2d-v4 \
  --n-peers 4 \
  --total-timesteps 500000 \
  --epoch-length 1000 \
  --use-trust \
  --use-agent-values \
  --temperature 1.0 \
  --lr 0.95 \
  --use-phase-aware \
  --attention-lr 1e-5 \
  --attention-update-freq 20 \
  --sequence-length 50 \
  --warmup-steps 500000 \
  --exp-name <project_name> \
  --log-dir <log_dir> \
  --use-wandb \
  --wandb-project <wandb_project_name> \
  --seed <seed> \
  --verbose 1
```

---

### Ant-v4 (500k Steps)

**Attention-Enhanced:**
```bash
python run_attention_peer.py \
  --env Ant-v4 \
  --n-peers 4 \
  --total-timesteps 500000 \
  --seed <seed> \
  --use-phase-aware \
  --attention-lr 1e-5 \
  --warmup-steps 500000 \
  --use-trust \
  --use-agent-values \
  --verbose 1 \
  --use-wandb \
  --wandb-project <wandb_project_name> \
  --exp-name <project_name> \
  --epoch-length 10000
```

---

### Room-v21 (500k Steps)

**Attention-Enhanced:**
```bash
python run_attention_peer.py \
  --env Room-v21 \
  --n-peers 4 \
  --total-timesteps 500000 \
  --epoch-length 5000 \
  --use-trust \
  --use-agent-values \
  --temperature 1.0 \
  --lr 0.95 \
  --use-phase-aware \
  --attention-lr 1e-5 \
  --attention-update-freq 20 \
  --sequence-length 50 \
  --warmup-steps 500000 \
  --exp-name <project_name> \
  --log-dir <log_dir> \
  --use-wandb \
  --wandb-project <wandb_project_name> \
  --seed <seed> \
  --verbose 1 \
  --device cuda
```

---

## Citation

If you use this code or refer to the technical architecture in your research, please cite:

```bibtex
@software{transformer_trust_peerlearning,
  title        = {Transformer-Based Trust Estimation for Peer Learning in Reinforcement Learning},
  author       = {},
  year         = {2026},
  url          = {},
  institution = {},
  note         = {Software repository (preprint under submission to IJCNN 2026).}
}
```

For technical architecture details, refer to the Technical Architecture Details section in this README or the implementation in `transformer_trust_peerlearning/`.

---


## License
See [LICENSE] file for details.

Baseline PeerLearning: https://github.com/kramerlab/PeerLearning



