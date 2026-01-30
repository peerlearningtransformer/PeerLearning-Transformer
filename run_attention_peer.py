"""
Enhanced Training Script for Attention-Enhanced Peer Learning
Supports LunarLander-v3, Hopper-v4, Walker2d-v4, Ant-v4, Room-v3/v6/v21/v27/v33

Features:
- Automatic environment detection
- Attention-specific hyperparameters
- Clean logging and callbacks
- W&B integration (optional)
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Dict, Any

import numpy as np
import gymnasium as gym
import torch

# Import attention peer classes
from attention_peer_learning.core.attention_peer import AttentionSACPeer, AttentionDQNPeer
from attention_peer_learning.configs.env_config import get_env_config, print_env_config

# Import base peer learning infrastructure
from peer import PeerGroup
from callbacks import PeerEvalCallback

# Register custom environments
import env as local_envs  # noqa: F401


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Attention-Enhanced Peer Learning')
    
    # Environment
    parser.add_argument('--env', type=str, required=True,
                       choices=['LunarLander-v3', 'Hopper-v4', 'Walker2d-v4', 'Ant-v4', 'Room-v3', 'Room-v6', 'Room-v21', 'Room-v27', 'Room-v33'],
                       help='Environment name')
    
    # Training
    parser.add_argument('--n-peers', type=int, default=4,
                       help='Number of peers (default: 4)')
    parser.add_argument('--n-epochs', type=int, default=500,
                       help='Number of training epochs (default: 500)')
    parser.add_argument('--epoch-length', type=int, default=1000,
                       help='Steps per epoch (default: 1000)')
    parser.add_argument('--total-timesteps', type=int, default=None,
                       help='Total timesteps (overrides n-epochs if provided)')
    
    # Peer learning
    parser.add_argument('--use-trust', action='store_true', default=True,
                       help='Use trust values (default: True)')
    parser.add_argument('--use-agent-values', action='store_true', default=True,
                       help='Use agent values (default: True)')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature (default: 1.0)')
    parser.add_argument('--lr', type=float, default=0.95,
                       help='Learning rate for trust updates (default: 0.95)')
    
    # Attention parameters
    parser.add_argument('--use-phase-aware', action='store_true', default=True,
                       help='Use phase-aware transformer (default: True)')
    parser.add_argument('--attention-lr', type=float, default=1e-5,# can be modified
                       help='Attention learning rate (default: 1e-5)')
    parser.add_argument('--attention-update-freq', type=int, default=20,
                       help='Attention update frequency (default: 20)')
    parser.add_argument('--sequence-length', type=int, default=50,
                       help='Sequence length for transformer (default: 50)')
    parser.add_argument('--warmup-steps', type=int, default=500000,
                       help='Warmup steps for attention mechanism (default: 500000)')
    
    # Logging
    parser.add_argument('--log-dir', type=str, default='experiments',
                       help='Log directory (default: experiments)')
    parser.add_argument('--exp-name', type=str, default=None,
                       help='Experiment name (default: auto-generated)')
    parser.add_argument('--use-wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='attention-peer-learning',
                       help='W&B project name')
    
    # Misc
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'cuda', 'auto'],
                       help='Device to use: cpu, cuda, or auto (default: auto)')
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level (default: 1)')
    parser.add_argument('--print-trust-freq', type=int, default=10,
                       help='Print trust values every N evaluations (default: 10, 0=never)')
    
    return parser.parse_args()


def create_experiment_name(args) -> str:
    """Create experiment name from arguments"""
    if args.exp_name:
        return args.exp_name
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
    env_short = args.env.split('-')[0].lower()
    phase = 'phase' if args.use_phase_aware else 'basic'
    
    return f"{timestamp}_{env_short}_attn_{phase}_n{args.n_peers}"


def get_algorithm_args(env_name: str, args) -> Dict[str, Any]:
    """Get algorithm-specific arguments"""
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Common args
    common_args = {
        'policy': 'MlpPolicy',  # Required by stable-baselines3
        'verbose': args.verbose,
        'device': device,
    }
    
    # Environment-specific
    if 'Lunar' in env_name or 'Room' in env_name:
        # DQN args (Discrete action space)
        algo_args = {
            'learning_rate': 5e-4,
            'buffer_size': 100000,
            'learning_starts': 1000,
            'batch_size': 128,
            'tau': 1.0,
            'gamma': 0.99,
            'train_freq': 4,
            'gradient_steps': 1,
            'target_update_interval': 1000,
            'exploration_fraction': 0.12,
            'exploration_final_eps': 0.1,
        }
    else:
        # SAC args (Hopper, Walker2D, Ant - Continuous action space)
        # Updated for baseline compatibility (tau=0.02, train_freq=8, gradient_steps=8)
        algo_args = {
            'learning_rate': 3e-4,
            'buffer_size': 1000000,
            'learning_starts': 10000,
            'batch_size': 256,
            'tau': 0.02,  # Baseline compatible (was 0.005)
            'gamma': 0.99,
            'train_freq': 8,  # Baseline compatible (was 1)
            'gradient_steps': 8,  # Baseline compatible (was 1)
        }
    
    algo_args.update(common_args)
    return algo_args


def create_peers(args, env_name: str, exp_dir: str):
    """Create attention peer agents"""
    
    # Get algorithm args
    algo_args = get_algorithm_args(env_name, args)
    
    # Determine peer class
    if 'Lunar' in env_name or 'Room' in env_name:
        PeerClass = AttentionDQNPeer
    else:
        PeerClass = AttentionSACPeer
    
    # Create peers
    peers = []
    for i in range(args.n_peers):
        peer = PeerClass(
            env_name=env_name,
            temperature=args.temperature,
            temp_decay=0.0,
            algo_args=algo_args,
            env=env_name,
            use_trust=args.use_trust,
            use_critic=True,
            init_trust_values=200.0,
            buffer_size=100,  # Reduced from 1000 to start trust updates earlier
            follow_steps=10,
            seed=args.seed + i if args.seed else None,
            use_trust_buffer=True,
            solo_training=False,
            # Attention-specific
            use_phase_aware=args.use_phase_aware,
            attention_lr=args.attention_lr,
            attention_update_freq=args.attention_update_freq,
            sequence_length=args.sequence_length,
            warmup_steps=args.warmup_steps,
        )
        peers.append(peer)
        
        if args.verbose > 0:
            print(f"✅ Created Peer {i}")
    
    return peers


def create_callbacks(args, peers, peer_group, env_name: str, exp_dir: str):
    """Create evaluation callbacks for each peer"""
    from stable_baselines3.common.monitor import Monitor
    
    # Create eval envs for all peers (wrapped with Monitor to avoid warnings)
    eval_envs = [Monitor(gym.make(env_name)) for _ in peers]
    
    callbacks = []
    for i, peer in enumerate(peers):
        # Create callback
        callback = PeerEvalCallback(
            peer_group=peer_group,
            eval_envs=eval_envs,
            eval_env=eval_envs[i],
            best_model_save_path=os.path.join(exp_dir, f'peer_{i}'),
            log_path=os.path.join(exp_dir, f'peer_{i}'),
            eval_freq=5000,  # Fixed interval as specified in paper
            n_eval_episodes=10,
            deterministic=True,
            render=False,
            verbose=args.verbose,
            print_trust_freq=args.print_trust_freq
        )
        callbacks.append(callback)
    
    return callbacks


def setup_wandb(args, exp_name: str):
    """Setup Weights & Biases logging"""
    if not args.use_wandb:
        return None
    
    try:
        import wandb
        
        config = {
            'env': args.env,
            'n_peers': args.n_peers,
            'n_epochs': args.n_epochs,
            'epoch_length': args.epoch_length,
            'use_trust': args.use_trust,
            'use_agent_values': args.use_agent_values,
            'temperature': args.temperature,
            'lr': args.lr,
            'use_phase_aware': args.use_phase_aware,
            'attention_lr': args.attention_lr,
            'attention_update_freq': args.attention_update_freq,
            'sequence_length': args.sequence_length,
        }
        
        wandb.init(
            project=args.wandb_project,
            name=exp_name,
            config=config,
            mode='online'
        )
        
        print("✅ W&B logging initialized")
        return wandb
    
    except ImportError:
        print("⚠️ wandb not installed, skipping W&B logging")
        return None


def main():
    """Main training loop"""
    args = parse_args()
    
    # Create experiment directory
    exp_name = create_experiment_name(args)
    exp_dir = os.path.join(args.log_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("ATTENTION-ENHANCED PEER LEARNING")
    print("="*80)
    print(f"Experiment: {exp_name}")
    print(f"Environment: {args.env}")
    print(f"Log dir: {exp_dir}")
    print("="*80 + "\n")
    
    # Print environment configuration
    env_config = get_env_config(args.env)
    print_env_config(env_config)
    print()
    
    # Setup W&B
    wandb_run = setup_wandb(args, exp_name)
    
    # Print device info
    if torch.cuda.is_available():
        print(f"\n🚀 Using CUDA: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}\n")
    else:
        print("\n⚠️  CUDA not available, using CPU\n")
    
    # Create peers
    print("Creating peers...")
    peers = create_peers(args, args.env, exp_dir)
    print(f"✅ Created {len(peers)} peers\n")
    
    # Create peer group
    print("Creating peer group...")
    peer_group = PeerGroup(
        peers=peers,
        use_agent_values=args.use_agent_values,
        init_agent_values=200.0,
        lr=args.lr,
        switch_ratio=0,
        use_advantage=False,
        max_peer_epochs=args.n_epochs
    )
    print(f"✅ Peer group created\n")
    
    # Create callbacks
    print("Creating callbacks...")
    callbacks = create_callbacks(args, peers, peer_group, args.env, exp_dir)
    print(f"✅ Created {len(callbacks)} callbacks\n")
    
    # Calculate total timesteps
    if args.total_timesteps:
        total_timesteps = args.total_timesteps
        n_epochs = total_timesteps // args.epoch_length
    else:
        n_epochs = args.n_epochs
        total_timesteps = n_epochs * args.epoch_length
    
    print(f"Training configuration:")
    print(f"  Epochs: {n_epochs}")
    print(f"  Steps per epoch: {args.epoch_length}")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Attention updates every: {args.attention_update_freq} steps")
    print()
    
    # Train
    print("="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")
    
    try:
        peer_group.learn(
            n_epochs=n_epochs,
            max_epoch_len=args.epoch_length,
            callbacks=callbacks
        )
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Results saved to: {exp_dir}")
        
        # Save final models
        for i, peer in enumerate(peers):
            save_path = os.path.join(exp_dir, f'peer_{i}_final')
            peer.save(save_path)
            print(f"✅ Saved Peer {i} to {save_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        print(f"Partial results saved to: {exp_dir}")
    
    finally:
        if wandb_run:
            wandb_run.finish()


if __name__ == '__main__':
    main()
