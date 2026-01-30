from typing import List, Union

import os
import time
import gymnasium as gym
import numpy as np
import wandb
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecEnv

from peer import PeerGroup
from attention_peer_learning.analysis.trust_logger import TrustAnalyzer


class PeerEvalCallback(EvalCallback):
    """
    Callback to track collective measurements about peers.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use
      ``eval_freq = max(eval_freq // n_envs, 1)``

    :param peer_group: The group of peers
    :param eval_env: The environment used for initialization
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the
        callback.
    :param log_path: Path to a folder where the evaluations
        (``evaluations.npz``) will be saved. It will be updated at each
        evaluation.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has
        not been wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        peer_group: PeerGroup,
        eval_envs: List[Union[gym.Env, VecEnv]],
        n_samples=100,
        print_trust_freq=10,
        **kwargs
    ):
        self.peer_group = peer_group
        self.eval_envs = eval_envs
        self.n_samples = n_samples
        self.print_trust_freq = print_trust_freq  # Print trust every N evaluations

        self.last_logged_matrix = None
        self.follow_matrix = np.zeros((len(peer_group), len(peer_group)))

        self.start_time = time.time()
        
        # Initialize TrustAnalyzer
        log_path = kwargs.get('log_path', '.')
        self.trust_analyzer = TrustAnalyzer(len(peer_group), log_path)
        
        # Initialize reward tracking for each peer
        n_peers = len(peer_group)
        self.reward_history = {i: [] for i in range(n_peers)}  # All rewards
        self.max_reward = {i: -np.inf for i in range(n_peers)}  # Best reward so far
        self.step_history = {i: [] for i in range(n_peers)}  # Step numbers for each eval

        super().__init__(**kwargs)

    def _on_step(self) -> bool:
        self.accumulate_followed_peers()  # needs to be done at every step

        # log time for debugging etc.
        self.logger.record("time/time_elapsed",
                           time.time() - self.start_time,
                           exclude="tensorboard")

        super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if 'agent_values' in self.peer_group.__dict__:
                self.track_agent_values()
            # Note: trust_values are now logged via _log_trust_to_wandb() for all peers
            
            active_peer = self.peer_group.active_peer
            self.track_followed_agent(active_peer)
            
            # Log trust dynamics to TrustAnalyzer for the ACTIVE peer only
            # (Report generation happens after ALL peers complete - see below)
            peer_obj = self.peer_group.peers[active_peer]
            
            # Unified trust logging for both baseline and attention peers
            if hasattr(peer_obj, 'get_trust_data'):
                try:
                    # Get trust data (works for both baseline and attention)
                    trust_data = peer_obj.get_trust_data()
                    
                    # Extract values
                    final_trust = trust_data.get('final_trust', {})
                    attention_trust = trust_data.get('attention_trust')
                    traditional_trust = trust_data.get('traditional_trust')
                    confidence = trust_data.get('confidence', 0.0)
                    omega_t = trust_data.get('omega_t', 0.0)
                    
                    # Log to trust analyzer
                    self.trust_analyzer.log_trust_step(
                        step=self.n_calls,
                        peer_id=active_peer,
                        trust_dict=final_trust,
                        attention_trust=attention_trust,
                        traditional_trust=traditional_trust,
                        confidence=confidence,
                        omega_t=omega_t
                    )
                    
                    # Log performance for the active peer
                    self.trust_analyzer.log_performance(
                        step=self.n_calls,
                        peer_id=active_peer,
                        reward=self.last_mean_reward
                    )
                    
                    # Print trust values to console (controlled by print_trust_freq)
                    if self.print_trust_freq > 0 and (self.n_calls // self.eval_freq) % self.print_trust_freq == 0:
                        self._print_trust_to_console(active_peer, trust_data)
                    
                    # Log individual trust values to WandB for ALL peers
                    for peer_id in range(len(self.peer_group.peers)):
                        peer_data = self.peer_group.peers[peer_id].get_trust_data()
                        self._log_trust_to_wandb(peer_id, peer_data)
                
                except Exception as e:
                    # Log warning but continue - don't crash entire training
                    print(f"Warning: Trust logging failed for peer {active_peer} at step {self.n_calls}: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            # Update metrics after logging
            self.trust_analyzer.update_metrics(self.n_calls)
            
            # Log trust metrics and visualizations to WandB
            self._log_metrics_to_wandb()
            self._log_trust_matrix_to_wandb()
            
            # Generate full trust report ONLY after ALL peers have completed this epoch
            # (active_peer == last peer means all peers finished this round)
            is_last_peer = active_peer == len(self.peer_group) - 1
            if is_last_peer and self.n_calls % 1000 == 0:
                try:
                    log_path = self.log_path if hasattr(self, 'log_path') else '.'
                    report_dir = os.path.join(log_path, 'trust_analysis')
                    self.trust_analyzer.generate_full_report(report_dir)
                except Exception as e:
                    print(f"Warning: Trust report generation failed at step {self.n_calls}: {str(e)}")
            
            # Track reward statistics for active peer
            self._track_reward_statistics(active_peer)
            
            # Log eval values (for active peer) and commit
            eval_values = {
                f"Peer{active_peer}_0/eval/mean_reward": self.last_mean_reward,
            }
            if wandb.run is not None:
                if is_last_peer:
                    eval_values["global_step"] = self.n_calls
                    wandb.log(eval_values, commit=True, step=self.num_timesteps)
                else:
                    wandb.log(eval_values, commit=False, step=self.num_timesteps)
        return True

    def track_agent_values(self):
        if wandb.run is None:
            return True
        n_agents = len(self.peer_group.peers)
        for i in range(n_agents):
            agent_value = self.peer_group.agent_values[i]
            wandb.log({'Peer{}_0/eval/agent_value'.format(i): agent_value},
                      commit=False, step=self.num_timesteps)
        return True

    def track_trust_values(self):
        if wandb.run is None:
            return True
        peer = self.peer_group.active_peer
        trust_i = self.peer_group.peers[peer].trust_values
        for j, el in np.ndenumerate(trust_i):
            wandb.log({'Peer{}_0/eval/trust_{}'.format(peer, j[0]): el},
                      commit=False, step=self.num_timesteps)
        return True

    def _track_reward_statistics(self, peer_id: int):
        """
        Track comprehensive reward statistics for a peer.
        Logs: max reward, mean over time, moving averages, first/last 100k comparison.
        """
        if wandb.run is None:
            return
        
        current_reward = self.last_mean_reward
        current_step = self.num_timesteps
        
        # Update tracking
        self.reward_history[peer_id].append(current_reward)
        self.step_history[peer_id].append(current_step)
        self.max_reward[peer_id] = max(self.max_reward[peer_id], current_reward)
        
        # Calculate statistics
        rewards = np.array(self.reward_history[peer_id])
        steps = np.array(self.step_history[peer_id])
        
        # Basic statistics
        mean_reward_overall = np.mean(rewards)
        
        # Moving average (last 10 evaluations)
        window_size = min(10, len(rewards))
        mean_reward_ma10 = np.mean(rewards[-window_size:])
        
        # Moving average (last 50 evaluations)
        window_size_50 = min(50, len(rewards))
        mean_reward_ma50 = np.mean(rewards[-window_size_50:])
        
        # First 100k vs Last 100k steps comparison
        first_100k_mask = steps <= 100000
        last_100k_mask = steps >= (current_step - 100000)
        
        mean_first_100k = np.mean(rewards[first_100k_mask]) if np.any(first_100k_mask) else 0.0
        mean_last_100k = np.mean(rewards[last_100k_mask]) if np.any(last_100k_mask) else current_reward
        
        # Log all statistics to W&B
        wandb.log({
            f'Peer{peer_id}_0/rewards/max_reward': self.max_reward[peer_id],
            f'Peer{peer_id}_0/rewards/mean_overall': mean_reward_overall,
            f'Peer{peer_id}_0/rewards/mean_ma10': mean_reward_ma10,
            f'Peer{peer_id}_0/rewards/mean_ma50': mean_reward_ma50,
            f'Peer{peer_id}_0/rewards/mean_first_100k': mean_first_100k,
            f'Peer{peer_id}_0/rewards/mean_last_100k': mean_last_100k,
            f'Peer{peer_id}_0/rewards/improvement': mean_last_100k - mean_first_100k,
        }, commit=False, step=self.num_timesteps)

    def accumulate_followed_peers(self):
        peer = self.peer_group.active_peer
        followed_peer = self.peer_group.peers[peer].followed_peer
        if followed_peer is not None:
            self.follow_matrix[peer, followed_peer] += 1
            
            # Log to TrustAnalyzer immediately at every step
            self.trust_analyzer.log_peer_selection(
                step=self.num_timesteps,
                follower=peer,
                followed=followed_peer
            )

    def track_followed_agent(self, active_peer):
        if self.last_logged_matrix is None:
            diff = self.follow_matrix
        else:
            diff = self.follow_matrix - self.last_logged_matrix

        if wandb.run is not None:
            for (followed_peer,), count in np.ndenumerate(
                    self.follow_matrix[active_peer]):
                wandb.log({'Peer{}_0/eval/follow_count{}'.format(
                    active_peer, followed_peer): count},  commit=False, step=self.num_timesteps)
                # also log difference
                wandb.log({'Peer{}_0/eval/follow_count_{}diff'.format(
                    active_peer, followed_peer): diff[active_peer, followed_peer]},
                          commit=False, step=self.num_timesteps)
        self.last_logged_matrix = np.copy(self.follow_matrix)

    def commit_global_step(self, timesteps):
        if wandb.run is not None and self.peer_group.active_peer == len(self.peer_group) - 1:
            eval_values = {"global_step": self.n_calls + self.eval_freq}
            wandb.log(eval_values, commit=True, step=self.num_timesteps)

        self.n_calls += timesteps
    
    def _log_trust_to_wandb(self, peer_id: int, trust_data: dict):
        """Log individual trust values to WandB"""
        if wandb.run is None:
            return
        
        # Batch all logs into a single wandb.log() call
        log_dict = {}
        
        # Check if this is attention-enhanced or baseline peer
        is_attention_peer = trust_data.get('attention_trust') is not None
        
        if is_attention_peer:
            # Attention peer: Log all three trust types
            final_trust = trust_data.get('final_trust', {})
            attention_trust = trust_data.get('attention_trust', {})
            traditional_trust = trust_data.get('traditional_trust', {})
            
            for target_peer_id in final_trust.keys():
                # Safely access with .get() to avoid KeyError
                if target_peer_id in attention_trust and target_peer_id in traditional_trust:
                    log_dict[f'Peer{peer_id}/trust/final_to_peer_{target_peer_id}'] = final_trust[target_peer_id]
                    log_dict[f'Peer{peer_id}/trust/attention_to_peer_{target_peer_id}'] = attention_trust[target_peer_id]
                    log_dict[f'Peer{peer_id}/trust/traditional_to_peer_{target_peer_id}'] = traditional_trust[target_peer_id]
            
            # Log confidence and omega for attention peers
            log_dict[f'Peer{peer_id}/trust/confidence'] = trust_data.get('confidence', 0.0)
            log_dict[f'Peer{peer_id}/trust/omega_t'] = trust_data.get('omega_t', 0.0)
        else:
            # Baseline peer: Only log final trust (traditional_trust is same as final_trust)
            final_trust = trust_data.get('final_trust', {})
            for target_peer_id, trust_value in final_trust.items():
                log_dict[f'Peer{peer_id}/trust/to_peer_{target_peer_id}'] = trust_value
        
        # Single log call for all metrics
        if log_dict:
            wandb.log(log_dict, commit=False, step=self.num_timesteps)
    
    def _print_trust_to_console(self, peer_id: int, trust_data: dict):
        """Print trust values to console in readable format"""
        print(f"\n{'='*80}")
        print(f"TRUST VALUES - Peer {peer_id} (Step {self.num_timesteps})")
        print(f"{'='*80}")
        
        is_attention_peer = trust_data.get('attention_trust') is not None
        
        if is_attention_peer:
            # Attention peer: Show all three trust types
            final_trust = trust_data.get('final_trust', {})
            attention_trust = trust_data.get('attention_trust', {})
            traditional_trust = trust_data.get('traditional_trust', {})
            confidence = trust_data.get('confidence', 0.0)
            omega_t = trust_data.get('omega_t', 0.0)
            
            print(f"Confidence: {confidence:.4f} | Attention Weight (ω_t): {omega_t:.4f}")
            print(f"{'-'*80}")
            print(f"{'Target':<10} {'Traditional':<15} {'Attention':<15} {'Hybrid (Final)':<15}")
            print(f"{'-'*80}")
            
            for target_peer_id in sorted(final_trust.keys()):
                trad_val = traditional_trust.get(target_peer_id, 0.0)
                attn_val = attention_trust.get(target_peer_id, 0.0)
                final_val = final_trust.get(target_peer_id, 0.0)
                
                print(f"Peer {target_peer_id:<5} {trad_val:<15.4f} {attn_val:<15.4f} {final_val:<15.4f}")
        else:
            # Baseline peer: Only traditional trust
            final_trust = trust_data.get('final_trust', {})
            
            print(f"Trust Type: Traditional (Baseline)")
            print(f"{'-'*80}")
            print(f"{'Target':<10} {'Trust Value':<15}")
            print(f"{'-'*80}")
            
            for target_peer_id in sorted(final_trust.keys()):
                trust_val = final_trust.get(target_peer_id, 0.0)
                print(f"Peer {target_peer_id:<5} {trust_val:<15.4f}")
        
        print(f"{'='*80}\n")
    
    def _log_metrics_to_wandb(self):
        """Log global trust metrics to WandB"""
        if wandb.run is None:
            return
        
        # Get latest metrics
        metrics = {}
        for metric_name, values in self.trust_analyzer.metrics.items():
            if values:
                metrics[metric_name] = values[-1][1]
        
        wandb.log({
            'trust/correlation': metrics.get('trust_correlation', 0),
            'trust/fairness': metrics.get('trust_fairness', 0),
            'trust/stability': metrics.get('trust_stability', 0),
            'trust/selection_entropy': metrics.get('selection_entropy', 0),
        }, commit=False)
    
    def _log_trust_matrix_to_wandb(self):
        """Log trust matrix as heatmap to WandB"""
        if wandb.run is None:
            return
        
        try:
            # Build trust matrix
            n_peers = len(self.peer_group.peers)
            trust_matrix = np.zeros((n_peers, n_peers))
            
            for peer_id in range(n_peers):
                peer_obj = self.peer_group.peers[peer_id]
                if hasattr(peer_obj, 'trust_values') and peer_obj.trust_values is not None:
                    trust_matrix[peer_id] = peer_obj.trust_values
                else:
                    # No trust - set to uniform (1.0)
                    trust_matrix[peer_id] = 1.0
            
            # Create trust matrix heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(trust_matrix, cmap='viridis', aspect='auto')
            
            ax.set_xlabel('Target Peer')
            ax.set_ylabel('Source Peer')
            ax.set_title(f'Trust Matrix (Step {self.num_timesteps})')
            ax.set_xticks(range(n_peers))
            ax.set_yticks(range(n_peers))
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Trust Value')
            
            # Add values as text
            for i in range(n_peers):
                for j in range(n_peers):
                    ax.text(j, i, f'{trust_matrix[i, j]:.1f}',
                           ha="center", va="center", color="w", fontsize=10)
            
            # Log to WandB
            wandb.log({
                "trust/trust_matrix_heatmap": wandb.Image(fig)
            }, commit=False)
            
            plt.close(fig)
            
            # Create follow matrix heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(self.follow_matrix, cmap='plasma', aspect='auto')
            
            ax.set_xlabel('Followed Peer')
            ax.set_ylabel('Follower Peer')
            ax.set_title(f'Follow Counts (Step {self.num_timesteps})')
            ax.set_xticks(range(n_peers))
            ax.set_yticks(range(n_peers))
            
            plt.colorbar(im, ax=ax, label='Follow Count')
            
            for i in range(n_peers):
                for j in range(n_peers):
                    ax.text(j, i, f'{int(self.follow_matrix[i, j])}',
                           ha="center", va="center", color="w", fontsize=10)
            
            wandb.log({
                "trust/follow_matrix_heatmap": wandb.Image(fig)
            }, commit=False)
            
            plt.close(fig)
        
        except Exception as e:
            # Log warning but continue - visualization not critical for training
            print(f"Warning: Trust matrix visualization failed at step {self.num_timesteps}: {str(e)}")
    
    def _on_training_end(self) -> None:
        """
        Called at the end of training to generate trust analysis report
        """
        try:
            # Generate full trust analysis report
            log_path = self.log_path if hasattr(self, 'log_path') else '.'
            report_dir = os.path.join(log_path, 'trust_analysis')
            self.trust_analyzer.generate_full_report(report_dir)
        except Exception as e:
            print(f"Warning: Failed to generate trust analysis report: {e}")
        
        # Call parent implementation if exists
        if hasattr(super(), '_on_training_end'):
            super()._on_training_end()
