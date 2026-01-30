"""
Trust Logger - Comprehensive Trust Dynamics Analysis System

Tracks and analyzes trust evolution across peers during training:
- Trust value histories (final, attention-based, traditional)
- Performance metrics and correlations
- Peer selection patterns
- Confidence and attention weight evolution
- Fairness and stability metrics
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from scipy.stats import pearsonr, entropy
from scipy.special import softmax


class TrustAnalyzer:
    """
    Comprehensive trust dynamics analysis and logging system
    
    Tracks:
    - Trust evolution (final, attention, traditional)
    - Performance metrics
    - Peer selection patterns
    - Confidence and attention weights
    - Fairness, stability, and entropy metrics
    """
    
    def __init__(self, n_peers: int, save_dir: str):
        """
        Initialize trust analyzer
        
        Args:
            n_peers: Number of peers in the system
            save_dir: Directory to save analysis outputs
        """
        self.n_peers = n_peers
        self.save_dir = save_dir
        
        # Trust histories: {peer_id: [(step, value), ...]}
        self.trust_history: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        self.attention_trust_history: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        self.traditional_trust_history: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        
        # Performance history: {peer_id: [(step, reward), ...]}
        self.performance_history: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        
        # Confidence and attention weight history: [(step, confidence, omega_t), ...]
        self.confidence_history: List[Tuple[int, float, float]] = []
        
        # Peer selection tracking: follow_counts[i][j] = times peer i followed peer j
        self.follow_counts = np.zeros((n_peers, n_peers))
        
        # Metrics over time: {metric_name: [(step, value), ...]}
        self.metrics: Dict[str, List[Tuple[int, float]]] = {
            'trust_correlation': [],
            'trust_fairness': [],
            'trust_stability': [],
            'selection_entropy': []
        }
    
    def log_trust_step(
        self,
        step: int,
        peer_id: int,
        trust_dict: Dict[int, float],
        attention_trust: Optional[Dict[int, float]] = None,
        traditional_trust: Optional[Dict[int, float]] = None,
        confidence: float = 0.0,
        omega_t: float = 0.0
    ):
        """
        Log trust values at a given step
        
        Args:
            step: Current training step
            peer_id: ID of peer whose trust values are being logged
            trust_dict: Final trust values for all peers
            attention_trust: Attention-based trust values (None for baseline)
            traditional_trust: Traditional trust values (None if not using hybrid)
            confidence: Current confidence metric (0.0 for baseline)
            omega_t: Current attention weight (0.0 for baseline)
        """
        # Log trust values for each peer
        for pid in range(self.n_peers):
            if pid in trust_dict:
                self.trust_history[pid].append((step, trust_dict[pid]))
            if attention_trust and pid in attention_trust:
                self.attention_trust_history[pid].append((step, attention_trust[pid]))
            if traditional_trust and pid in traditional_trust:
                self.traditional_trust_history[pid].append((step, traditional_trust[pid]))
        
        # Log confidence and attention weight (only if using attention)
        if attention_trust is not None:
            self.confidence_history.append((step, confidence, omega_t))
    
    def log_peer_selection(self, step: int, follower: int, followed: int):
        """
        Log when one peer follows another
        
        Args:
            step: Current training step
            follower: ID of peer doing the following
            followed: ID of peer being followed
        """
        self.follow_counts[follower, followed] += 1
    
    def log_performance(self, step: int, peer_id: int, reward: float):
        """
        Log performance metric for a peer
        
        Args:
            step: Current training step
            peer_id: ID of peer
            reward: Mean evaluation reward
        """
        self.performance_history[peer_id].append((step, reward))
    
    def compute_trust_performance_correlation(
        self,
        peer_id: int,
        window: int = 10
    ) -> float:
        """
        Compute correlation between trust RECEIVED BY peer and their performance
        
        Measures: Do high-performing peers receive more trust from others?
        
        Args:
            peer_id: Peer to analyze
            window: Number of recent points to consider
        
        Returns:
            Pearson correlation coefficient
        """
        perf_data = self.performance_history[peer_id]
        
        if len(perf_data) < 2:
            return 0.0
        
        # Compute average trust RECEIVED by this peer from ALL OTHER peers
        # For each time step, average the trust that other peers have in peer_id
        trust_received_history = []
        
        for step, _ in perf_data[-window:]:
            # Get trust values from all OTHER peers towards peer_id at this step
            trust_values_at_step = []
            for other_peer in range(self.n_peers):
                if other_peer == peer_id:
                    continue  # Skip self-trust
                
                # Find trust value from other_peer's history closest to this step
                other_trust_history = self.trust_history.get(peer_id, [])
                if other_trust_history:
                    # Find value closest to step
                    closest = min(other_trust_history, key=lambda x: abs(x[0] - step))
                    if abs(closest[0] - step) < 1000:  # Within reasonable window
                        trust_values_at_step.append(closest[1])
            
            if trust_values_at_step:
                trust_received_history.append(np.mean(trust_values_at_step))
        
        if len(trust_received_history) < 2:
            return 0.0
        
        # Get performance values
        recent_perf = perf_data[-window:] if len(perf_data) > window else perf_data
        perf_vals = [v for _, v in recent_perf[-len(trust_received_history):]]
        
        # Ensure same length
        min_len = min(len(trust_received_history), len(perf_vals))
        if min_len < 2:
            return 0.0
        
        trust_received = trust_received_history[-min_len:]
        perf_vals = perf_vals[-min_len:]
        
        # Check if arrays are constant (avoid correlation warning)
        if np.std(trust_received) == 0 or np.std(perf_vals) == 0:
            return 0.0
        
        # Compute correlation
        try:
            corr, _ = pearsonr(trust_received, perf_vals)
            return float(corr) if not np.isnan(corr) else 0.0
        except:
            return 0.0
    
    def compute_trust_fairness(self) -> float:
        """
        Compute Gini coefficient of ALL trust relationships in the system
        
        Measures: How evenly is trust distributed across all peer pairs?
        Lower values (closer to 0) = more equal distribution
        Higher values (closer to 1) = more concentrated trust
        
        Returns:
            Gini coefficient [0, 1]
        """
        # Collect ALL trust values from all peer-to-peer relationships
        all_trust_values = []
        
        for peer_id in range(self.n_peers):
            if self.trust_history[peer_id]:
                # Get latest trust value this peer has in each other peer
                latest_trust = self.trust_history[peer_id][-1][1]
                all_trust_values.append(latest_trust)
        
        if len(all_trust_values) < 2:
            return 0.0
        
        # Compute Gini coefficient
        sorted_trusts = np.sort(all_trust_values)
        n = len(sorted_trusts)
        trust_sum = np.sum(sorted_trusts)
        
        # Avoid division by zero
        if trust_sum == 0 or np.abs(trust_sum) < 1e-10:
            return 0.0
        
        # Gini coefficient formula
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_trusts)) / (n * trust_sum) - (n + 1) / n
        
        return float(gini) if not np.isnan(gini) else 0.0
    
    def compute_trust_stability(self, peer_id: int, window: int = 10) -> float:
        """
        Compute stability of trust values over time
        
        Measures: How much does this peer's trust assessment change?
        Lower std = more stable trust values
        Higher std = more volatile trust assessments
        
        Args:
            peer_id: Peer to analyze
            window: Number of recent points to consider
        
        Returns:
            Standard deviation of trust changes (lower = more stable)
        """
        trust_data = self.trust_history[peer_id]
        
        if len(trust_data) < 2:
            return 0.0
        
        # Get recent values
        recent = trust_data[-window:] if len(trust_data) > window else trust_data
        values = [v for _, v in recent]
        
        # Compute differences between consecutive values
        if len(values) < 2:
            return 0.0
        
        diffs = np.diff(values)
        
        # Return std of changes (lower = more stable)
        return float(np.std(diffs)) if len(diffs) > 0 else 0.0
    
    def compute_selection_entropy(self, peer_id: int, window: int = 100) -> float:
        """
        Compute entropy of peer selection distribution
        
        Higher entropy means more diverse selection.
        
        Args:
            peer_id: Peer whose selections to analyze
            window: Window size (not used in current implementation)
        
        Returns:
            Selection entropy
        """
        # Get selection counts for this peer
        selections = self.follow_counts[peer_id]
        
        if np.sum(selections) == 0:
            return 0.0
        
        # Compute probability distribution
        probs = selections / np.sum(selections)
        
        # Filter out zeros
        probs = probs[probs > 0]
        
        # Compute entropy
        ent = entropy(probs)
        
        return float(ent) if not np.isnan(ent) else 0.0
    
    def update_metrics(self, step: int):
        """
        Compute and store all metrics at current step
        
        Metrics computed:
        1. Trust-Performance Correlation: Do high performers receive more trust?
        2. Trust Fairness (Gini): How evenly distributed is trust in the system?
        3. Trust Stability: How volatile are trust assessments?
        4. Selection Entropy: How diverse are peer selections?
        
        Args:
            step: Current training step
        """
        # Trust-performance correlation (average across ALL peers)
        # Measures: "Do peers who perform better receive more trust from others?"
        correlations = []
        for peer_id in range(self.n_peers):
            corr = self.compute_trust_performance_correlation(peer_id)
            correlations.append(corr)
        avg_corr = np.mean(correlations) if correlations else 0.0
        self.metrics['trust_correlation'].append((step, avg_corr))
        
        # Trust fairness (Gini coefficient of ALL trust relationships)
        # Lower = more equal, Higher = more concentrated
        # Measures: "Is trust evenly distributed or do some peers monopolize it?"
        fairness = self.compute_trust_fairness()
        self.metrics['trust_fairness'].append((step, fairness))
        
        # Trust stability (average std of changes across peers)
        # Lower = more stable, Higher = more volatile
        # Measures: "How much do trust assessments fluctuate over time?"
        stabilities = []
        for peer_id in range(self.n_peers):
            stab = self.compute_trust_stability(peer_id)
            stabilities.append(stab)
        avg_stab = np.mean(stabilities) if stabilities else 0.0
        self.metrics['trust_stability'].append((step, avg_stab))
        
        # Selection entropy (average across peers)
        # Higher = more diverse, Lower = more concentrated selection
        # Measures: "Do peers diversify their selections or stick to favorites?"
        entropies = []
        for peer_id in range(self.n_peers):
            ent = self.compute_selection_entropy(peer_id)
            entropies.append(ent)
        avg_ent = np.mean(entropies) if entropies else 0.0
        self.metrics['selection_entropy'].append((step, avg_ent))
    
    def plot_trust_evolution(self, peer_id: int, save_path: str):
        """
        Plot comprehensive trust evolution for a peer
        
        Creates 2x2 subplot:
        - Final trust over time
        - Attention vs Traditional trust
        - Confidence and omega_t evolution
        - Trust vs Performance correlation
        
        Args:
            peer_id: Peer to plot
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Trust Evolution - Peer {peer_id}', fontsize=16)
        
        # Plot 1: Final trust over time
        ax = axes[0, 0]
        if self.trust_history[peer_id]:
            steps, values = zip(*self.trust_history[peer_id])
            ax.plot(steps, values, 'b-', linewidth=2, label='Final Trust')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Trust Value')
            ax.set_title('Final Trust Evolution')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Plot 2: Attention vs Traditional trust
        ax = axes[0, 1]
        has_attention = self.attention_trust_history[peer_id] and len(self.attention_trust_history[peer_id]) > 0
        has_traditional = self.traditional_trust_history[peer_id] and len(self.traditional_trust_history[peer_id]) > 0
        
        if has_attention and has_traditional:
            attn_steps, attn_vals = zip(*self.attention_trust_history[peer_id])
            trad_steps, trad_vals = zip(*self.traditional_trust_history[peer_id])
            ax.plot(attn_steps, attn_vals, 'r-', linewidth=2, label='Attention Trust', alpha=0.7)
            ax.plot(trad_steps, trad_vals, 'g-', linewidth=2, label='Traditional Trust', alpha=0.7)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Trust Value')
            ax.set_title('Attention vs Traditional Trust')
            ax.grid(True, alpha=0.3)
            ax.legend()
        elif has_traditional:
            # Baseline: only traditional trust
            trad_steps, trad_vals = zip(*self.traditional_trust_history[peer_id])
            ax.plot(trad_steps, trad_vals, 'g-', linewidth=2, label='Trust (Baseline)', alpha=0.7)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Trust Value')
            ax.set_title('Trust Evolution (Baseline)')
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No trust data available', ha='center', va='center', transform=ax.transAxes)
        
        # Plot 3: Confidence and omega_t
        ax = axes[1, 0]
        if self.confidence_history and len(self.confidence_history) > 0:
            steps, confidences, omegas = zip(*self.confidence_history)
            ax2 = ax.twinx()
            l1 = ax.plot(steps, confidences, 'purple', linewidth=2, label='Confidence', alpha=0.7)
            l2 = ax2.plot(steps, omegas, 'orange', linewidth=2, label='$\\omega_t$', alpha=0.7)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Confidence', color='purple')
            ax2.set_ylabel('Attention Weight ($\\omega_t$)', color='orange')
            ax.set_title('Confidence and Attention Weight Evolution')
            ax.grid(True, alpha=0.3)
            # Combine legends
            lns = l1 + l2
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs, loc='best')
        else:
            # Baseline: no attention system
            ax.text(0.5, 0.5, 'N/A (Baseline - No Attention)', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Confidence and Attention Weight (N/A for Baseline)')
        
        # Plot 4: Trust vs Performance
        ax = axes[1, 1]
        if self.trust_history[peer_id] and self.performance_history[peer_id]:
            trust_steps, trust_vals = zip(*self.trust_history[peer_id])
            perf_steps, perf_vals = zip(*self.performance_history[peer_id])
            
            ax2 = ax.twinx()
            l1 = ax.plot(trust_steps, trust_vals, 'b-', linewidth=2, label='Trust', alpha=0.7)
            l2 = ax2.plot(perf_steps, perf_vals, 'darkgreen', linewidth=2, label='Performance', alpha=0.7)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Trust Value', color='b')
            ax2.set_ylabel('Mean Reward', color='darkgreen')
            ax.set_title('Trust vs Performance')
            ax.grid(True, alpha=0.3)
            # Combine legends
            lns = l1 + l2
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs, loc='best')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_trust_heatmap(self, step: int, save_path: str):
        """
        Plot heatmaps of trust matrices at given step
        
        Creates 3 heatmaps: Final, Attention, Traditional trust
        
        Args:
            step: Training step to visualize
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Trust Matrices at Step {step}', fontsize=16)
        
        # Build trust matrices
        final_matrix = np.zeros((self.n_peers, self.n_peers))
        attn_matrix = np.zeros((self.n_peers, self.n_peers))
        trad_matrix = np.zeros((self.n_peers, self.n_peers))
        
        for peer_id in range(self.n_peers):
            for other_id in range(self.n_peers):
                # Get value closest to requested step
                if self.trust_history[other_id]:
                    final_matrix[peer_id, other_id] = self._get_value_at_step(
                        self.trust_history[other_id], step
                    )
                if self.attention_trust_history[other_id]:
                    attn_matrix[peer_id, other_id] = self._get_value_at_step(
                        self.attention_trust_history[other_id], step
                    )
                if self.traditional_trust_history[other_id]:
                    trad_matrix[peer_id, other_id] = self._get_value_at_step(
                        self.traditional_trust_history[other_id], step
                    )
        
        # Plot final trust
        im1 = axes[0].imshow(final_matrix, cmap='viridis', aspect='auto')
        axes[0].set_title('Final Trust')
        axes[0].set_xlabel('Trusted Peer')
        axes[0].set_ylabel('Trusting Peer')
        plt.colorbar(im1, ax=axes[0])
        
        # Plot attention trust
        im2 = axes[1].imshow(attn_matrix, cmap='Reds', aspect='auto')
        axes[1].set_title('Attention Trust')
        axes[1].set_xlabel('Trusted Peer')
        axes[1].set_ylabel('Trusting Peer')
        plt.colorbar(im2, ax=axes[1])
        
        # Plot traditional trust
        im3 = axes[2].imshow(trad_matrix, cmap='Greens', aspect='auto')
        axes[2].set_title('Traditional Trust')
        axes[2].set_xlabel('Trusted Peer')
        axes[2].set_ylabel('Trusting Peer')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _get_value_at_step(self, history: List[Tuple[int, float]], target_step: int) -> float:
        """Get value from history closest to target step"""
        if not history:
            return 0.0
        
        # Find closest step
        closest = min(history, key=lambda x: abs(x[0] - target_step))
        return closest[1]
    
    def plot_metrics_summary(self, save_path: str):
        """
        Plot all metrics over time in 2x2 grid
        
        Args:
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Trust Dynamics Metrics', fontsize=16)
        
        metric_titles = {
            'trust_correlation': 'Trust-Performance Correlation',
            'trust_fairness': 'Trust Fairness (Gini Coefficient)',
            'trust_stability': 'Trust Stability (Std of Changes)',
            'selection_entropy': 'Selection Entropy'
        }
        
        for idx, (metric_name, title) in enumerate(metric_titles.items()):
            ax = axes[idx // 2, idx % 2]
            
            if self.metrics[metric_name]:
                steps, values = zip(*self.metrics[metric_name])
                ax.plot(steps, values, linewidth=2)
                ax.set_xlabel('Training Step')
                ax.set_ylabel(metric_name.replace('_', ' ').title())
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_to_json(self, filepath: str):
        """
        Export all data to JSON format
        
        Args:
            filepath: Path to save JSON file
        """
        data = {
            'n_peers': self.n_peers,
            'trust_history': {
                str(k): [(int(s), float(v)) for s, v in v_list]
                for k, v_list in self.trust_history.items()
            },
            'attention_trust_history': {
                str(k): [(int(s), float(v)) for s, v in v_list]
                for k, v_list in self.attention_trust_history.items()
            },
            'traditional_trust_history': {
                str(k): [(int(s), float(v)) for s, v in v_list]
                for k, v_list in self.traditional_trust_history.items()
            },
            'performance_history': {
                str(k): [(int(s), float(v)) for s, v in v_list]
                for k, v_list in self.performance_history.items()
            },
            'confidence_history': [
                (int(s), float(c), float(o)) for s, c, o in self.confidence_history
            ],
            'follow_counts': self.follow_counts.tolist(),
            'metrics': {
                k: [(int(s), float(v)) for s, v in v_list]
                for k, v_list in self.metrics.items()
            }
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def generate_full_report(self, output_dir: str):
        """
        Generate complete analysis report with all plots and data
        
        Args:
            output_dir: Directory to save all outputs
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # PNG generation disabled - using wandb for visualization
        # Uncomment below if you need local PNG files:
        
        # # Generate per-peer trust evolution plots
        # for peer_id in range(self.n_peers):
        #     plot_path = os.path.join(output_dir, f'peer_{peer_id}_trust_evolution.png')
        #     self.plot_trust_evolution(peer_id, plot_path)
        
        # # Generate trust heatmaps at final step
        # if self.confidence_history:
        #     final_step = self.confidence_history[-1][0]
        #     heatmap_path = os.path.join(output_dir, 'trust_heatmaps_final.png')
        #     self.plot_trust_heatmap(final_step, heatmap_path)
        
        # # Generate metrics summary
        # metrics_path = os.path.join(output_dir, 'metrics_summary.png')
        # self.plot_metrics_summary(metrics_path)
        
        # Export data to JSON only
        json_path = os.path.join(output_dir, 'trust_analysis_data.json')
        self.export_to_json(json_path)
