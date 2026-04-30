#!/usr/bin/env python3
"""
Configuration for KAPALM Ablation Study - GossipCop Dataset
"""

from enum import Enum
from dataclasses import dataclass
from typing import List

class AblationMode(Enum):
    """
    Ablation study configurations based on the KAPALM paper.
    
    Full Model: concat(h, a, s) - text + coarse (with interaction node) + fine-grained (with attentive pooling)
    """
    FULL_MODEL = "full"                    # Full KAPALM: concat(h, a, s)
    WITHOUT_GRAPH_POOLING = "wo_gp"        # w/o Attentive Graph Pooling: concat(h, a)
    WITHOUT_INTERACTION_NODE = "wo_in"     # w/o Interaction Node: concat(h, s)
    FINE_GRAINED_ONLY = "fine_only"        # Fine-Grained Only: concat(h, s) using pruned graph
    COARSE_ONLY = "coarse_only"            # Coarse Only: concat(h, a) using original graph


# Model Configuration
MODEL_CONFIG = {
    'model_name': 'bert-base-uncased',  # BERT for English GossipCop text
    'max_seq_length': 512,
    'batch_size': 32,
    'learning_rate': 1e-5,
    'num_epochs': 100,
    'warmup_steps': 500,
    'max_grad_norm': 1.0,
    'hidden_dim': 768,
    'gat_heads': 8,
    'gat_dropout': 0.3,
    'classifier_dropout': 0.3,
    'num_labels': 2,  # Binary: Real (0) vs Fake (1)
    'centrality_top_n': 20,  # For pruning in applicable modes
}

# Cross-validation Configuration
CV_CONFIG = {
    'n_folds': 5,
    'random_seeds': [42, 123, 456, 789, 1024],  # One seed per fold for reproducibility
}

# Paths Configuration
PATHS = {
    'dataset': r'D:\farrell2\Fake News Politifact\dataset\gossipcop_01.csv',
    'graphs_dir': r'D:\farrell2\Fake News Politifact\processed_graphs_gossipcop\successful',
    'cache_dir': r'D:\farrell2\Fake News Politifact\ablation_study\gossipcop\cache',
    'results_dir': r'D:\farrell2\Fake News Politifact\ablation_study\gossipcop\results',
    'checkpoints_dir': r'D:\farrell2\Fake News Politifact\ablation_study\gossipcop\checkpoints',
}

# Label mapping
LABEL_MAP = {
    0: 'real',
    1: 'fake'
}


@dataclass
class AblationConfig:
    """Configuration for a specific ablation experiment"""
    mode: AblationMode
    use_coarse: bool           # Whether to use coarse-grained representation
    use_fine: bool             # Whether to use fine-grained representation
    use_interaction_node: bool # Whether to use interaction node (added to coarse)
    use_attentive_pooling: bool # Whether to use attentive pooling (for fine)
    use_pruned_graph: bool     # Whether to use centrality-pruned graph
    description: str


# Define all ablation configurations
ABLATION_CONFIGS = {
    AblationMode.FULL_MODEL: AblationConfig(
        mode=AblationMode.FULL_MODEL,
        use_coarse=True,
        use_fine=True,
        use_interaction_node=True,
        use_attentive_pooling=True,
        use_pruned_graph=True,
        description="Full KAPALM: concat(h, a, s) - text + coarse (with IN) + fine (with GP)"
    ),
    AblationMode.WITHOUT_GRAPH_POOLING: AblationConfig(
        mode=AblationMode.WITHOUT_GRAPH_POOLING,
        use_coarse=True,
        use_fine=False,  # Remove fine-grained module
        use_interaction_node=True,
        use_attentive_pooling=False,
        use_pruned_graph=True,
        description="w/o Attentive Graph Pooling: concat(h, a) - removes fine-grained module"
    ),
    AblationMode.WITHOUT_INTERACTION_NODE: AblationConfig(
        mode=AblationMode.WITHOUT_INTERACTION_NODE,
        use_coarse=True,  # Still use coarse GAT but with mean pooling instead of IN
        use_fine=True,
        use_interaction_node=False,  # Remove interaction node
        use_attentive_pooling=True,
        use_pruned_graph=True,
        description="w/o Interaction Node: concat(h, s) - mean pooling instead of IN"
    ),
    AblationMode.FINE_GRAINED_ONLY: AblationConfig(
        mode=AblationMode.FINE_GRAINED_ONLY,
        use_coarse=False,  # No coarse representation
        use_fine=True,
        use_interaction_node=False,
        use_attentive_pooling=True,
        use_pruned_graph=True,  # Use pruned graph for fine-grained
        description="Fine-Grained Only: concat(h, s) - pruned graph with GAT + attentive pooling"
    ),
    AblationMode.COARSE_ONLY: AblationConfig(
        mode=AblationMode.COARSE_ONLY,
        use_coarse=True,
        use_fine=False,
        use_interaction_node=True,
        use_attentive_pooling=False,
        use_pruned_graph=False,  # Use original/full graph
        description="Coarse Only: concat(h, a) - original graph with GAT + interaction node"
    ),
}


def get_ablation_config(mode: AblationMode) -> AblationConfig:
    """Get the configuration for a specific ablation mode"""
    return ABLATION_CONFIGS[mode]


def print_ablation_summary():
    """Print a summary of all ablation configurations"""
    print("=" * 80)
    print("KAPALM ABLATION STUDY CONFIGURATIONS")
    print("=" * 80)
    print("\nNotation:")
    print("  h = text embedding (from BERT)")
    print("  a = coarse-grained knowledge (GAT + interaction node)")
    print("  s = fine-grained knowledge (GAT + attentive pooling)")
    print("\n" + "-" * 80)
    
    for mode, config in ABLATION_CONFIGS.items():
        print(f"\n{mode.value.upper()}: {config.description}")
        print(f"  - Use Coarse: {config.use_coarse}")
        print(f"  - Use Fine: {config.use_fine}")
        print(f"  - Interaction Node: {config.use_interaction_node}")
        print(f"  - Attentive Pooling: {config.use_attentive_pooling}")
        print(f"  - Pruned Graph: {config.use_pruned_graph}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    print_ablation_summary()
