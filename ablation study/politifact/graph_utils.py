#!/usr/bin/env python3
"""
Graph Processing Utilities for KAPALM Ablation Study
Handles graph loading, pruning, and conversion to PyTorch Geometric format
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import networkx as nx

try:
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: PyTorch Geometric not available")

logger = logging.getLogger(__name__)


class GraphProcessor:
    """
    Handles graph loading, pruning, and conversion to PyTorch Geometric format.
    Supports different pruning strategies for ablation studies.
    """
    
    def __init__(self, cache_dir: str, centrality_top_n: int = 20):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.centrality_top_n = centrality_top_n
        
    def load_graph(self, graph_path: str) -> Optional[nx.Graph]:
        """Load a NetworkX graph from pickle file"""
        try:
            with open(graph_path, 'rb') as f:
                G = pickle.load(f)
            return G
        except Exception as e:
            logger.warning(f"Failed to load graph {graph_path}: {e}")
            return None
    
    def compute_centrality_pruning(self, G: nx.Graph) -> nx.Graph:
        """Prune graph based on degree centrality - keeps top-N central nodes"""
        if G.number_of_nodes() <= self.centrality_top_n:
            return G.copy()
        
        # Compute degree centrality
        centrality = nx.degree_centrality(G)
        
        # Get top-N nodes by centrality
        top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:self.centrality_top_n]
        top_node_ids = [node for node, _ in top_nodes]
        
        # Create subgraph with top nodes
        pruned_G = G.subgraph(top_node_ids).copy()
        
        logger.debug(f"Centrality pruning: {G.number_of_nodes()} -> {pruned_G.number_of_nodes()} nodes")
        return pruned_G
    
    def first_degree_pruning(self, G: nx.Graph, original_entities: List[str] = None) -> nx.Graph:
        """Further prune by keeping only first-degree neighbors of original entities"""
        if not original_entities:
            original_entities = []
            
        # Find nodes that are original entities
        entity_nodes = []
        for node in G.nodes():
            node_data = G.nodes[node]
            if (node_data.get('type') == 'main_entity' or 
                node in original_entities or
                node_data.get('label', '').lower() in [e.lower() for e in original_entities]):
                entity_nodes.append(node)
        
        if not entity_nodes:
            return G.copy()
        
        # Get first-degree neighbors
        neighbors = set(entity_nodes)
        for entity in entity_nodes:
            neighbors.update(G.neighbors(entity))
        
        first_degree_G = G.subgraph(neighbors).copy()
        logger.debug(f"First-degree pruning: {G.number_of_nodes()} -> {first_degree_G.number_of_nodes()} nodes")
        return first_degree_G
    
    def graph_to_pyg_data(self, G: nx.Graph) -> Optional[Data]:
        """Convert NetworkX graph to PyTorch Geometric Data object"""
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for graph processing")
            
        if G.number_of_nodes() == 0:
            return None
            
        # Create node mapping
        node_list = list(G.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        
        # Node features (use node degree)
        degrees = [G.degree(node) for node in node_list]
        node_features = torch.tensor(degrees, dtype=torch.float).unsqueeze(1)
        
        # Edge indices
        edge_list = []
        for edge in G.edges():
            src, dst = edge
            edge_list.append([node_to_idx[src], node_to_idx[dst]])
            edge_list.append([node_to_idx[dst], node_to_idx[src]])  # Undirected
            
        if not edge_list:
            # No edges, create self-loops
            edge_list = [[i, i] for i in range(len(node_list))]
            
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        return Data(x=node_features, edge_index=edge_index)
    
    def process_graph_for_ablation(
        self, 
        graph_path: str, 
        article_id: str,
        use_pruned: bool = True,
        cache_suffix: str = ""
    ) -> Tuple[Optional[Data], Optional[Data]]:
        """
        Process a graph for ablation study.
        
        Args:
            graph_path: Path to the graph pickle file
            article_id: Unique identifier for caching
            use_pruned: If True, apply centrality pruning; if False, use original graph
            cache_suffix: Suffix for cache file to distinguish different ablation modes
            
        Returns:
            (coarse_data, fine_data) - PyTorch Geometric Data objects
        """
        # Check cache
        cache_file = self.cache_dir / f"graph_{article_id}{cache_suffix}.pt"
        if cache_file.exists():
            try:
                cached = torch.load(cache_file)
                return cached['coarse'], cached['fine']
            except:
                pass
        
        # Load original graph
        G = self.load_graph(graph_path)
        if G is None:
            return None, None
        
        if use_pruned:
            # Apply centrality pruning for coarse representation
            coarse_G = self.compute_centrality_pruning(G)
            # Apply first-degree pruning for fine representation
            fine_G = self.first_degree_pruning(coarse_G)
        else:
            # Use original graph (no pruning)
            coarse_G = G.copy()
            fine_G = G.copy()
        
        coarse_data = self.graph_to_pyg_data(coarse_G)
        fine_data = self.graph_to_pyg_data(fine_G)
        
        # Cache results
        torch.save({'coarse': coarse_data, 'fine': fine_data}, cache_file)
        
        return coarse_data, fine_data


def collate_fn(batch):
    """Custom collate function for batching with graphs"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_masks = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    # Handle graph data
    coarse_graphs = [item['coarse_graph'] for item in batch if item['coarse_graph'] is not None]
    fine_graphs = [item['fine_graph'] for item in batch if item['fine_graph'] is not None]
    
    # Create batch objects for graphs
    if TORCH_GEOMETRIC_AVAILABLE:
        coarse_batch = Batch.from_data_list(coarse_graphs) if coarse_graphs else None
        fine_batch = Batch.from_data_list(fine_graphs) if fine_graphs else None
    else:
        coarse_batch = None
        fine_batch = None
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'coarse_batch': coarse_batch,
        'fine_batch': fine_batch,
        'labels': labels,
        'article_ids': [item['article_id'] for item in batch]
    }
