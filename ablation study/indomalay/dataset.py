#!/usr/bin/env python3
"""
Dataset Classes for KAPALM Ablation Study
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from typing import Optional

from graph_utils import GraphProcessor


class IndoMalayDataset(Dataset):
    """
    Dataset class for Indo-Malay fake news detection with graph data.
    Supports different graph processing modes for ablation studies.
    """
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        tokenizer, 
        graph_processor: GraphProcessor,
        graphs_dir: str, 
        max_length: int = 512,
        use_pruned_graph: bool = True,
        cache_suffix: str = ""
    ):
        """
        Args:
            df: DataFrame with 'id', 'fulltext', 'label' columns
            tokenizer: HuggingFace tokenizer
            graph_processor: GraphProcessor instance
            graphs_dir: Directory containing graph pickle files
            max_length: Maximum sequence length for tokenization
            use_pruned_graph: Whether to use centrality-pruned graphs
            cache_suffix: Suffix for cache files (to distinguish ablation modes)
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.graph_processor = graph_processor
        self.graphs_dir = Path(graphs_dir)
        self.max_length = max_length
        self.use_pruned_graph = use_pruned_graph
        self.cache_suffix = cache_suffix
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        article_id = str(row['id'])
        text = str(row['fulltext'])
        label = int(row['label'])
        
        # Tokenize text
        encoding = self.tokenizer.encode_plus(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Load and process graph data
        graph_path = self.graphs_dir / f"graph_{article_id}.gpickle"
        coarse_graph, fine_graph = self.graph_processor.process_graph_for_ablation(
            str(graph_path), 
            article_id,
            use_pruned=self.use_pruned_graph,
            cache_suffix=self.cache_suffix
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'coarse_graph': coarse_graph,
            'fine_graph': fine_graph,
            'label': torch.tensor(label, dtype=torch.long),
            'article_id': article_id
        }


def load_and_filter_dataset(dataset_path: str, graphs_dir: str):
    """
    Load dataset and filter to only include samples with available graphs.
    
    Returns:
        Filtered DataFrame
    """
    df = pd.read_csv(dataset_path)
    
    # Find available graphs
    graphs_path = Path(graphs_dir)
    available_graphs = set()
    for graph_file in graphs_path.glob("graph_*.gpickle"):
        graph_id = graph_file.stem.replace("graph_", "")
        available_graphs.add(graph_id)
    
    # Filter dataframe
    df['id'] = df['id'].astype(str)
    df_filtered = df[df['id'].isin(available_graphs)].copy()
    
    print(f"Dataset: {len(df)} total -> {len(df_filtered)} with graphs")
    print(f"Class distribution: Real={len(df_filtered[df_filtered['label']==0])}, "
          f"Fake={len(df_filtered[df_filtered['label']==1])}")
    
    return df_filtered
