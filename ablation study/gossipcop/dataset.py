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


class GossipCopDataset(Dataset):
    """
    Dataset class for GossipCop fake news detection with graph data.
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
            df: DataFrame with normalized columns: 'id', 'text', 'label'
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
        text = str(row['text'])
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


def _resolve_graphs_dir(graphs_dir: str) -> Path:
    """Resolve graph directory. Supports both root and nested successful folder."""
    root = Path(graphs_dir)
    candidates = [
        root,
        root / 'successful',
        root / 'processed_graphs' / 'successful',
    ]
    for candidate in candidates:
        if candidate.exists() and any(candidate.glob('graph_*.gpickle')):
            return candidate
    return root


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize GossipCop-like datasets to id/text/label schema."""
    id_candidates = ['id', 'ID', 'news_id', 'article_id']
    text_candidates = ['FullText', 'fulltext', 'text', 'content']
    label_candidates = ['label', 'hoax', 'is_fake', 'target']

    id_col = next((c for c in id_candidates if c in df.columns), None)
    text_col = next((c for c in text_candidates if c in df.columns), None)
    label_col = next((c for c in label_candidates if c in df.columns), None)

    missing = []
    if id_col is None:
        missing.append('id')
    if text_col is None:
        missing.append('text')
    if label_col is None:
        missing.append('label')
    if missing:
        raise ValueError(
            f"Could not find required columns {missing} in dataset. Available columns: {list(df.columns)}"
        )

    normalized = df[[id_col, text_col, label_col]].copy()
    normalized.columns = ['id', 'text', 'label']
    normalized['label'] = normalized['label'].astype(int)
    return normalized


def load_and_filter_dataset(dataset_path: str, graphs_dir: str):
    """
    Load dataset and filter to only include samples with available graphs.
    
    Returns:
        Filtered DataFrame
    """
    df_raw = pd.read_csv(dataset_path)
    df = _normalize_columns(df_raw)
    
    # Find available graphs
    graphs_path = _resolve_graphs_dir(graphs_dir)
    available_graphs = set()
    for graph_file in graphs_path.glob("graph_*.gpickle"):
        graph_id = graph_file.stem.replace("graph_", "")
        available_graphs.add(graph_id)
    
    # Filter dataframe
    df['id'] = df['id'].astype(str)
    df_filtered = df[df['id'].isin(available_graphs)].copy()
    
    print(f"Graphs directory in use: {graphs_path}")
    print(f"Dataset: {len(df)} total -> {len(df_filtered)} with graphs")
    print(f"Class distribution: Real={len(df_filtered[df_filtered['label']==0])}, "
          f"Fake={len(df_filtered[df_filtered['label']==1])}")
    
    return df_filtered


# Backward-compatible alias (copied code may still import this name)
IndoMalayDataset = GossipCopDataset
