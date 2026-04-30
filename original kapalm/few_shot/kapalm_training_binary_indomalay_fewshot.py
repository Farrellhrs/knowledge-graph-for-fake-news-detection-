#!/usr/bin/env python3
"""
Original KAPALM (Knowledge grAPh Attentive Language Model) for Fake News Detection
Few-Shot Learning with Indo-Malay Dataset

Original KAPALM WITHOUT centrality-based pruning:
- Uses full knowledge graphs without centrality-based node selection
- Maintains dual representation (coarse + fine-grained via first-degree pruning)

Following the paper methodology:
- k ∈ {2, 4, 8, 16, 100} samples per class for training
- Validation set of the same size as training
- Test set = remaining samples
- 10 random seeds, remove max and min, average remaining 8 scores (trimmed mean)
"""

import os
import pickle
import random
import logging
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, roc_auc_score

import transformers
from transformers import (
    BertTokenizer, BertModel, BertConfig,
    AutoTokenizer, AutoModel
)
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW

# Try to import adapter-transformers, fallback to regular transformers
try:
    from adapters import BertAdapterModel
    from adapters.composition import Stack
    ADAPTERS_AVAILABLE = True
    print("Using adapter-transformers for Adapter Tuning")
except ImportError:
    ADAPTERS_AVAILABLE = False
    print("adapter-transformers not available, using regular BERT fine-tuning")

# PyTorch Geometric imports
try:
    import torch_geometric
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool
    from torch_geometric.loader import DataLoader as GeometricDataLoader
    TORCH_GEOMETRIC_AVAILABLE = True
    print("PyTorch Geometric available for GNN components")
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("PyTorch Geometric not available, implementing basic GNN layers")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration for Binary Classification - Indo-Malay Dataset with IndoBERT
# Original KAPALM: No centrality-based pruning - uses full knowledge graphs
CONFIG = {
    'model_name': 'indolem/indobert-base-uncased',  # IndoBERT for Indonesian/Malay text
    'max_seq_length': 512,
    'batch_size': 32,
    'learning_rate': 1e-5,  
    'num_epochs': 50,  # Reduced for few-shot (can be adjusted)
    'warmup_steps': 100,  # Reduced warmup for small datasets
    'max_grad_norm': 1.0,
    'hidden_dim': 768,
    'gat_heads': 8,
    'gat_dropout': 0.3,
    'classifier_dropout': 0.3,
    'num_labels': 2,  # Binary classification: 0 (Real) vs 1 (Fake/Hoax)
    'base_seed': 42,
    'cache_dir': './cache_binary_indomalay_fewshot',
    'save_path': './kapalm_binary_indomalay_fewshot.pt',
    'log_file': './training_log_binary_indomalay_fewshot.csv'
}

# Few-shot experiment configuration following the paper
FEWSHOT_CONFIG = {
    'shot_sizes': [2, 4, 8, 16, 100],  # k ∈ {2, 4, 8, 16, 100}
    'num_seeds': 10,  # Run with 10 different random seeds
    'random_seeds': [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]  # 10 seeds
}

# Binary Label mapping for Indo-Malay Dataset
LABEL_MAP = {
    0: 'real',
    1: 'fake'  # hoax/misinformation
}

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GraphProcessor:
    """Handles graph loading, pruning, and conversion to PyTorch Geometric format
    
    Original KAPALM: No centrality-based pruning - uses full knowledge graphs
    """
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        # Original KAPALM: No centrality-based pruning
        
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
        """
        Original KAPALM: No centrality-based pruning
        Returns the full graph without any centrality-based node selection
        """
        return G.copy()
    
    def first_degree_pruning(self, G: nx.Graph, original_entities: List[str]) -> nx.Graph:
        """Further prune by keeping only first-degree neighbors of original entities"""
        if not original_entities:
            return G.copy()
        
        # Find nodes that are original entities (based on node attributes or naming)
        entity_nodes = []
        for node in G.nodes():
            node_data = G.nodes[node]
            # Check if this is an original entity (main_entity type or in original_entities list)
            if (node_data.get('type') == 'main_entity' or 
                node in original_entities or
                node_data.get('label', '').lower() in [e.lower() for e in original_entities]):
                entity_nodes.append(node)
        
        if not entity_nodes:
            # If no entity nodes found, return the graph as is
            return G.copy()
        
        # Get first-degree neighbors of entity nodes
        neighbors = set(entity_nodes)
        for entity in entity_nodes:
            neighbors.update(G.neighbors(entity))
        
        # Create subgraph
        first_degree_G = G.subgraph(neighbors).copy()
        
        logger.debug(f"First-degree pruning: {G.number_of_nodes()} -> {first_degree_G.number_of_nodes()} nodes")
        return first_degree_G
    
    def graph_to_pyg_data(self, G: nx.Graph) -> Optional[Data]:
        """Convert NetworkX graph to PyTorch Geometric Data object"""
        if G.number_of_nodes() == 0:
            return None
            
        # Create node mapping
        node_list = list(G.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        
        # Node features (simple: use node degree, can be enhanced)
        degrees = [G.degree(node) for node in node_list]
        node_features = torch.tensor(degrees, dtype=torch.float).unsqueeze(1)
        
        # Edge indices
        edge_list = []
        for edge in G.edges():
            src, dst = edge
            edge_list.append([node_to_idx[src], node_to_idx[dst]])
            edge_list.append([node_to_idx[dst], node_to_idx[src]])  # Add reverse edge for undirected
            
        if not edge_list:
            # No edges, create self-loops
            edge_list = [[i, i] for i in range(len(node_list))]
            
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        return Data(x=node_features, edge_index=edge_index)
    
    def process_graph(self, graph_path: str, article_id: str, original_entities: Optional[List[str]] = None) -> Tuple[Optional[Data], Optional[Data]]:
        """
        Process a graph file to create both coarse and fine-grained representations
        Returns: (coarse_grained_data, fine_grained_data)
        """
        # Check cache first
        cache_file = self.cache_dir / f"processed_graph_{article_id}.pt"
        if cache_file.exists():
            try:
                cached_data = torch.load(cache_file)
                return cached_data['coarse'], cached_data['fine']
            except:
                pass  # Cache corrupted, recompute
        
        # Load original graph
        G = self.load_graph(graph_path)
        if G is None:
            return None, None
            
        # Step 1: Centrality-based pruning for coarse-grained representation
        coarse_G = self.compute_centrality_pruning(G)
        coarse_data = self.graph_to_pyg_data(coarse_G)
        
        # Step 2: First-degree pruning on centrality-pruned graph for fine-grained representation  
        fine_G = self.first_degree_pruning(coarse_G, original_entities or [])
        fine_data = self.graph_to_pyg_data(fine_G)
        
        # Cache the results
        torch.save({'coarse': coarse_data, 'fine': fine_data}, cache_file)
        
        return coarse_data, fine_data


class FakeNewsDataset(Dataset):
    """Dataset class for fake news detection with graph data"""
    
    def __init__(self, df: pd.DataFrame, tokenizer, graph_processor: GraphProcessor, 
                 graphs_dir: str, max_length: int = 512):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.graph_processor = graph_processor
        self.graphs_dir = Path(graphs_dir)
        self.max_length = max_length
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        article_id = str(row['id'])
        text = str(row['fulltext'])  # Indo-Malay dataset uses 'fulltext' column
        label = int(row['label'])    # Indo-Malay dataset uses 'label' column
        
        # Tokenize text
        encoding = self.tokenizer.encode_plus(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Load graph data
        graph_path = self.graphs_dir / f"graph_{article_id}.gpickle"
        coarse_graph, fine_graph = self.graph_processor.process_graph(str(graph_path), article_id)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'coarse_graph': coarse_graph,
            'fine_graph': fine_graph,
            'label': torch.tensor(label, dtype=torch.long),
            'article_id': article_id
        }


def collate_fn(batch):
    """Custom collate function for batching with graphs"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_masks = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    # Handle graph data
    coarse_graphs = [item['coarse_graph'] for item in batch if item['coarse_graph'] is not None]
    fine_graphs = [item['fine_graph'] for item in batch if item['fine_graph'] is not None]
    
    # Create batch objects for graphs
    coarse_batch = Batch.from_data_list(coarse_graphs) if coarse_graphs else None
    fine_batch = Batch.from_data_list(fine_graphs) if fine_graphs else None
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'coarse_batch': coarse_batch,
        'fine_batch': fine_batch,
        'labels': labels,
        'article_ids': [item['article_id'] for item in batch]
    }


class GraphAttentionNetwork(nn.Module):
    """Graph Attention Network for processing knowledge graphs"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, heads: int = 8, dropout: float = 0.3):
        super().__init__()
        
        if TORCH_GEOMETRIC_AVAILABLE:
            self.gat1 = GATConv(input_dim, hidden_dim // heads, heads=heads, dropout=dropout, concat=True)
            self.gat2 = GATConv(hidden_dim, output_dim, heads=1, dropout=dropout, concat=False)
        else:
            # Fallback implementation
            self.linear1 = nn.Linear(input_dim, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, output_dim)
            self.dropout = nn.Dropout(dropout)
            
        self.dropout_layer = nn.Dropout(dropout)
        self.output_dim = output_dim
        
    def forward(self, x, edge_index, batch):
        if TORCH_GEOMETRIC_AVAILABLE:
            # Use PyTorch Geometric GAT
            x = F.relu(self.gat1(x, edge_index))
            x = self.dropout_layer(x)
            x = self.gat2(x, edge_index)
            
            # Global pooling
            out = global_mean_pool(x, batch)
        else:
            # Fallback: simple linear transformation
            x = F.relu(self.linear1(x))
            x = self.dropout(x)
            x = self.linear2(x)
            
            # Simple mean pooling
            if batch is not None:
                batch_size = batch.max().item() + 1
                out = torch.zeros(batch_size, self.output_dim, device=x.device)
                for i in range(batch_size):
                    mask = (batch == i)
                    if mask.sum() > 0:
                        out[i] = x[mask].mean(dim=0)
            else:
                out = x.mean(dim=0, keepdim=True)
                
        return out


class AttentiveGraphPooling(nn.Module):
    """Simplified attentive pooling for fine-grained graph representation"""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads=4, batch_first=True)
        self.linear = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x, batch=None):
        # Simple mean pooling since we process graphs individually
        if x.size(0) == 0:
            return torch.zeros(1, x.size(1), device=x.device)
        
        # Apply self-attention if we have multiple nodes
        if x.size(0) > 1:
            x_unsqueezed = x.unsqueeze(0)  # Add batch dimension
            attended, _ = self.attention(x_unsqueezed, x_unsqueezed, x_unsqueezed)
            pooled = attended.squeeze(0).mean(dim=0)
        else:
            pooled = x.squeeze(0)
            
        return self.linear(pooled)


class KAPALMModel(nn.Module):
    """
    Modified KAPALM model with centrality-based pruning and dual representations
    Using IndoBERT for Indonesian/Malay text
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Text encoder (IndoBERT with adapters if available)
        if ADAPTERS_AVAILABLE:
            self.bert = BertAdapterModel.from_pretrained(config['model_name'])
            # Add adapter for classification
            adapter_name = "fake_news_adapter"
            self.bert.add_adapter(adapter_name, config="pfeiffer")
            self.bert.train_adapter(adapter_name)
            # Set the active adapter - pass as list
            self.bert.set_active_adapters([adapter_name])
            print(f"Using IndoBERT ({config['model_name']}) with adapter tuning")
        else:
            # Use AutoModel for IndoBERT compatibility
            self.bert = AutoModel.from_pretrained(config['model_name'])
            # Freeze BERT parameters for efficiency (can be unfrozen if needed)
            for param in self.bert.parameters():
                param.requires_grad = False
            # Unfreeze the last layer
            for param in self.bert.encoder.layer[-1].parameters():
                param.requires_grad = True
            print(f"Using IndoBERT ({config['model_name']}) with last layer fine-tuning")
        
        # Graph processors
        self.coarse_gat = GraphAttentionNetwork(
            input_dim=1,  # Node degree as input feature
            hidden_dim=config['hidden_dim'],
            output_dim=config['hidden_dim'] // 2,
            heads=config['gat_heads'],
            dropout=config['gat_dropout']
        )
        
        self.fine_gat = GraphAttentionNetwork(
            input_dim=1,
            hidden_dim=config['hidden_dim'],
            output_dim=config['hidden_dim'] // 4,
            heads=config['gat_heads'] // 2,
            dropout=config['gat_dropout']
        )
        
        self.fine_pooling = AttentiveGraphPooling(
            input_dim=config['hidden_dim'] // 4,
            hidden_dim=config['hidden_dim'] // 4
        )
        
        # Interaction node for coarse-grained representation
        self.interaction_embedding = nn.Parameter(torch.randn(config['hidden_dim'] // 2))
        
        # Fusion layer
        fusion_input_dim = config['hidden_dim'] + (config['hidden_dim'] // 2) + (config['hidden_dim'] // 4)
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(config['classifier_dropout']),
            nn.Linear(config['hidden_dim'], config['num_labels'])
        )
        
    def forward(self, input_ids, attention_mask, coarse_batch, fine_batch):
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Text encoding - use output_hidden_states for adapter models
        if ADAPTERS_AVAILABLE:
            # For adapter models, get hidden states directly
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            # Use the last hidden state from the hidden_states tuple
            last_hidden_states = bert_outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
            # Mean pooling over sequence length (excluding padding tokens)
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
            sum_embeddings = torch.sum(last_hidden_states * attention_mask_expanded, 1)
            sum_mask = attention_mask_expanded.sum(1)
            text_embedding = sum_embeddings / torch.clamp(sum_mask, min=1e-9)  # [batch_size, hidden_dim]
        else:
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            # Handle different output formats for regular BERT
            if hasattr(bert_outputs, 'pooler_output') and bert_outputs.pooler_output is not None:
                # Regular BERT model with pooler
                text_embedding = bert_outputs.pooler_output  # [batch_size, hidden_dim]
            else:
                # Use mean pooling of last hidden states
                last_hidden_states = bert_outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
                # Mean pooling over sequence length (excluding padding tokens)
                attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
                sum_embeddings = torch.sum(last_hidden_states * attention_mask_expanded, 1)
                sum_mask = attention_mask_expanded.sum(1)
                text_embedding = sum_embeddings / torch.clamp(sum_mask, min=1e-9)  # [batch_size, hidden_dim]
        
        # Coarse-grained knowledge representation
        if coarse_batch is not None and coarse_batch.x.size(0) > 0:
            # Process each graph individually to avoid batching issues
            coarse_repr = torch.zeros(batch_size, self.config['hidden_dim'] // 2, device=device)
            
            # Get unique batch indices
            unique_batch_ids = torch.unique(coarse_batch.batch)
            
            for batch_idx in unique_batch_ids:
                if batch_idx >= batch_size:
                    continue  # Skip if batch index exceeds current batch size
                    
                # Extract nodes for this specific graph
                mask = (coarse_batch.batch == batch_idx)
                graph_nodes = coarse_batch.x[mask].float()
                
                # Extract edges for this specific graph
                edge_mask = mask[coarse_batch.edge_index[0]] & mask[coarse_batch.edge_index[1]]
                if edge_mask.sum() > 0:
                    # Remap edge indices to local node indices
                    node_mapping = torch.zeros(coarse_batch.x.size(0), dtype=torch.long, device=device)
                    node_mapping[mask] = torch.arange(mask.sum(), device=device)
                    local_edges = node_mapping[coarse_batch.edge_index[:, edge_mask]]
                    
                    # Create a single-graph batch tensor
                    single_batch = torch.zeros(graph_nodes.size(0), dtype=torch.long, device=device)
                    
                    # Process this single graph
                    if graph_nodes.size(0) > 0 and local_edges.size(1) > 0:
                        single_embedding = self.coarse_gat(graph_nodes, local_edges, single_batch)
                        # Mean pooling for this single graph and add interaction embedding
                        coarse_repr[batch_idx] = single_embedding.mean(dim=0) + self.interaction_embedding
                else:
                    # No edges, use interaction embedding only
                    coarse_repr[batch_idx] = self.interaction_embedding
        else:
            coarse_repr = torch.zeros(batch_size, self.config['hidden_dim'] // 2, device=device)
        
        # Fine-grained knowledge representation  
        if fine_batch is not None and fine_batch.x.size(0) > 0:
            # Process each graph individually to avoid batching issues
            fine_repr = torch.zeros(batch_size, self.config['hidden_dim'] // 4, device=device)
            
            # Get unique batch indices
            unique_batch_ids = torch.unique(fine_batch.batch)
            
            for batch_idx in unique_batch_ids:
                if batch_idx >= batch_size:
                    continue  # Skip if batch index exceeds current batch size
                    
                # Extract nodes for this specific graph
                mask = (fine_batch.batch == batch_idx)
                graph_nodes = fine_batch.x[mask].float()
                
                # Extract edges for this specific graph
                edge_mask = mask[fine_batch.edge_index[0]] & mask[fine_batch.edge_index[1]]
                if edge_mask.sum() > 0:
                    # Remap edge indices to local node indices
                    node_mapping = torch.zeros(fine_batch.x.size(0), dtype=torch.long, device=device)
                    node_mapping[mask] = torch.arange(mask.sum(), device=device)
                    local_edges = node_mapping[fine_batch.edge_index[:, edge_mask]]
                    
                    # Create a single-graph batch tensor
                    single_batch = torch.zeros(graph_nodes.size(0), dtype=torch.long, device=device)
                    
                    # Process this single graph
                    if graph_nodes.size(0) > 0 and local_edges.size(1) > 0:
                        single_embedding = self.fine_gat(graph_nodes, local_edges, single_batch)
                        # Mean pooling for this single graph
                        fine_repr[batch_idx] = self.fine_pooling.linear(single_embedding.mean(dim=0))
                else:
                    # No edges, use mean of node features
                    if graph_nodes.size(0) > 0:
                        fine_repr[batch_idx] = self.fine_pooling.linear(graph_nodes.mean(dim=0).repeat(self.config['hidden_dim'] // 4)[:self.config['hidden_dim'] // 4])
        else:
            fine_repr = torch.zeros(batch_size, self.config['hidden_dim'] // 4, device=device)
        
        # Ensure all representations have the same batch size
        if coarse_repr.size(0) != batch_size:
            if coarse_repr.size(0) > batch_size:
                coarse_repr = coarse_repr[:batch_size]
            else:
                padding = torch.zeros(batch_size - coarse_repr.size(0), coarse_repr.size(1), device=device)
                coarse_repr = torch.cat([coarse_repr, padding])
        
        if fine_repr.size(0) != batch_size:
            if fine_repr.size(0) > batch_size:
                fine_repr = fine_repr[:batch_size]
            else:
                padding = torch.zeros(batch_size - fine_repr.size(0), fine_repr.size(1), device=device)
                fine_repr = torch.cat([fine_repr, padding])
        
        # Fusion
        fused_representation = torch.cat([text_embedding, coarse_repr, fine_repr], dim=1)
        logits = self.fusion_layer(fused_representation)
        
        return logits


def train_model(model, train_loader, val_loader, config, verbose=True):
    """Train the KAPALM model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    total_steps = len(train_loader) * config['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=total_steps
    )
    
    # Mixed precision training
    scaler = GradScaler('cuda')
    
    # Training logs
    training_log = []
    
    model.train()
    best_val_f1 = 0.0
    best_model_state = None
    
    for epoch in range(config['num_epochs']):
        total_loss = 0.0
        predictions = []
        true_labels = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            coarse_batch = batch['coarse_batch'].to(device) if batch['coarse_batch'] else None
            fine_batch = batch['fine_batch'].to(device) if batch['fine_batch'] else None
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast('cuda'):
                logits = model(input_ids, attention_mask, coarse_batch, fine_batch)
                loss = F.cross_entropy(logits, labels)
            
            # Backward pass with mixed precision
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            
            # Collect predictions
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
        
        # Calculate training metrics
        train_acc = accuracy_score(true_labels, predictions)
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary', zero_division=0)
        
        # Validation
        val_acc, val_precision, val_recall, val_f1, val_auc = evaluate_model(model, val_loader, device)
        
        avg_loss = total_loss / len(train_loader)
        
        if verbose and (epoch + 1) % 10 == 0:
            logger.info(f'Epoch {epoch+1}/{config["num_epochs"]}: '
                       f'Train Loss: {avg_loss:.4f}, Train F1: {train_f1:.4f}, '
                       f'Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}')
        
        # Save training log
        training_log.append({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'val_auc': val_auc
        })
        
        # Save best model state
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, training_log, best_val_f1


def evaluate_model(model, data_loader, device):
    """Evaluate the model with binary classification metrics"""
    model.eval()
    predictions = []
    true_labels = []
    prediction_probs = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            coarse_batch = batch['coarse_batch'].to(device) if batch['coarse_batch'] else None
            fine_batch = batch['fine_batch'].to(device) if batch['fine_batch'] else None
            
            with autocast('cuda'):
                logits = model(input_ids, attention_mask, coarse_batch, fine_batch)
            
            # Get probabilities for ROC-AUC
            probs = F.softmax(logits, dim=1)
            prediction_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
            
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    model.train()
    
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='binary', zero_division=0)
    
    # Calculate ROC-AUC for binary classification
    try:
        roc_auc = roc_auc_score(true_labels, prediction_probs)
    except:
        roc_auc = 0.0
    
    return accuracy, precision, recall, f1, roc_auc


def create_few_shot_split(df_filtered, n_shot, seed=42):
    """
    Create few-shot splits following the paper methodology:
    - Train: k samples per class (total 2k samples)
    - Val: k samples per class (total 2k samples)
    - Test: remaining samples
    
    Args:
        df_filtered: DataFrame with filtered data
        n_shot: Number of samples per class for train and val
        seed: Random seed for reproducibility
    
    Returns:
        train_df, val_df, test_df
    """
    set_seed(seed)
    
    # Separate by class (using 'label' column for Indo-Malay dataset)
    real_samples = df_filtered[df_filtered['label'] == 0]
    fake_samples = df_filtered[df_filtered['label'] == 1]
    
    # Ensure we have enough samples
    min_class_size = min(len(real_samples), len(fake_samples))
    if n_shot * 2 > min_class_size:
        logger.warning(f"Not enough samples for {n_shot}-shot. Using {min_class_size // 2} per class instead.")
        n_shot = min_class_size // 2
    
    # Sample n_shot from each class for train
    train_real = real_samples.sample(n=n_shot, random_state=seed)
    train_fake = fake_samples.sample(n=n_shot, random_state=seed)
    
    # Get remaining samples after train
    remaining_real = real_samples.drop(train_real.index)
    remaining_fake = fake_samples.drop(train_fake.index)
    
    # Sample n_shot from remaining for val
    val_real = remaining_real.sample(n=min(n_shot, len(remaining_real)), random_state=seed + 1)
    val_fake = remaining_fake.sample(n=min(n_shot, len(remaining_fake)), random_state=seed + 1)
    
    # Rest goes to test
    test_real = remaining_real.drop(val_real.index)
    test_fake = remaining_fake.drop(val_fake.index)
    
    # Combine and shuffle
    train_df = pd.concat([train_real, train_fake]).sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = pd.concat([val_real, val_fake]).sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = pd.concat([test_real, test_fake]).sample(frac=1, random_state=seed).reset_index(drop=True)
    
    return train_df, val_df, test_df


def compute_trimmed_mean(scores):
    """
    Compute trimmed mean: remove max and min, average the rest
    Following the paper: "average value calculated after deleting the maximum and minimum scores"
    """
    if len(scores) <= 2:
        return np.mean(scores)
    
    scores_sorted = sorted(scores)
    # Remove the minimum (first) and maximum (last)
    trimmed_scores = scores_sorted[1:-1]
    return np.mean(trimmed_scores)


def compute_trimmed_std(scores):
    """
    Compute standard deviation of trimmed scores (after removing max and min)
    """
    if len(scores) <= 2:
        return np.std(scores)
    
    scores_sorted = sorted(scores)
    trimmed_scores = scores_sorted[1:-1]
    return np.std(trimmed_scores)


def run_single_experiment(n_shot, seed, df_filtered, graphs_dir, tokenizer, graph_processor, config):
    """
    Run a single few-shot experiment with given shot size and seed
    
    Returns:
        Dictionary with all metrics
    """
    set_seed(seed)
    
    # Create few-shot split
    train_df, val_df, test_df = create_few_shot_split(df_filtered, n_shot, seed)
    
    # Create datasets
    train_dataset = FakeNewsDataset(train_df, tokenizer, graph_processor, 
                                   str(graphs_dir), config['max_seq_length'])
    val_dataset = FakeNewsDataset(val_df, tokenizer, graph_processor, 
                                 str(graphs_dir), config['max_seq_length'])
    test_dataset = FakeNewsDataset(test_df, tokenizer, graph_processor, 
                                  str(graphs_dir), config['max_seq_length'])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=min(config['batch_size'], len(train_dataset)), 
                             shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=min(config['batch_size'], len(val_dataset)), 
                           shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                            shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # Initialize model (fresh for each experiment)
    model = KAPALMModel(config)
    
    # Train model
    model, training_log, best_val_f1 = train_model(model, train_loader, val_loader, config, verbose=False)
    
    # Evaluate on test set
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    test_acc, test_precision, test_recall, test_f1, test_auc = evaluate_model(model, test_loader, device)
    
    # Get macro metrics
    model.eval()
    test_predictions = []
    test_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            coarse_batch = batch['coarse_batch'].to(device) if batch['coarse_batch'] else None
            fine_batch = batch['fine_batch'].to(device) if batch['fine_batch'] else None
            
            with autocast('cuda'):
                logits = model(input_ids, attention_mask, coarse_batch, fine_batch)
            
            preds = torch.argmax(logits, dim=1)
            test_predictions.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    # Calculate macro and weighted averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        test_labels, test_predictions, average='macro', zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        test_labels, test_predictions, average='weighted', zero_division=0)
    
    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()
    
    return {
        'n_shot': n_shot,
        'seed': seed,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'accuracy': test_acc,
        'precision_binary': test_precision,
        'recall_binary': test_recall,
        'f1_binary': test_f1,
        'roc_auc': test_auc,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'best_val_f1': best_val_f1
    }


def main():
    """
    Main training and evaluation function with few-shot learning
    Following the paper methodology:
    - k ∈ {2, 4, 8, 16, 100} samples per class
    - 10 random seeds
    - Remove max and min scores, average the remaining 8 (trimmed mean)
    """
    logger.info("=" * 100)
    logger.info("KAPALM Few-Shot Learning for Indo-Malay Fake News Detection")
    logger.info("=" * 100)
    logger.info(f"Model: {CONFIG['model_name']} (IndoBERT)")
    logger.info(f"Few-shot sizes: {FEWSHOT_CONFIG['shot_sizes']}")
    logger.info(f"Number of seeds: {FEWSHOT_CONFIG['num_seeds']}")
    logger.info(f"Random seeds: {FEWSHOT_CONFIG['random_seeds']}")
    logger.info(f"Methodology: Trimmed mean (remove max and min, average remaining 8)")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info("=" * 100)
    
    # Load data
    logger.info("\nLoading Indo-Malay dataset...")
    data_path = "../../dataset/dataset_sunda_malay__filtered_46k_no_dup.csv"
    
    # Check if path exists, if not try alternative
    if not os.path.exists(data_path):
        data_path = "dataset_sunda_malay__filtered_46k_no_dup.csv"
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} articles from dataset")
    
    # Filter data to only include samples with available graphs
    # Go up 2 levels from few_shot/indo_malay/ to reach processed_graphs_indo_malay/
    graphs_dir = Path("../../processed_graphs_indo_malay/successful")
    
    # Check if path exists
    if not graphs_dir.exists():
        # Try alternative paths
        alternative_paths = [
            Path("processed_graphs_indo_malay/successful"),
            Path("../processed_graphs_indo_malay/successful"),
            Path("../../processed_graphs_indo_malay_part_1/successful"),
        ]
        for alt_path in alternative_paths:
            if alt_path.exists():
                graphs_dir = alt_path
                break
    
    logger.info(f"Looking for graphs in: {graphs_dir}")
    
    available_graphs = set()
    for graph_file in graphs_dir.glob("graph_*.gpickle"):
        graph_id = graph_file.stem.replace("graph_", "")
        available_graphs.add(graph_id)
    
    logger.info(f"Found {len(available_graphs)} graphs")
    
    # Filter dataframe - convert IDs to string for matching
    df['id'] = df['id'].astype(str)
    df_filtered = df[df['id'].isin(available_graphs)].copy()
    logger.info(f"Dataset size: {len(df)} -> {len(df_filtered)} (with graphs)")
    
    # Check class distribution
    class_counts = df_filtered['label'].value_counts()
    logger.info(f"Class distribution: Real={class_counts.get(0, 0)}, Fake={class_counts.get(1, 0)}")
    
    # Initialize components (shared across all experiments)
    logger.info(f"\nLoading IndoBERT tokenizer: {CONFIG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    graph_processor = GraphProcessor(CONFIG['cache_dir'])  # Original KAPALM: no centrality pruning
    
    # Results storage
    all_individual_results = []  # Store all individual experiment results
    aggregated_results = []  # Store aggregated results per shot size
    
    # Run experiments for each few-shot size
    for n_shot in FEWSHOT_CONFIG['shot_sizes']:
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting {n_shot}-shot experiments (10 seeds)")
        logger.info(f"{'='*80}")
        
        shot_results = []
        
        # Run with each of the 10 seeds
        for seed_idx, seed in enumerate(FEWSHOT_CONFIG['random_seeds']):
            logger.info(f"\n  [{seed_idx+1}/10] Running {n_shot}-shot with seed={seed}...")
            
            try:
                result = run_single_experiment(
                    n_shot=n_shot,
                    seed=seed,
                    df_filtered=df_filtered,
                    graphs_dir=graphs_dir,
                    tokenizer=tokenizer,
                    graph_processor=graph_processor,
                    config=CONFIG
                )
                
                shot_results.append(result)
                all_individual_results.append(result)
                
                logger.info(f"    Seed {seed}: Acc={result['accuracy']:.4f}, "
                           f"F1={result['f1_binary']:.4f}, AUC={result['roc_auc']:.4f}")
                
            except Exception as e:
                logger.error(f"    Error with seed {seed}: {str(e)}")
                continue
        
        if len(shot_results) < 3:
            logger.warning(f"Not enough successful runs for {n_shot}-shot. Skipping aggregation.")
            continue
        
        # Compute trimmed mean for all metrics
        metrics_to_aggregate = ['accuracy', 'precision_binary', 'recall_binary', 'f1_binary', 
                               'roc_auc', 'precision_macro', 'recall_macro', 'f1_macro',
                               'precision_weighted', 'recall_weighted', 'f1_weighted']
        
        aggregated = {'n_shot': n_shot, 'num_runs': len(shot_results)}
        
        for metric in metrics_to_aggregate:
            scores = [r[metric] for r in shot_results]
            aggregated[f'{metric}_mean'] = compute_trimmed_mean(scores)
            aggregated[f'{metric}_std'] = compute_trimmed_std(scores)
            aggregated[f'{metric}_all'] = scores
        
        aggregated_results.append(aggregated)
        
        # Log aggregated results
        logger.info(f"\n{'-'*60}")
        logger.info(f"AGGREGATED RESULTS FOR {n_shot}-SHOT (Trimmed Mean of {len(shot_results)} runs)")
        logger.info(f"{'-'*60}")
        logger.info(f"  Accuracy:  {aggregated['accuracy_mean']:.4f} ± {aggregated['accuracy_std']:.4f}")
        logger.info(f"  Precision: {aggregated['precision_binary_mean']:.4f} ± {aggregated['precision_binary_std']:.4f}")
        logger.info(f"  Recall:    {aggregated['recall_binary_mean']:.4f} ± {aggregated['recall_binary_std']:.4f}")
        logger.info(f"  F1 Score:  {aggregated['f1_binary_mean']:.4f} ± {aggregated['f1_binary_std']:.4f}")
        logger.info(f"  ROC-AUC:   {aggregated['roc_auc_mean']:.4f} ± {aggregated['roc_auc_std']:.4f}")
        logger.info(f"  F1 Macro:  {aggregated['f1_macro_mean']:.4f} ± {aggregated['f1_macro_std']:.4f}")
        logger.info(f"  F1 Weighted: {aggregated['f1_weighted_mean']:.4f} ± {aggregated['f1_weighted_std']:.4f}")
    
    # Save all individual results
    individual_df = pd.DataFrame(all_individual_results)
    individual_df.to_csv('few_shot_individual_results_indomalay.csv', index=False)
    logger.info(f"\nSaved all individual results to 'few_shot_individual_results_indomalay.csv'")
    
    # Create summary dataframe
    summary_data = []
    for agg in aggregated_results:
        summary_data.append({
            'n_shot': agg['n_shot'],
            'num_runs': agg['num_runs'],
            'accuracy': f"{agg['accuracy_mean']:.4f} ± {agg['accuracy_std']:.4f}",
            'precision': f"{agg['precision_binary_mean']:.4f} ± {agg['precision_binary_std']:.4f}",
            'recall': f"{agg['recall_binary_mean']:.4f} ± {agg['recall_binary_std']:.4f}",
            'f1_binary': f"{agg['f1_binary_mean']:.4f} ± {agg['f1_binary_std']:.4f}",
            'roc_auc': f"{agg['roc_auc_mean']:.4f} ± {agg['roc_auc_std']:.4f}",
            'f1_macro': f"{agg['f1_macro_mean']:.4f} ± {agg['f1_macro_std']:.4f}",
            'f1_weighted': f"{agg['f1_weighted_mean']:.4f} ± {agg['f1_weighted_std']:.4f}",
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('few_shot_summary_indomalay.csv', index=False)
    
    # Save detailed aggregated results as JSON
    json_results = {
        'experiment_info': {
            'model': CONFIG['model_name'],
            'dataset': 'Indo-Malay',
            'shot_sizes': FEWSHOT_CONFIG['shot_sizes'],
            'num_seeds': FEWSHOT_CONFIG['num_seeds'],
            'random_seeds': FEWSHOT_CONFIG['random_seeds'],
            'methodology': 'Trimmed mean (remove max and min, average remaining 8)',
            'timestamp': datetime.now().isoformat()
        },
        'aggregated_results': []
    }
    
    for agg in aggregated_results:
        result_entry = {
            'n_shot': agg['n_shot'],
            'num_runs': agg['num_runs'],
            'metrics': {}
        }
        for metric in metrics_to_aggregate:
            result_entry['metrics'][metric] = {
                'trimmed_mean': agg[f'{metric}_mean'],
                'trimmed_std': agg[f'{metric}_std'],
                'all_scores': agg[f'{metric}_all']
            }
        json_results['aggregated_results'].append(result_entry)
    
    with open('few_shot_results_indomalay.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Print final summary table
    logger.info(f"\n{'='*100}")
    logger.info("FINAL FEW-SHOT LEARNING SUMMARY (Indo-Malay Dataset)")
    logger.info(f"{'='*100}")
    logger.info(f"Model: {CONFIG['model_name']} (IndoBERT)")
    logger.info(f"Methodology: 10 seeds, trimmed mean (remove max and min)")
    logger.info(f"{'='*100}\n")
    
    print("\n" + "="*120)
    print("RESULTS TABLE")
    print("="*120)
    print(f"{'K-Shot':<10} {'Accuracy':<20} {'F1 (Binary)':<20} {'ROC-AUC':<20} {'F1 (Macro)':<20} {'F1 (Weighted)':<20}")
    print("-"*120)
    
    for agg in aggregated_results:
        print(f"{agg['n_shot']:<10} "
              f"{agg['accuracy_mean']:.4f} ± {agg['accuracy_std']:.4f}    "
              f"{agg['f1_binary_mean']:.4f} ± {agg['f1_binary_std']:.4f}    "
              f"{agg['roc_auc_mean']:.4f} ± {agg['roc_auc_std']:.4f}    "
              f"{agg['f1_macro_mean']:.4f} ± {agg['f1_macro_std']:.4f}    "
              f"{agg['f1_weighted_mean']:.4f} ± {agg['f1_weighted_std']:.4f}")
    
    print("="*120)
    
    # Save summary text file
    with open('few_shot_summary_indomalay.txt', 'w') as f:
        f.write("KAPALM Few-Shot Learning Results - Indo-Malay Dataset\n")
        f.write("="*100 + "\n\n")
        f.write("Experimental Setup:\n")
        f.write(f"  - Model: {CONFIG['model_name']} (IndoBERT)\n")
        f.write(f"  - Shot sizes: {FEWSHOT_CONFIG['shot_sizes']}\n")
        f.write(f"  - Number of random seeds: {FEWSHOT_CONFIG['num_seeds']}\n")
        f.write(f"  - Seeds: {FEWSHOT_CONFIG['random_seeds']}\n")
        f.write(f"  - Scoring: Trimmed mean (remove best and worst, average remaining 8)\n\n")
        f.write("="*120 + "\n\n")
        f.write(f"{'K-Shot':<10} {'Accuracy':<20} {'F1 (Binary)':<20} {'ROC-AUC':<20} {'F1 (Macro)':<20} {'F1 (Weighted)':<20}\n")
        f.write("-"*120 + "\n")
        
        for agg in aggregated_results:
            f.write(f"{agg['n_shot']:<10} "
                   f"{agg['accuracy_mean']:.4f} ± {agg['accuracy_std']:.4f}    "
                   f"{agg['f1_binary_mean']:.4f} ± {agg['f1_binary_std']:.4f}    "
                   f"{agg['roc_auc_mean']:.4f} ± {agg['roc_auc_std']:.4f}    "
                   f"{agg['f1_macro_mean']:.4f} ± {agg['f1_macro_std']:.4f}    "
                   f"{agg['f1_weighted_mean']:.4f} ± {agg['f1_weighted_std']:.4f}\n")
        
        f.write("="*120 + "\n")
    
    logger.info("\nAll results saved:")
    logger.info("  - few_shot_individual_results_indomalay.csv (all 50 individual runs)")
    logger.info("  - few_shot_summary_indomalay.csv (aggregated summary table)")
    logger.info("  - few_shot_results_indomalay.json (detailed results with all scores)")
    logger.info("  - few_shot_summary_indomalay.txt (human-readable summary)")
    logger.info("\nIndo-Malay few-shot learning experiments completed!")


if __name__ == "__main__":
    main()
