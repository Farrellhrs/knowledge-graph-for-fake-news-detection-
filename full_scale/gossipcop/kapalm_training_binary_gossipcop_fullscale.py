#!/usr/bin/env python3
"""
Custom KAPALM (Knowledge grAPh# Configuration for Binary Classification - GossipCop Dataset
CONFIG = {
    'model_name': 'bert-base-uncased',
    'max_seq_length': 512,
    'batch_size': 32,
    'learning_rate': 1e-5,
    'num_epochs': 20,  # Reduced epochs for binary classification
    'warmup_steps': 500,
    'max_grad_norm': 1.0,
    'centrality_top_n': 100,
    'hidden_dim': 768,
    'gat_heads': 8,
    'gat_dropout': 0.3,
    'classifier_dropout': 0.3,  # Reduced dropout for binary classification
    'num_labels': 2,  # Binary classification: 0 (True) vs 1 (False)
    'seed': 42,
    'cache_dir': './cache_binary_gossipcop',
    'save_path': './kapalm_binary_gossipcop.pt',
    'log_file': './training_log_binary_gossipcop.csv'
}odel) for Fake News Detection
Implements modified KAPALM with centrality-based pruning and dual knowledge representations
"""

import os
import pickle
import random
import logging
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path

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
    BertTokenizer, BertModel, BertConfig
)
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW

# Try to import adapter-transformers, fallback to regular transformers
try:
    from adapters import BertAdapterModel
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

# Configuration for Binary Classification - GossipCop Dataset
CONFIG = {
    'model_name': 'bert-base-uncased',  # BERT for English text
    'max_seq_length': 512,
    'batch_size': 32,
    'learning_rate': 1e-5,  
    'num_epochs': 100,  # Same as Indo-Malay for fair comparison
    'warmup_steps': 500,
    'max_grad_norm': 1.0,
    'centrality_top_n': 20,  # Same as Indo-Malay
    'hidden_dim': 768,
    'gat_heads': 8,
    'gat_dropout': 0.3,
    'classifier_dropout': 0.3,
    'num_labels': 2,  # Binary classification: 0 (Real) vs 1 (Fake)
    'seed': 42,
    'cache_dir': './cache_binary_gossipcop',
    'save_path': './kapalm_binary_gossipcop.pt',
    'log_file': './training_log_binary_gossipcop.csv'
}

# Binary Label mapping for GossipCop Dataset
LABEL_MAP = {
    0: 'real',
    1: 'fake'
}

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CONFIG['seed'])


class GraphProcessor:
    """Handles graph loading, pruning, and conversion to PyTorch Geometric format"""
    
    def __init__(self, cache_dir: str, centrality_top_n: int = 100):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
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
        """Prune graph based on degree centrality"""
        if G.number_of_nodes() <= self.centrality_top_n:
            return G.copy()
        
        # Compute degree centrality
        centrality = nx.degree_centrality(G)
        
        # Get top-N nodes by centrality
        top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:self.centrality_top_n]
        top_node_ids = [node for node, _ in top_nodes]
        
        # Create subgraph with top nodes
        pruned_G = G.subgraph(top_node_ids).copy()
        
        logger.debug(f"Pruned graph from {G.number_of_nodes()} to {pruned_G.number_of_nodes()} nodes")
        return pruned_G
    
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
        text = str(row['FullText'])
        label = int(row['hoax'])
        
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
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Text encoder (BERT with adapters if available)
        if ADAPTERS_AVAILABLE:
            self.bert = BertAdapterModel.from_pretrained(config['model_name'])
            # Add adapter for classification (not masked LM)
            adapter_name = "fake_news_adapter"
            self.bert.add_adapter(adapter_name, config="pfeiffer")
            self.bert.train_adapter(adapter_name)
            # Set the active adapter - pass as list
            self.bert.set_active_adapters([adapter_name])
            print("Using BERT with adapter tuning")
        else:
            self.bert = BertModel.from_pretrained(config['model_name'])
            # Freeze BERT parameters for efficiency (can be unfrozen if needed)
            for param in self.bert.parameters():
                param.requires_grad = False
            # Unfreeze the last layer
            for param in self.bert.encoder.layer[-1].parameters():
                param.requires_grad = True
            print("Using BERT with last layer fine-tuning")
        
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


def train_model(model, train_loader, val_loader, config):
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
            
            if (batch_idx + 1) % 50 == 0:
                logger.info(f'Epoch {epoch+1}/{config["num_epochs"]}, '
                           f'Batch {batch_idx+1}/{len(train_loader)}, '
                           f'Loss: {loss.item():.4f}')
        
        # Calculate training metrics - use binary for binary classification
        train_acc = accuracy_score(true_labels, predictions)
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary', zero_division=0)
        
        # Validation
        val_acc, val_precision, val_recall, val_f1, val_auc = evaluate_model(model, val_loader, device)
        
        avg_loss = total_loss / len(train_loader)
        
        logger.info(f'Epoch {epoch+1}/{config["num_epochs"]}:')
        logger.info(f'  Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}')
        logger.info(f'  Val Acc: {val_acc:.4f}, Val Precision: {val_precision:.4f}, '
                   f'Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}')
        
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
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'epoch': epoch,
                'val_f1': val_f1
            }, config['save_path'])
            logger.info(f'New best model saved with Val F1: {val_f1:.4f}')
    
    # Save training log
    log_df = pd.DataFrame(training_log)
    log_df.to_csv(config['log_file'], index=False)
    
    return model, training_log


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


def main():
    """Main training and evaluation function"""
    logger.info("Starting KAPALM training for binary fake news detection - GossipCop Dataset")
    logger.info("Binary Classification: 0=Real vs 1=Fake")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    # Load data
    logger.info("Loading GossipCop dataset...")
    data_path = "../../merged_dataset.csv"
    df = pd.read_csv(data_path)
    
    # Filter data to only include samples with available graphs
    # Go up 2 levels from full_scale/gossipcop/ to reach processed_graphs_gossipcop/
    graphs_dir = Path("../../processed_graphs_gossipcop/successful")
    available_graphs = set()
    for graph_file in graphs_dir.glob("graph_*.gpickle"):
        # Extract the full ID including prefix (e.g., "gossipcop-882573" from "graph_gossipcop-882573.gpickle")
        graph_id = graph_file.stem.replace("graph_", "")
        available_graphs.add(graph_id)
    
    # Filter dataframe - use string matching for GossipCop IDs
    df_filtered = df[df['id'].isin(available_graphs)].copy()
    logger.info(f"Dataset size: {len(df)} -> {len(df_filtered)} (with graphs)")
    
    # Data split
    train_df, temp_df = train_test_split(df_filtered, test_size=0.2, random_state=CONFIG['seed'], 
                                        stratify=df_filtered['hoax'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=CONFIG['seed'],
                                      stratify=temp_df['hoax'])
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Initialize components
    tokenizer = BertTokenizer.from_pretrained(CONFIG['model_name'])
    graph_processor = GraphProcessor(CONFIG['cache_dir'], CONFIG['centrality_top_n'])
    
    # Create datasets
    train_dataset = FakeNewsDataset(train_df, tokenizer, graph_processor, 
                                   str(graphs_dir), CONFIG['max_seq_length'])
    val_dataset = FakeNewsDataset(val_df, tokenizer, graph_processor, 
                                 str(graphs_dir), CONFIG['max_seq_length'])
    test_dataset = FakeNewsDataset(test_df, tokenizer, graph_processor, 
                                  str(graphs_dir), CONFIG['max_seq_length'])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                             shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                           shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], 
                            shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # Initialize model
    logger.info("Initializing KAPALM model...")
    model = KAPALMModel(CONFIG)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    logger.info("Starting training...")
    model, training_log = train_model(model, train_loader, val_loader, CONFIG)
    
    # Final evaluation on test set
    logger.info("Evaluating on test set...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load best model
    checkpoint = torch.load(CONFIG['save_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Test evaluation using the same evaluation function
    test_acc, test_precision, test_recall, test_f1, test_auc = evaluate_model(model, test_loader, device)
    
    # Get predictions for classification report
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
    
    logger.info("=== FINAL BINARY CLASSIFICATION RESULTS ===")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Test Precision: {test_precision:.4f}")
    logger.info(f"Test Recall: {test_recall:.4f}")
    logger.info(f"Test F1 Score: {test_f1:.4f}")
    logger.info(f"Test ROC-AUC: {test_auc:.4f}")
    
    # Detailed classification report
    report = classification_report(test_labels, test_predictions, 
                                 target_names=[LABEL_MAP[i] for i in range(CONFIG['num_labels'])],
                                 digits=4)
    logger.info("Detailed Classification Report:")
    logger.info("\n" + str(report))
    
    # Save final results with binary classification specific outputs
    results = {
        'test_accuracy': test_acc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_roc_auc': test_auc,
        'classification_report': report,
        'config': CONFIG
    }
    
    with open('final_results_binary_gossipcop.json', 'w') as f:
        json.dump({k: v for k, v in results.items() if k != 'classification_report'}, f, indent=2)
    
    with open('classification_report_binary_gossipcop.txt', 'w') as f:
        f.write(str(report))
    
    logger.info("GossipCop binary classification training completed! Results saved.")


if __name__ == "__main__":
    main()
