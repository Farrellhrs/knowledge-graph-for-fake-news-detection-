#!/usr/bin/env python3
"""
KAPALM Model Variants for Ablation Study
Modular implementation supporting different ablation configurations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

try:
    from adapters import BertAdapterModel
    ADAPTERS_AVAILABLE = True
except ImportError:
    ADAPTERS_AVAILABLE = False

try:
    from torch_geometric.nn import GATConv, global_mean_pool
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

from transformers import AutoModel

from config import AblationConfig, AblationMode


class GraphAttentionNetwork(nn.Module):
    """Graph Attention Network for processing knowledge graphs"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 heads: int = 8, dropout: float = 0.3):
        super().__init__()
        
        if TORCH_GEOMETRIC_AVAILABLE:
            self.gat1 = GATConv(input_dim, hidden_dim // heads, heads=heads, 
                               dropout=dropout, concat=True)
            self.gat2 = GATConv(hidden_dim, output_dim, heads=1, 
                               dropout=dropout, concat=False)
        else:
            self.linear1 = nn.Linear(input_dim, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, output_dim)
            
        self.dropout = nn.Dropout(dropout)
        self.output_dim = output_dim
        
    def forward(self, x, edge_index, batch):
        if TORCH_GEOMETRIC_AVAILABLE:
            x = F.relu(self.gat1(x, edge_index))
            x = self.dropout(x)
            x = self.gat2(x, edge_index)
            out = global_mean_pool(x, batch)
        else:
            x = F.relu(self.linear1(x))
            x = self.dropout(x)
            x = self.linear2(x)
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
    """Attentive pooling for fine-grained graph representation"""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads=4, batch_first=True)
        self.linear = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x, batch=None):
        if x.size(0) == 0:
            return torch.zeros(1, x.size(1), device=x.device)
        
        if x.size(0) > 1:
            x_unsqueezed = x.unsqueeze(0)
            attended, _ = self.attention(x_unsqueezed, x_unsqueezed, x_unsqueezed)
            pooled = attended.squeeze(0).mean(dim=0)
        else:
            pooled = x.squeeze(0)
            
        return self.linear(pooled)


class KAPALMAblation(nn.Module):
    """
    KAPALM Model with configurable components for ablation study.
    
    Full model: concat(h, a, s)
    - h: text embedding from BERT
    - a: coarse-grained knowledge (GAT output + interaction node)
    - s: fine-grained knowledge (GAT + attentive graph pooling)
    
    Ablation modes:
    - w/o GP: concat(h, a) - remove fine-grained module
    - w/o IN: concat(h, s) - remove interaction node, use mean pooling
    - Fine-only: concat(h, s) - use pruned graph for both, no coarse
    - Coarse-only: concat(h, a) - use original graph, no fine-grained
    """
    
    def __init__(self, model_config: Dict, ablation_config: AblationConfig):
        super().__init__()
        self.model_config = model_config
        self.ablation_config = ablation_config
        
        hidden_dim = model_config['hidden_dim']
        
        # ==================== Text Encoder ====================
        if ADAPTERS_AVAILABLE:
            self.bert = BertAdapterModel.from_pretrained(model_config['model_name'])
            adapter_name = "fake_news_adapter"
            self.bert.add_adapter(adapter_name, config="pfeiffer")
            self.bert.train_adapter(adapter_name)
            self.bert.set_active_adapters([adapter_name])
            self.use_adapters = True
        else:
            self.bert = AutoModel.from_pretrained(model_config['model_name'])
            for param in self.bert.parameters():
                param.requires_grad = False
            for param in self.bert.encoder.layer[-1].parameters():
                param.requires_grad = True
            self.use_adapters = False
        
        # ==================== Graph Components ====================
        # Coarse-grained GAT (used in: full, wo_gp, wo_in, coarse_only)
        if ablation_config.use_coarse:
            self.coarse_gat = GraphAttentionNetwork(
                input_dim=1,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim // 2,
                heads=model_config['gat_heads'],
                dropout=model_config['gat_dropout']
            )
            
            # Interaction node (used in: full, wo_gp, coarse_only)
            if ablation_config.use_interaction_node:
                self.interaction_embedding = nn.Parameter(torch.randn(hidden_dim // 2))
            else:
                self.interaction_embedding = None
        else:
            self.coarse_gat = None
            self.interaction_embedding = None
        
        # Fine-grained components (used in: full, wo_in, fine_only)
        if ablation_config.use_fine:
            self.fine_gat = GraphAttentionNetwork(
                input_dim=1,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim // 4,
                heads=model_config['gat_heads'] // 2,
                dropout=model_config['gat_dropout']
            )
            
            # Attentive pooling (used in: full, wo_in, fine_only)
            if ablation_config.use_attentive_pooling:
                self.fine_pooling = AttentiveGraphPooling(
                    input_dim=hidden_dim // 4,
                    hidden_dim=hidden_dim // 4
                )
            else:
                self.fine_pooling = None
        else:
            self.fine_gat = None
            self.fine_pooling = None
        
        # ==================== Fusion Layer ====================
        # Calculate fusion input dimension based on ablation config
        fusion_dim = hidden_dim  # Text embedding always present
        
        if ablation_config.use_coarse:
            fusion_dim += hidden_dim // 2  # Coarse representation
            
        if ablation_config.use_fine:
            fusion_dim += hidden_dim // 4  # Fine representation
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(model_config['classifier_dropout']),
            nn.Linear(hidden_dim, model_config['num_labels'])
        )
        
        # Store dimensions for forward pass
        self.coarse_dim = hidden_dim // 2
        self.fine_dim = hidden_dim // 4
        
    def _encode_text(self, input_ids, attention_mask):
        """Encode text using BERT"""
        if self.use_adapters:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, 
                               output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
        else:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state
        
        # Mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * mask_expanded, 1)
        sum_mask = mask_expanded.sum(1)
        text_embedding = sum_embeddings / torch.clamp(sum_mask, min=1e-9)
        
        return text_embedding
    
    def _process_coarse_graph(self, coarse_batch, batch_size, device):
        """Process coarse-grained graph representation"""
        coarse_repr = torch.zeros(batch_size, self.coarse_dim, device=device)
        
        if coarse_batch is None or coarse_batch.x.size(0) == 0:
            return coarse_repr
        
        unique_batch_ids = torch.unique(coarse_batch.batch)
        
        for batch_idx in unique_batch_ids:
            if batch_idx >= batch_size:
                continue
                
            mask = (coarse_batch.batch == batch_idx)
            graph_nodes = coarse_batch.x[mask].float()
            
            edge_mask = mask[coarse_batch.edge_index[0]] & mask[coarse_batch.edge_index[1]]
            
            if edge_mask.sum() > 0 and graph_nodes.size(0) > 0:
                node_mapping = torch.zeros(coarse_batch.x.size(0), dtype=torch.long, device=device)
                node_mapping[mask] = torch.arange(mask.sum(), device=device)
                local_edges = node_mapping[coarse_batch.edge_index[:, edge_mask]]
                single_batch = torch.zeros(graph_nodes.size(0), dtype=torch.long, device=device)
                
                if local_edges.size(1) > 0:
                    embedding = self.coarse_gat(graph_nodes, local_edges, single_batch)
                    gat_output = embedding.mean(dim=0)
                    
                    # Add interaction node if enabled
                    if self.interaction_embedding is not None:
                        coarse_repr[batch_idx] = gat_output + self.interaction_embedding
                    else:
                        # Mean pooling only (for w/o IN ablation)
                        coarse_repr[batch_idx] = gat_output
            else:
                if self.interaction_embedding is not None:
                    coarse_repr[batch_idx] = self.interaction_embedding
                    
        return coarse_repr
    
    def _process_fine_graph(self, fine_batch, batch_size, device):
        """Process fine-grained graph representation"""
        fine_repr = torch.zeros(batch_size, self.fine_dim, device=device)
        
        if fine_batch is None or fine_batch.x.size(0) == 0:
            return fine_repr
        
        unique_batch_ids = torch.unique(fine_batch.batch)
        
        for batch_idx in unique_batch_ids:
            if batch_idx >= batch_size:
                continue
                
            mask = (fine_batch.batch == batch_idx)
            graph_nodes = fine_batch.x[mask].float()
            
            edge_mask = mask[fine_batch.edge_index[0]] & mask[fine_batch.edge_index[1]]
            
            if edge_mask.sum() > 0 and graph_nodes.size(0) > 0:
                node_mapping = torch.zeros(fine_batch.x.size(0), dtype=torch.long, device=device)
                node_mapping[mask] = torch.arange(mask.sum(), device=device)
                local_edges = node_mapping[fine_batch.edge_index[:, edge_mask]]
                single_batch = torch.zeros(graph_nodes.size(0), dtype=torch.long, device=device)
                
                if local_edges.size(1) > 0:
                    embedding = self.fine_gat(graph_nodes, local_edges, single_batch)
                    gat_output = embedding.mean(dim=0)
                    
                    # Apply attentive pooling if enabled
                    if self.fine_pooling is not None:
                        fine_repr[batch_idx] = self.fine_pooling.linear(gat_output)
                    else:
                        fine_repr[batch_idx] = gat_output
                        
        return fine_repr
    
    def forward(self, input_ids, attention_mask, coarse_batch, fine_batch):
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Text encoding (always present)
        text_embedding = self._encode_text(input_ids, attention_mask)
        
        # Build representation list for concatenation
        representations = [text_embedding]
        
        # Coarse-grained representation (if enabled)
        if self.ablation_config.use_coarse and self.coarse_gat is not None:
            coarse_repr = self._process_coarse_graph(coarse_batch, batch_size, device)
            representations.append(coarse_repr)
        
        # Fine-grained representation (if enabled)
        if self.ablation_config.use_fine and self.fine_gat is not None:
            fine_repr = self._process_fine_graph(fine_batch, batch_size, device)
            representations.append(fine_repr)
        
        # Fusion
        fused = torch.cat(representations, dim=1)
        logits = self.fusion_layer(fused)
        
        return logits


def create_model(model_config: Dict, ablation_config: AblationConfig) -> KAPALMAblation:
    """Factory function to create model with specific ablation configuration"""
    return KAPALMAblation(model_config, ablation_config)


def get_model_summary(model: KAPALMAblation) -> str:
    """Get a summary of the model configuration"""
    config = model.ablation_config
    
    summary = f"""
Model Configuration: {config.mode.value}
{'='*50}
Description: {config.description}
{'='*50}
Components:
    - Text Encoder: BERT (Adapter Tuning: {model.use_adapters})
  - Coarse GAT: {'✓' if config.use_coarse else '✗'}
  - Interaction Node: {'✓' if config.use_interaction_node else '✗'}
  - Fine GAT: {'✓' if config.use_fine else '✗'}
  - Attentive Pooling: {'✓' if config.use_attentive_pooling else '✗'}
  - Pruned Graph: {'✓' if config.use_pruned_graph else '✗'}

Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}
Total Parameters: {sum(p.numel() for p in model.parameters()):,}
"""
    return summary
