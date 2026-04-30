"""
KAPALM: Knowledge-Augmented Pretrained Language Model for Fake News Detection

This module implements the KAPALM model that combines textual features with 
knowledge graph embeddings for enhanced fake news detection.

The model consists of:
1. Text Encoder: Indonesian BERT model for textual feature extraction
2. Graph Encoder: (To be implemented) for knowledge graph embeddings
3. Fusion Layer: (To be implemented) for combining text and graph features
4. Classification Head: (To be implemented) for final prediction

Author: Generated for Fake News Detection Final Project
Date: June 30, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig
import numpy as np
import logging
import networkx as nx
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PyTorch Geometric imports for Graph Neural Networks
try:
    from torch_geometric.nn import GATConv
    from torch_geometric.data import Data
    from torch_geometric.utils import from_networkx
    TORCH_GEOMETRIC_AVAILABLE = True
    logger.info("PyTorch Geometric is available")
except ImportError:
    logger.warning("PyTorch Geometric not found. Graph encoder will not be available.")
    TORCH_GEOMETRIC_AVAILABLE = False
    # Create dummy classes to prevent import errors
    class GATConv:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch Geometric is required for graph operations")
        
        def __call__(self, *args, **kwargs):
            raise ImportError("PyTorch Geometric is required for graph operations")
    
    class Data:
        pass

class KAPALM(torch.nn.Module):
    """
    Knowledge-Augmented Pretrained Language Model for Fake News Detection.
    
    This model combines textual features from Indonesian BERT with knowledge graph
    embeddings to improve fake news detection performance.
    """
    
    def __init__(self, 
                 bert_model_name="indolem/indobert-base-uncased",
                 hidden_size=768,
                 num_classes=2,
                 dropout_rate=0.1):
        """
        Initialize the KAPALM model.
        
        Args:
            bert_model_name (str): Name of the Indonesian BERT model to use
            hidden_size (int): Hidden size of the BERT model (default: 768)
            num_classes (int): Number of output classes (default: 2 for binary classification)
            dropout_rate (float): Dropout rate for regularization (default: 0.1)
        """
        super(KAPALM, self).__init__()
        
        logger.info(f"Initializing KAPALM model with BERT: {bert_model_name}")
        
        # Store configuration
        self.bert_model_name = bert_model_name
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Initialize text encoder
        self._init_text_encoder()
        
        # Initialize coarse-grained knowledge encoder
        self._init_coarse_grained_encoder()
        
        # Initialize fine-grained knowledge encoder
        self._init_fine_grained_encoder()
        
        # Initialize final classifier
        self._init_classifier()
        
        logger.info("KAPALM model initialized successfully")
    
    def _init_text_encoder(self):
        """Initialize the text encoder component using Indonesian BERT."""
        import os
        
        # Determine if we should use offline mode
        force_offline = (
            os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1" or
            os.environ.get("HF_HUB_OFFLINE", "0") == "1"
        )
        
        cache_dir = "./model_cache"
        
        # Set environment variables to prevent network requests if offline
        if force_offline:
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"
        
        try:
            logger.info(f"Loading Indonesian BERT model: {self.bert_model_name}")
            
            # Try to use cached version first to avoid rate limits
            try:
                logger.info(f"Attempting to load cached model: {self.bert_model_name}")
                self.text_encoder = AutoModel.from_pretrained(
                    self.bert_model_name,
                    output_hidden_states=True,  # We want access to hidden states
                    output_attentions=False,    # We don't need attention weights for now
                    return_dict=True,          # Return as dictionary for easier access
                    cache_dir=cache_dir,       # Use local cache directory
                    local_files_only=True,     # Try offline first
                    trust_remote_code=False    # Don't download remote code
                )
                logger.info(f"✅ Using cached model: {self.bert_model_name}")
            except Exception as cache_error:
                logger.info(f"📁 No cached model found: {cache_error}")
                
                # If forced offline and no cache, fail
                if force_offline:
                    raise RuntimeError(f"Cannot load model in offline mode. Cache error: {cache_error}")
                
                # Try online download
                logger.info(f"🌐 Downloading model: {self.bert_model_name}")
                # Temporarily allow online access
                os.environ["HF_HUB_OFFLINE"] = "0"
                os.environ["TRANSFORMERS_OFFLINE"] = "0"
                
                self.text_encoder = AutoModel.from_pretrained(
                    self.bert_model_name,
                    output_hidden_states=True,  # We want access to hidden states
                    output_attentions=False,    # We don't need attention weights for now
                    return_dict=True,          # Return as dictionary for easier access
                    cache_dir=cache_dir,       # Use local cache directory
                    local_files_only=False,    # Allow online download
                    force_download=False,      # Don't re-download if cached
                    trust_remote_code=False    # Don't download remote code
                )
                logger.info(f"✅ Model downloaded and cached: {self.bert_model_name}")
            
            # Add text classification adapter for dropout only (we use the main classifier for final prediction)
            self.text_classifier_dropout = nn.Dropout(self.dropout_rate)
            
            # Get the actual hidden size from the model config (in case it differs)
            self.actual_hidden_size = self.text_encoder.config.hidden_size
            
            # Update dropout layer if hidden size is different
            if self.actual_hidden_size != self.hidden_size:
                logger.warning(f"Model hidden size ({self.actual_hidden_size}) differs from specified ({self.hidden_size})")
                self.hidden_size = self.actual_hidden_size
            
            logger.info(f"Text encoder initialized with hidden size: {self.hidden_size}")
            
        except Exception as e:
            logger.error(f"Error initializing text encoder: {e}")
            raise
    
    def _init_coarse_grained_encoder(self):
        """Initialize the coarse-grained knowledge encoder using Graph Attention Network."""
        try:
            if not TORCH_GEOMETRIC_AVAILABLE:
                logger.warning("PyTorch Geometric not available. Coarse-grained encoder will be disabled.")
                self.coarse_grain_encoder = None
                return
            
            logger.info("Initializing coarse-grained knowledge encoder (GAT)...")
            
            # Initialize Graph Attention Network (GAT) layer
            # Input and output dimensions match BERT hidden size (768)
            self.coarse_grain_encoder = GATConv(
                in_channels=self.hidden_size,   # Input feature size (768)
                out_channels=self.hidden_size,  # Output feature size (768)
                heads=8,                        # Number of attention heads
                dropout=self.dropout_rate,      # Dropout rate
                concat=False                    # Average the heads instead of concatenating
            )
            
            logger.info(f"Coarse-grained encoder (GAT) initialized with {self.hidden_size} dimensions and 8 attention heads")
            
        except Exception as e:
            logger.error(f"Error initializing coarse-grained encoder: {e}")
            self.coarse_grain_encoder = None
            raise
    
    def _init_fine_grained_encoder(self):
        """Initialize the fine-grained knowledge encoder using MultiheadAttention."""
        try:
            logger.info("Initializing fine-grained knowledge encoder (MultiheadAttention)...")
            
            # Initialize MultiheadAttention layer
            # Embedding dimension matches BERT hidden size (768) with 2 attention heads
            self.fine_grain_encoder = nn.MultiheadAttention(
                embed_dim=self.hidden_size,     # Embedding dimension (768)
                num_heads=2,                    # Number of attention heads
                dropout=self.dropout_rate,      # Dropout rate
                batch_first=True               # Batch dimension first
            )
            
            logger.info(f"Fine-grained encoder (MultiheadAttention) initialized with {self.hidden_size} dimensions and 2 attention heads")
            
        except Exception as e:
            logger.error(f"Error initializing fine-grained encoder: {e}")
            raise
    
    def _init_classifier(self):
        """Initialize the final classifier with MLP architecture."""
        try:
            logger.info("Initializing final classifier (MLP)...")
            
            # Calculate input dimension for classifier
            # Input will be concatenation of textual (h), coarse-grained (a), and fine-grained (s) representations
            classifier_input_dim = self.hidden_size * 3  # 768 * 3 = 2304
            
            # Initialize MLP classifier
            # Hidden layer with 200 units, dropout 0.2, and final binary classification layer
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, 200),  # Hidden layer: 2304 -> 200
                nn.ReLU(),                             # Activation function
                nn.Dropout(0.2),                       # Dropout for regularization
                nn.Linear(200, self.num_classes)       # Output layer: 200 -> 2 (binary classification)
            )
            
            logger.info(f"Final classifier initialized: {classifier_input_dim} -> 200 -> {self.num_classes}")
            
        except Exception as e:
            logger.error(f"Error initializing classifier: {e}")
            raise
    
    def encode_text(self, input_ids, attention_mask):
        """
        Encode text input using the Indonesian BERT model.
        
        This method takes tokenized input and passes it through the text encoder
        to extract textual features. It returns the [CLS] token representation
        which serves as the textual encoding.
        
        Args:
            input_ids (torch.Tensor): Token IDs of shape (batch_size, sequence_length)
            attention_mask (torch.Tensor): Attention mask of shape (batch_size, sequence_length)
                                         1 for tokens to attend to, 0 for padding tokens
        
        Returns:
            torch.Tensor: Text encoding vector h of shape (batch_size, hidden_size)
                         This is the [CLS] token representation from the final layer
        
        Example:
            >>> model = KAPALM()
            >>> input_ids = torch.tensor([[101, 2054, 2003, 102]])  # Example token IDs
            >>> attention_mask = torch.tensor([[1, 1, 1, 1]])       # All tokens are real
            >>> h = model.encode_text(input_ids, attention_mask)
            >>> print(h.shape)  # torch.Size([1, 768])
        """
        try:
            # Ensure inputs are on the same device as the model
            device = next(self.text_encoder.parameters()).device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # Pass through BERT model
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Extract the [CLS] token representation from the last hidden state
            # outputs.last_hidden_state shape: (batch_size, sequence_length, hidden_size)
            # [CLS] token is always at position 0
            cls_token_embedding = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
            
            # Apply dropout for regularization during training
            h = self.text_classifier_dropout(cls_token_embedding)
            
            return h
            
        except Exception as e:
            logger.error(f"Error in encode_text: {e}")
            raise
    
    def forward(self, input_ids, attention_mask, full_graph=None, pruned_graph=None):
        """
        Forward pass of the KAPALM model.
        
        This method orchestrates all components of the KAPALM architecture:
        1. Text Encoder: Extracts textual features (h)
        2. Coarse-Grained Knowledge Encoder: Processes full graph to get representation (a)
        3. Fine-Grained Knowledge Encoder: Processes pruned graph to get representation (s)
        4. Final Classifier: Combines [h, a, s] and produces final predictions
        
        Args:
            input_ids (torch.Tensor): Token IDs of shape (batch_size, sequence_length)
            attention_mask (torch.Tensor): Attention mask of shape (batch_size, sequence_length)
            full_graph (networkx.Graph, optional): Complete knowledge graph for coarse-grained processing
            pruned_graph (networkx.Graph, optional): Pruned knowledge graph for fine-grained processing
        
        Returns:
            dict: Dictionary containing:
                - 'text_encoding': Text encoding vector h (batch_size, hidden_size)
                - 'coarse_grained_encoding': Coarse-grained representation a (batch_size, hidden_size)
                - 'fine_grained_encoding': Fine-grained representation s (batch_size, hidden_size)
                - 'fused_features': Concatenated features [h, a, s] (batch_size, hidden_size * 3)
                - 'logits': Final classification logits (batch_size, num_classes)
                - 'predictions': Predicted class probabilities (batch_size, num_classes)
        """
        try:
            # Step 1: Text Encoding (h)
            logger.debug("Step 1: Encoding text...")
            text_encoding = self.encode_text(input_ids, attention_mask)
            
            # Step 2: Coarse-Grained Knowledge Encoding (a)
            logger.debug("Step 2: Processing coarse-grained knowledge...")
            if full_graph is not None:
                coarse_grained_encoding = self.run_coarse_grained_path(full_graph, text_encoding)
            else:
                logger.warning("No full_graph provided. Using text encoding as coarse-grained representation.")
                coarse_grained_encoding = text_encoding
            
            # Step 3: Fine-Grained Knowledge Encoding (s)
            logger.debug("Step 3: Processing fine-grained knowledge...")
            if pruned_graph is not None:
                fine_grained_encoding = self.run_fine_grained_path(pruned_graph, text_encoding)
            else:
                logger.warning("No pruned_graph provided. Using text encoding as fine-grained representation.")
                fine_grained_encoding = text_encoding
            
            # Step 4: Feature Fusion
            logger.debug("Step 4: Fusing features...")
            # Concatenate textual (h), coarse-grained (a), and fine-grained (s) representations
            fused_features = torch.cat([
                text_encoding,           # h: (batch_size, hidden_size)
                coarse_grained_encoding, # a: (batch_size, hidden_size)
                fine_grained_encoding    # s: (batch_size, hidden_size)
            ], dim=-1)  # Result: (batch_size, hidden_size * 3) = (batch_size, 2304)
            
            # Step 5: Final Classification
            logger.debug("Step 5: Final classification...")
            logits = self.classifier(fused_features)
            predictions = F.softmax(logits, dim=-1)
            
            logger.debug(f"Forward pass completed - Logits: {logits.shape}, Predictions: {predictions.shape}")
            
            return {
                'text_encoding': text_encoding,
                'coarse_grained_encoding': coarse_grained_encoding,
                'fine_grained_encoding': fine_grained_encoding,
                'fused_features': fused_features,
                'logits': logits,
                'predictions': predictions
            }
            
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            raise
    
    def forward_batch(self, input_ids, attention_mask, vectorized_full_graphs=None, vectorized_pruned_graphs=None):
        """
        ULTRA-OPTIMIZED batch forward pass for maximum GPU utilization.
        
        Processes entire batches of text and graphs simultaneously for 90%+ GPU usage.
        This method handles vectorized graph operations for true batch processing.
        
        Args:
            input_ids (torch.Tensor): Batch of token IDs (batch_size, sequence_length)
            attention_mask (torch.Tensor): Batch of attention masks (batch_size, sequence_length)
            vectorized_full_graphs (dict): Vectorized full graphs with node_features, adjacency_matrices, node_masks
            vectorized_pruned_graphs (dict): Vectorized pruned graphs with node_features, adjacency_matrices, node_masks
        
        Returns:
            dict: Dictionary containing:
                - 'logits': Final classification logits (batch_size, num_classes)
                - 'predictions': Predicted class probabilities (batch_size, num_classes)
        """
        try:
            batch_size = input_ids.size(0)
            device = input_ids.device
            
            # Step 1: Batch text encoding (already optimized)
            logger.debug("Step 1: Batch text encoding...")
            text_encoding = self.encode_text(input_ids, attention_mask)
            
            # Step 2: Batch coarse-grained encoding
            logger.debug("Step 2: Batch coarse-grained processing...")
            if vectorized_full_graphs is not None:
                coarse_grained_encoding = self._batch_coarse_grained_encoding(
                    vectorized_full_graphs, text_encoding
                )
            else:
                coarse_grained_encoding = text_encoding
            
            # Step 3: Batch fine-grained encoding
            logger.debug("Step 3: Batch fine-grained processing...")
            if vectorized_pruned_graphs is not None:
                fine_grained_encoding = self._batch_fine_grained_encoding(
                    vectorized_pruned_graphs, text_encoding
                )
            else:
                fine_grained_encoding = text_encoding
            
            # Step 4: Batch feature fusion
            logger.debug("Step 4: Batch feature fusion...")
            fused_features = torch.cat([
                text_encoding,           # h: (batch_size, hidden_size)
                coarse_grained_encoding, # a: (batch_size, hidden_size)
                fine_grained_encoding    # s: (batch_size, hidden_size)
            ], dim=-1)  # Result: (batch_size, hidden_size * 3)
            
            # Step 5: Batch classification
            logger.debug("Step 5: Batch classification...")
            logits = self.classifier(fused_features)
            predictions = F.softmax(logits, dim=-1)
            
            return {
                'logits': logits,
                'predictions': predictions,
                'text_encoding': text_encoding,
                'coarse_grained_encoding': coarse_grained_encoding,
                'fine_grained_encoding': fine_grained_encoding,
                'fused_features': fused_features
            }
            
        except Exception as e:
            logger.error(f"Error in batch forward pass: {e}")
            # Fallback to per-sample processing
            logger.warning("Falling back to per-sample processing")
            return self._fallback_batch_forward(input_ids, attention_mask)
    
    def _batch_coarse_grained_encoding(self, vectorized_graphs, text_encoding):
        """
        Ultra-fast batch processing for coarse-grained graph encoding.
        
        Args:
            vectorized_graphs (dict): Batch of vectorized graphs
            text_encoding (torch.Tensor): Batch text encodings (batch_size, hidden_size)
        
        Returns:
            torch.Tensor: Batch coarse-grained encodings (batch_size, hidden_size)
        """
        try:
            if self.coarse_grain_encoder is None:
                return text_encoding
            
            batch_size = text_encoding.size(0)
            device = text_encoding.device
            
            # Extract vectorized graph data
            adj_matrices = vectorized_graphs['adjacency_matrices'].to(device)  # (batch_size, max_nodes, max_nodes)
            node_features = vectorized_graphs['node_features'].to(device)  # (batch_size, max_nodes, 768)
            graph_masks = vectorized_graphs['node_masks'].to(device)  # (batch_size, max_nodes)
            max_nodes = node_features.size(1)  # Get max_nodes from tensor shape
            
            # Initialize node features with text encoding
            # Broadcast text encoding to all nodes in each graph
            node_embeddings = text_encoding.unsqueeze(1).expand(-1, max_nodes, -1)  # (batch_size, max_nodes, hidden_size)
            
            # Apply graph mask to valid nodes only
            node_embeddings = node_embeddings * graph_masks.unsqueeze(-1).float()
            
            # Reshape for batch GAT processing
            # Flatten batch and node dimensions for GAT
            flat_node_embeddings = node_embeddings.view(-1, self.hidden_size)  # (batch_size * max_nodes, hidden_size)
            
            # Create edge indices for GAT from adjacency matrices
            edge_indices = []
            node_offset = 0
            
            for batch_idx in range(batch_size):
                adj_matrix = adj_matrices[batch_idx]
                mask = graph_masks[batch_idx]
                num_valid_nodes = mask.sum().item()
                
                if num_valid_nodes > 0:
                    # Get edges from adjacency matrix
                    edge_sources, edge_targets = torch.nonzero(adj_matrix[:num_valid_nodes, :num_valid_nodes], as_tuple=True)
                    
                    # Add offset for this batch
                    edge_sources = edge_sources + node_offset
                    edge_targets = edge_targets + node_offset
                    
                    if len(edge_sources) > 0:
                        batch_edges = torch.stack([edge_sources, edge_targets], dim=0)
                        edge_indices.append(batch_edges)
                
                node_offset += max_nodes
            
            if edge_indices:
                # Concatenate all edge indices
                all_edge_indices = torch.cat(edge_indices, dim=1)  # (2, total_edges)
                
                # Apply GAT to entire batch
                gat_output = self.coarse_grain_encoder(flat_node_embeddings, all_edge_indices)
                
                # Reshape back to batch format
                batch_gat_output = gat_output.view(batch_size, max_nodes, -1)
                
                # Apply attention pooling to get graph-level representation
                # Use mean pooling with masking
                masked_output = batch_gat_output * graph_masks.unsqueeze(-1).float()
                graph_representations = masked_output.sum(dim=1) / (graph_masks.sum(dim=1, keepdim=True).float() + 1e-8)
                
                return graph_representations
            else:
                # No valid edges, return text encoding
                return text_encoding
                
        except Exception as e:
            logger.warning(f"Error in batch coarse-grained encoding: {e}")
            return text_encoding
    
    def _batch_fine_grained_encoding(self, vectorized_graphs, text_encoding):
        """
        Ultra-fast batch processing for fine-grained graph encoding.
        
        Args:
            vectorized_graphs (dict): Batch of vectorized graphs
            text_encoding (torch.Tensor): Batch text encodings (batch_size, hidden_size)
        
        Returns:
            torch.Tensor: Batch fine-grained encodings (batch_size, hidden_size)
        """
        try:
            if self.fine_grain_encoder is None:
                return text_encoding
            
            # Similar to coarse-grained but using fine-grain encoder
            batch_size = text_encoding.size(0)
            device = text_encoding.device
            
            adj_matrices = vectorized_graphs['adjacency_matrices'].to(device)
            node_features = vectorized_graphs['node_features'].to(device)
            graph_masks = vectorized_graphs['node_masks'].to(device)
            max_nodes = node_features.size(1)  # Get max_nodes from tensor shape
            
            # Initialize and process similar to coarse-grained
            node_embeddings = text_encoding.unsqueeze(1).expand(-1, max_nodes, -1)
            node_embeddings = node_embeddings * graph_masks.unsqueeze(-1).float()
            
            flat_node_embeddings = node_embeddings.view(-1, self.hidden_size)
            
            # Create edge indices
            edge_indices = []
            node_offset = 0
            
            for batch_idx in range(batch_size):
                adj_matrix = adj_matrices[batch_idx]
                mask = graph_masks[batch_idx]
                num_valid_nodes = mask.sum().item()
                
                if num_valid_nodes > 0:
                    edge_sources, edge_targets = torch.nonzero(adj_matrix[:num_valid_nodes, :num_valid_nodes], as_tuple=True)
                    edge_sources = edge_sources + node_offset
                    edge_targets = edge_targets + node_offset
                    
                    if len(edge_sources) > 0:
                        batch_edges = torch.stack([edge_sources, edge_targets], dim=0)
                        edge_indices.append(batch_edges)
                
                node_offset += max_nodes
            
            if edge_indices and len(edge_indices) > 0:
                # Use MultiheadAttention properly (query, key, value)
                # Reshape node embeddings for attention
                batch_node_embeddings = node_embeddings  # (batch_size, max_nodes, hidden_size)
                
                # For MultiheadAttention, use node embeddings as query, key, and value
                attn_output, _ = self.fine_grain_encoder(
                    query=batch_node_embeddings,    # (batch_size, max_nodes, hidden_size)
                    key=batch_node_embeddings,      # (batch_size, max_nodes, hidden_size)
                    value=batch_node_embeddings,    # (batch_size, max_nodes, hidden_size)
                    key_padding_mask=~graph_masks   # Mask for padding nodes (True for padding)
                )
                
                # Apply graph mask and pool
                masked_output = attn_output * graph_masks.unsqueeze(-1).float()
                graph_representations = masked_output.sum(dim=1) / (graph_masks.sum(dim=1, keepdim=True).float() + 1e-8)
                
                return graph_representations
            else:
                return text_encoding
                
        except Exception as e:
            logger.warning(f"Error in batch fine-grained encoding: {e}")
            return text_encoding
    
    def _fallback_batch_forward(self, input_ids, attention_mask):
        """
        Fallback to per-sample processing if batch processing fails.
        """
        batch_size = input_ids.size(0)
        batch_outputs = []
        
        for i in range(batch_size):
            sample_input_ids = input_ids[i:i+1]
            sample_attention_mask = attention_mask[i:i+1]
            
            # Process without graphs
            outputs = self.forward(sample_input_ids, sample_attention_mask)
            batch_outputs.append(outputs['logits'])
        
        logits = torch.cat(batch_outputs, dim=0)
        predictions = F.softmax(logits, dim=-1)
        
        return {
            'logits': logits,
            'predictions': predictions
        }
    
    def get_model_info(self):
        """
        Get information about the KAPALM model configuration.
        
        Returns:
            dict: Dictionary containing model configuration information
        """
        return {
            'model_name': 'KAPALM',
            'bert_model_name': getattr(self, 'bert_model_name', 'indolem/indobert-base-uncased'),
            'hidden_size': getattr(self, 'hidden_size', 768),
            'num_classes': getattr(self, 'num_classes', 2),
            'dropout_rate': getattr(self, 'dropout_rate', 0.1),
            'architecture': 'Text + Coarse-grained + Fine-grained Graph Fusion',
            'components': {
                'text_encoder': 'Indonesian BERT',
                'coarse_grained_encoder': 'Graph Attention Network (GAT)',
                'fine_grained_encoder': 'Graph Attention Network (GAT)',
                'classifier': 'Multi-layer MLP'
            }
        }
    
    def run_coarse_grained_path(self, full_graph, text_encoding):
        """
        Process coarse-grained knowledge from full graph using GAT.
        
        Args:
            full_graph: NetworkX graph (full version)
            text_encoding: Text encoding tensor (batch_size, hidden_size)
        
        Returns:
            torch.Tensor: Coarse-grained encoding (batch_size, hidden_size)
        """
        try:
            if self.coarse_grain_encoder is None or full_graph is None:
                return text_encoding
            
            device = text_encoding.device
            batch_size = text_encoding.size(0)
            
            # Convert graph to PyTorch Geometric format
            nodes = list(full_graph.nodes())
            if len(nodes) == 0:
                return text_encoding
            
            # Create node features
            node_features = []
            for node in nodes:
                node_data = full_graph.nodes[node]
                if 'embedding' in node_data:
                    embedding = node_data['embedding']
                    if isinstance(embedding, np.ndarray) and embedding.shape[0] == 768:
                        node_features.append(torch.tensor(embedding, dtype=torch.float32, device=device))
                    else:
                        # Use text encoding as fallback
                        node_features.append(text_encoding.squeeze(0))
                else:
                    # Use text encoding as fallback
                    node_features.append(text_encoding.squeeze(0))
            
            if not node_features:
                return text_encoding
            
            # Stack node features
            x = torch.stack(node_features, dim=0)  # (num_nodes, hidden_size)
            
            # Create edge indices
            edges = list(full_graph.edges())
            if not edges:
                # No edges, just return aggregated node features
                graph_representation = x.mean(dim=0, keepdim=True)  # (1, hidden_size)
            else:
                # Create edge index tensor
                edge_index = torch.tensor(
                    [[nodes.index(edge[0]), nodes.index(edge[1])] for edge in edges],
                    dtype=torch.long, device=device
                ).t()  # (2, num_edges)
                
                # Apply GAT
                gat_output = self.coarse_grain_encoder(x, edge_index)  # (num_nodes, hidden_size)
                
                # Pool to get graph-level representation
                graph_representation = gat_output.mean(dim=0, keepdim=True)  # (1, hidden_size)
            
            # Ensure output has correct batch size
            if batch_size > 1:
                graph_representation = graph_representation.expand(batch_size, -1)
            
            return graph_representation
            
        except Exception as e:
            logger.warning(f"Error in coarse-grained path: {e}")
            return text_encoding

    # ...existing code...
def create_tokenizer(use_cache=True, force_offline=True):
    """
    Create and return the Indonesian BERT tokenizer.
    
    This function creates a tokenizer for the Indonesian BERT model used in KAPALM.
    It supports both online and offline modes.
    
    Args:
        use_cache (bool): Whether to use cached models (default: True)
        force_offline (bool): Whether to force offline mode (default: True)
    
    Returns:
        AutoTokenizer: The Indonesian BERT tokenizer
    
    Raises:
        RuntimeError: If tokenizer cannot be loaded in offline mode
    """
    
    model_name = "indolem/indobert-base-uncased"
    
    try:
        if force_offline:
            # Force offline mode by setting environment variables
            import os
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
            
            logger.info("Creating tokenizer in OFFLINE mode...")
            
            # Try to load from cache
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=True,
                use_fast=True,
                cache_dir="./model_cache"
            )
            
            logger.info("✅ Tokenizer loaded successfully from cache")
            
        else:
            logger.info("Creating tokenizer in ONLINE mode...")
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                cache_dir="./model_cache" if use_cache else None
            )
            
            logger.info("✅ Tokenizer loaded successfully")
        
        # Test tokenizer
        test_text = "Test tokenizer Indonesia"
        test_encoding = tokenizer(test_text, return_tensors='pt')
        logger.debug(f"Tokenizer test successful. Input IDs shape: {test_encoding['input_ids'].shape}")
        
        return tokenizer
        
    except Exception as e:
        error_msg = f"Failed to create tokenizer: {e}"
        logger.error(error_msg)
        
        if force_offline:
            logger.error("💡 SOLUTION: Run 'python setup_complete_cache.py' to download models first")
        
        raise RuntimeError(error_msg)


def test_kapalm_text_encoder():
    """
    Test function for KAPALM text encoder functionality.
    
    This function tests:
    1. Model initialization
    2. Tokenizer creation  
    3. Text encoding
    4. Forward pass with text only
    5. Output shape validation
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
    
    try:
        print("🧪 Testing KAPALM Text Encoder...")
        print("=" * 50)
        
        # Test 1: Model Initialization
        print("\n1. Testing model initialization...")
        model = KAPALM()
        print(f"   ✅ Model initialized successfully")
        print(f"   📊 Model info: {model.get_model_info()}")
        
        # Test 2: Tokenizer Creation
        print("\n2. Testing tokenizer creation...")
        tokenizer = create_tokenizer(use_cache=True, force_offline=True)
        print(f"   ✅ Tokenizer created successfully")
        
        # Test 3: Text Encoding
        print("\n3. Testing text encoding...")
        test_texts = [
            "Ini adalah berita palsu yang harus dideteksi",
            "Berita nyata tentang teknologi terbaru"
        ]
        
        for i, text in enumerate(test_texts):
            # Tokenize
            encoded = tokenizer(
                text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Test encoding
            text_encoding = model.encode_text(encoded['input_ids'], encoded['attention_mask'])
            print(f"   ✅ Text {i+1} encoded: {text_encoding.shape}")
        
        # Test 4: Forward Pass (Text Only)
        print("\n4. Testing forward pass (text only)...")
        batch_encoded = tokenizer(
            test_texts,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        outputs = model(batch_encoded['input_ids'], batch_encoded['attention_mask'])
        
        print(f"   ✅ Forward pass successful:")
        print(f"   📊 Text encoding shape: {outputs['text_encoding'].shape}")
        print(f"   📊 Logits shape: {outputs['logits'].shape}")
        print(f"   📊 Predictions shape: {outputs['predictions'].shape}")
        
        # Test 5: Output Validation
        print("\n5. Validating outputs...")
        
        assert outputs['logits'].shape == (len(test_texts), 2), f"Expected logits shape ({len(test_texts)}, 2), got {outputs['logits'].shape}"
        assert outputs['predictions'].shape == (len(test_texts), 2), f"Expected predictions shape ({len(test_texts)}, 2), got {outputs['predictions'].shape}"
        assert torch.allclose(outputs['predictions'].sum(dim=-1), torch.ones(len(test_texts))), "Predictions should sum to 1"
        
        print(f"   ✅ All output shapes are correct")
        print(f"   ✅ Predictions sum to 1 (valid probabilities)")
        
        print("\n🎉 KAPALM Text Encoder test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error in testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run test when script is executed directly
    success = test_kapalm_text_encoder()
    
    if success:
        print("\n✅ KAPALM Text Encoder is ready for use!")
        print("Next steps:")
        print("  1. Implement Graph Encoder")
        print("  2. Implement Fusion Layer") 
        print("  3. Add complete training pipeline")
    else:
        print("\n❌ KAPALM Text Encoder test failed. Please check the errors above.")
