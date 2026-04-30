#!/usr/bin/env python3
"""
Training and Evaluation Utilities for KAPALM Ablation Study
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    classification_report, roc_auc_score, confusion_matrix
)

import numpy as np
import logging
from typing import Dict, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    precision_macro: float
    recall_macro: float
    f1_macro: float


def train_epoch(model, train_loader, optimizer, scheduler, scaler, device, max_grad_norm=1.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    predictions = []
    true_labels = []
    
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        coarse_batch = batch['coarse_batch'].to(device) if batch['coarse_batch'] else None
        fine_batch = batch['fine_batch'].to(device) if batch['fine_batch'] else None
        
        optimizer.zero_grad()
        
        with autocast('cuda'):
            logits = model(input_ids, attention_mask, coarse_batch, fine_batch)
            loss = F.cross_entropy(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
    
    train_acc = accuracy_score(true_labels, predictions)
    train_f1 = precision_recall_fscore_support(true_labels, predictions, average='binary', zero_division=0)[2]
    
    return total_loss / len(train_loader), train_acc, train_f1


def evaluate(model, data_loader, device) -> EvaluationMetrics:
    """Evaluate model on a dataset"""
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
            
            probs = F.softmax(logits, dim=1)
            prediction_probs.extend(probs[:, 1].cpu().numpy())
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='binary', zero_division=0)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, predictions, average='macro', zero_division=0)
    
    try:
        roc_auc = roc_auc_score(true_labels, prediction_probs)
    except:
        roc_auc = 0.0
    
    return EvaluationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        precision_macro=precision_macro,
        recall_macro=recall_macro,
        f1_macro=f1_macro
    )


def train_model(
    model, 
    train_loader, 
    val_loader, 
    config: Dict,
    verbose: bool = True,
    early_stopping_patience: int = 10
) -> Tuple[nn.Module, List[Dict], float]:
    """
    Train the model with early stopping.
    
    Returns:
        (trained_model, training_log, best_val_f1)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    total_steps = len(train_loader) * config['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=total_steps
    )
    scaler = GradScaler('cuda')
    
    training_log = []
    best_val_f1 = 0.0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(config['num_epochs']):
        # Training
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, config['max_grad_norm']
        )
        
        # Validation
        val_metrics = evaluate(model, val_loader, device)
        
        # Logging
        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_f1': train_f1,
            'val_acc': val_metrics.accuracy,
            'val_f1': val_metrics.f1,
            'val_auc': val_metrics.roc_auc
        }
        training_log.append(log_entry)
        
        if verbose and (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{config['num_epochs']}: "
                       f"Train Loss={train_loss:.4f}, Train F1={train_f1:.4f}, "
                       f"Val F1={val_metrics.f1:.4f}, Val AUC={val_metrics.roc_auc:.4f}")
        
        # Early stopping check
        if val_metrics.f1 > best_val_f1:
            best_val_f1 = val_metrics.f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, training_log, best_val_f1


def get_detailed_evaluation(model, data_loader, device, label_names=['Real', 'Fake']) -> Dict:
    """Get detailed evaluation including confusion matrix and classification report"""
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
            
            probs = F.softmax(logits, dim=1)
            prediction_probs.extend(probs[:, 1].cpu().numpy())
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # Metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='binary', zero_division=0)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, predictions, average='macro', zero_division=0)
    
    try:
        roc_auc = roc_auc_score(true_labels, prediction_probs)
    except:
        roc_auc = 0.0
    
    cm = confusion_matrix(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=label_names, output_dict=True)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': predictions,
        'true_labels': true_labels,
        'prediction_probs': prediction_probs
    }
