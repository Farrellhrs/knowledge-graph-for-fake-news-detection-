#!/usr/bin/env python3
"""
KAPALM Ablation Study - Main Experiment Runner
PolitiFact Fake News Detection

This script runs ablation experiments with 5-fold cross-validation:
- Full Model: concat(h, a, s)
- w/o Graph Pooling: concat(h, a)  
- w/o Interaction Node: concat(h, s)
- Fine-Grained Only: concat(h, s) with pruned graph
- Coarse Only: concat(h, a) with original graph

Usage:
    python run_ablation.py --mode full           # Run full model only
    python run_ablation.py --mode wo_gp          # Run w/o Graph Pooling
    python run_ablation.py --mode wo_in          # Run w/o Interaction Node
    python run_ablation.py --mode fine_only      # Run Fine-Grained Only
    python run_ablation.py --mode coarse_only    # Run Coarse Only
    python run_ablation.py --mode all            # Run all ablation experiments
"""

import os
import sys
import argparse
import logging
import json
import random
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from transformers.models.auto.tokenization_auto import AutoTokenizer

# Local imports
from config import (
    AblationMode, AblationConfig, ABLATION_CONFIGS,
    MODEL_CONFIG, CV_CONFIG, PATHS, get_ablation_config
)
from graph_utils import GraphProcessor, collate_fn
from models import create_model, get_model_summary
from trainer import train_model, evaluate, get_detailed_evaluation

# Force-load local dataset module to avoid conflict with external 'dataset' packages.
_dataset_spec = importlib.util.spec_from_file_location(
    "politifact_local_dataset",
    Path(__file__).with_name("dataset.py")
)
if _dataset_spec is None or _dataset_spec.loader is None:
    raise ImportError("Failed to load local dataset module from dataset.py")
dataset_module: Any = importlib.util.module_from_spec(_dataset_spec)
_dataset_spec.loader.exec_module(dataset_module)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ablation_experiment.log')
    ]
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_single_fold(
    fold: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer,
    graph_processor: GraphProcessor,
    ablation_config: AblationConfig,
    seed: int
) -> dict:
    """Run a single fold experiment"""
    set_seed(seed)
    
    # Determine cache suffix based on ablation mode
    cache_suffix = f"_{ablation_config.mode.value}"
    
    # Create datasets
    train_dataset = dataset_module.PolitiFactDataset(
        train_df, tokenizer, graph_processor, PATHS['graphs_dir'],
        max_length=MODEL_CONFIG['max_seq_length'],
        use_pruned_graph=ablation_config.use_pruned_graph,
        cache_suffix=cache_suffix
    )
    val_dataset = dataset_module.PolitiFactDataset(
        val_df, tokenizer, graph_processor, PATHS['graphs_dir'],
        max_length=MODEL_CONFIG['max_seq_length'],
        use_pruned_graph=ablation_config.use_pruned_graph,
        cache_suffix=cache_suffix
    )
    test_dataset = dataset_module.PolitiFactDataset(
        test_df, tokenizer, graph_processor, PATHS['graphs_dir'],
        max_length=MODEL_CONFIG['max_seq_length'],
        use_pruned_graph=ablation_config.use_pruned_graph,
        cache_suffix=cache_suffix
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=MODEL_CONFIG['batch_size'],
        shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=MODEL_CONFIG['batch_size'],
        shuffle=False, collate_fn=collate_fn, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=MODEL_CONFIG['batch_size'],
        shuffle=False, collate_fn=collate_fn, num_workers=0
    )
    
    # Create model
    model = create_model(MODEL_CONFIG, ablation_config)
    
    logger.info(f"  Training fold {fold}...")
    
    # Train model
    model, training_log, best_val_f1 = train_model(
        model, train_loader, val_loader, MODEL_CONFIG, verbose=False
    )
    
    # Evaluate on test set
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    test_results = get_detailed_evaluation(model, test_loader, device)
    
    logger.info(f"  Fold {fold} - Test Acc: {test_results['accuracy']:.4f}, "
               f"F1: {test_results['f1']:.4f}, AUC: {test_results['roc_auc']:.4f}")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return {
        'fold': fold,
        'seed': seed,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'accuracy': test_results['accuracy'],
        'precision': test_results['precision'],
        'recall': test_results['recall'],
        'f1': test_results['f1'],
        'roc_auc': test_results['roc_auc'],
        'f1_macro': test_results['f1_macro'],
        'best_val_f1': best_val_f1,
        'training_log': training_log
    }


def run_ablation_experiment(ablation_mode: AblationMode) -> dict:
    """Run complete ablation experiment with 5-fold cross-validation"""
    ablation_config = get_ablation_config(ablation_mode)
    
    logger.info("=" * 80)
    logger.info(f"ABLATION EXPERIMENT: {ablation_mode.value}")
    logger.info(f"Description: {ablation_config.description}")
    logger.info("=" * 80)
    
    # Load and filter dataset
    df = dataset_module.load_and_filter_dataset(PATHS['dataset'], PATHS['graphs_dir'])
    
    # Initialize tokenizer and graph processor
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['model_name'])
    graph_processor = GraphProcessor(
        PATHS['cache_dir'], 
        centrality_top_n=MODEL_CONFIG['centrality_top_n']
    )
    
    # 5-fold stratified cross-validation
    skf = StratifiedKFold(n_splits=CV_CONFIG['n_folds'], shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_val_idx, test_idx) in enumerate(skf.split(df, df['label']), 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold}/{CV_CONFIG['n_folds']}")
        logger.info(f"{'='*60}")
        
        # Split data
        train_val_df = df.iloc[train_val_idx]
        test_df = df.iloc[test_idx]
        
        # Further split train_val into train and validation (80/20)
        train_df = train_val_df.sample(frac=0.8, random_state=CV_CONFIG['random_seeds'][fold-1])
        val_df = train_val_df.drop(train_df.index)
        
        # Run fold
        seed = CV_CONFIG['random_seeds'][fold - 1]
        result = run_single_fold(
            fold, train_df, val_df, test_df,
            tokenizer, graph_processor, ablation_config, seed
        )
        fold_results.append(result)
    
    # Aggregate results
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'f1_macro']
    aggregated = {
        'mode': ablation_mode.value,
        'description': ablation_config.description,
        'config': {
            'use_coarse': ablation_config.use_coarse,
            'use_fine': ablation_config.use_fine,
            'use_interaction_node': ablation_config.use_interaction_node,
            'use_attentive_pooling': ablation_config.use_attentive_pooling,
            'use_pruned_graph': ablation_config.use_pruned_graph
        },
        'fold_results': fold_results,
        'summary': {}
    }
    
    for metric in metrics:
        values = [r[metric] for r in fold_results]
        aggregated['summary'][metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info(f"RESULTS SUMMARY: {ablation_mode.value}")
    logger.info("=" * 80)
    for metric in metrics:
        mean = aggregated['summary'][metric]['mean']
        std = aggregated['summary'][metric]['std']
        logger.info(f"  {metric.upper()}: {mean:.4f} ± {std:.4f}")
    
    return aggregated


def run_all_ablations():
    """Run all ablation experiments"""
    all_results = {}
    
    modes = [
        AblationMode.FULL_MODEL,
        AblationMode.WITHOUT_GRAPH_POOLING,
        AblationMode.WITHOUT_INTERACTION_NODE,
        AblationMode.FINE_GRAINED_ONLY,
        AblationMode.COARSE_ONLY
    ]
    
    for mode in modes:
        results = run_ablation_experiment(mode)
        all_results[mode.value] = results
        
        # Save intermediate results
        save_results(all_results, 'ablation_results_intermediate.json')
    
    return all_results


def save_results(results: dict, filename: str):
    """Save results to JSON file"""
    results_dir = Path(PATHS['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = results_dir / filename
    
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    
    logger.info(f"Results saved to {filepath}")


def generate_results_table(results: dict) -> pd.DataFrame:
    """Generate a results table for reporting"""
    rows = []
    
    for mode, data in results.items():
        row = {'Model Variant': mode}
        
        # Add fold results
        for i, fold_result in enumerate(data['fold_results'], 1):
            row[f'Fold {i}'] = f"{fold_result['f1']:.4f}"
        
        # Add mean and std
        row['Mean'] = f"{data['summary']['f1']['mean']:.4f}"
        row['Std'] = f"{data['summary']['f1']['std']:.4f}"
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def print_final_report(results: dict):
    """Print comprehensive final report"""
    print("\n" + "=" * 100)
    print("KAPALM ABLATION STUDY - FINAL REPORT")
    print("Dataset: PolitiFact Fake News Detection")
    print("=" * 100)
    
    # Results table
    print("\n📊 F1 SCORE RESULTS (5-Fold Cross-Validation)")
    print("-" * 100)
    
    header = f"{'Model Variant':<25} {'Fold 1':<10} {'Fold 2':<10} {'Fold 3':<10} {'Fold 4':<10} {'Fold 5':<10} {'Mean':<12} {'Std':<10}"
    print(header)
    print("-" * 100)
    
    for mode, data in results.items():
        folds = [f"{r['f1']:.4f}" for r in data['fold_results']]
        mean = data['summary']['f1']['mean']
        std = data['summary']['f1']['std']
        
        row = f"{mode:<25} {folds[0]:<10} {folds[1]:<10} {folds[2]:<10} {folds[3]:<10} {folds[4]:<10} {mean:.4f}       {std:.4f}"
        print(row)
    
    print("-" * 100)
    
    # Additional metrics table
    print("\n📊 ALL METRICS SUMMARY (Mean ± Std)")
    print("-" * 100)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    header = f"{'Model Variant':<25} " + " ".join([f"{m.upper():<18}" for m in metrics])
    print(header)
    print("-" * 100)
    
    for mode, data in results.items():
        values = []
        for m in metrics:
            mean = data['summary'][m]['mean']
            std = data['summary'][m]['std']
            values.append(f"{mean:.4f}±{std:.4f}")
        
        row = f"{mode:<25} " + " ".join([f"{v:<18}" for v in values])
        print(row)
    
    print("-" * 100)
    
    # Analysis
    print("\n" + "=" * 100)
    print("📝 ANALYSIS: IMPLICATIONS FOR POLITIFACT KNOWLEDGE UTILIZATION")
    print("=" * 100)
    print("""
1. FULL MODEL vs w/o GRAPH POOLING (w/o GP):
   - If Full > w/o GP: Fine-grained knowledge (attentive pooling) helps capture 
    entity-specific relationships important for PolitiFact fake news detection.
   - If Full ≈ w/o GP: Fine-grained details may be redundant; coarse structure sufficient.

2. FULL MODEL vs w/o INTERACTION NODE (w/o IN):
   - If Full > w/o IN: The interaction node provides valuable global graph context
     that improves classification beyond simple mean pooling.
   - If Full ≈ w/o IN: Global structure can be captured through mean aggregation.

3. FINE-GRAINED ONLY vs COARSE ONLY:
   - If Fine > Coarse: Local entity neighborhoods are more informative for 
         PolitiFact fake news, suggesting entity-centric misinformation patterns.
   - If Coarse > Fine: Overall knowledge structure is more important than 
     specific entity relationships; fake news affects broader knowledge patterns.

4. BOTH REPRESENTATIONS vs SINGLE:
   - If Full > max(Fine, Coarse): Complementary information from both scales
         is needed for optimal detection in PolitiFact context.
     - This would suggest fake news in PolitiFact both distorts local entity
     facts AND disrupts broader knowledge coherence.

5. IMPACT OF GRAPH PRUNING:
   - Compare Coarse Only (no pruning) vs Full Model (with pruning):
    - If pruning helps: Noise reduction benefits PolitiFact knowledge graphs.
   - If pruning hurts: Full graph structure is important for this dataset.
""")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description='KAPALM Ablation Study for PolitiFact Dataset')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['full', 'wo_gp', 'wo_in', 'fine_only', 'coarse_only', 'all'],
                       help='Ablation mode to run')
    parser.add_argument('--output', type=str, default='ablation_results.json',
                       help='Output filename for results')
    
    args = parser.parse_args()
    
    # Create directories
    for path_key in ['cache_dir', 'results_dir', 'checkpoints_dir']:
        Path(PATHS[path_key]).mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("KAPALM ABLATION STUDY - POLITIFACT FAKE NEWS DETECTION")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info("=" * 80)
    
    if args.mode == 'all':
        results = run_all_ablations()
    else:
        mode_map = {
            'full': AblationMode.FULL_MODEL,
            'wo_gp': AblationMode.WITHOUT_GRAPH_POOLING,
            'wo_in': AblationMode.WITHOUT_INTERACTION_NODE,
            'fine_only': AblationMode.FINE_GRAINED_ONLY,
            'coarse_only': AblationMode.COARSE_ONLY
        }
        results = {args.mode: run_ablation_experiment(mode_map[args.mode])}
    
    # Save final results
    save_results(results, args.output)
    
    # Generate and save table
    if len(results) > 1:
        table = generate_results_table(results)
        table_path = Path(PATHS['results_dir']) / 'ablation_results_table.csv'
        table.to_csv(table_path, index=False)
        logger.info(f"Results table saved to {table_path}")
        
        # Print final report
        print_final_report(results)
    
    logger.info(f"\nCompleted: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
