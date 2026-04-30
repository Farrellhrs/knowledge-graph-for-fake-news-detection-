"""
Improved Parallel Dataset Preprocessing Script

This version addresses the bottlenecks in the original parallel implementation:
1. Pre-loads models to avoid repeated initialization
2. Better memory management
3. Option to split dataset for true parallel execution across multiple instances

Usage for single instance:
    python preprocess_dataset_improved.py --parallel --workers 2

Usage for multiple instances (recommended):
    python preprocess_dataset_improved.py --split 4 --part 0
    python preprocess_dataset_improved.py --split 4 --part 1
    python preprocess_dataset_improved.py --split 4 --part 2
    python preprocess_dataset_improved.py --split 4 --part 3
"""

import os
import sys
import pandas as pd
import networkx as nx
from datetime import datetime
import pickle
import traceback
from tqdm import tqdm
import time
import argparse
import multiprocessing as mp
import glob
import threading
import hashlib
import torch
import gc

# Import our custom GraphPipeline and utilities
from graph_pipeline_indo import GraphPipeline
from graph_io_utils import save_graph as save_graph_util

# Configuration
GEMINI_API_KEY = "AIzaSyAvMQynyVcZqvSqPQF3JRZDVaco5Lpa7iE"
DATASET_PATH = "../dataset/dataset_sunda_malay__filtered_46k_no_dup.csv"
OUTPUT_DIR = "../processed_graphs_indo_malay"

# Global pipeline for multiprocessing (to avoid repeated initialization)
global_pipeline = None
USE_CPU_ONLY = False  # Will be set by command line argument

def clear_gpu_memory():
    """Clear GPU memory to prevent OOM errors"""
    if torch.cuda.is_available() and not USE_CPU_ONLY:
        torch.cuda.empty_cache()
        gc.collect()

def initialize_worker(gemini_api_key, use_cpu=False):
    """Initialize the global pipeline for worker processes"""
    global global_pipeline, USE_CPU_ONLY
    USE_CPU_ONLY = use_cpu
    
    print(f"🤖 Initializing worker process {os.getpid()}")
    
    # Force CPU if requested
    if use_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print(f"  ⚠️ Forcing CPU-only mode")
    
    global_pipeline = GraphPipeline(gemini_api_key=gemini_api_key)
    print(f"✅ Worker {os.getpid()} initialized")

def worker_process_article(args):
    """Optimized worker that uses pre-initialized pipeline"""
    global global_pipeline
    
    try:
        text, article_id, row_index, total, output_dir = args
        
        if global_pipeline is None:
            raise RuntimeError("Pipeline not initialized in worker")
        
        # Use the global pipeline (already initialized)
        return process_single_article(global_pipeline, text, article_id, row_index, total, output_dir)
    
    except Exception as e:
        print(f"❌ Worker error for article {article_id}: {e}")
        return False, {'error': str(e), 'article_id': article_id}

def split_dataset(df, num_splits, part_index):
    """Split dataset into parts for parallel execution across multiple instances"""
    if part_index >= num_splits:
        raise ValueError(f"Part index {part_index} must be less than num_splits {num_splits}")
    
    total_rows = len(df)
    rows_per_part = total_rows // num_splits
    remainder = total_rows % num_splits
    
    # Calculate start and end indices
    start_idx = part_index * rows_per_part
    
    # Distribute remainder among first few parts
    if part_index < remainder:
        start_idx += part_index
        end_idx = start_idx + rows_per_part + 1
    else:
        start_idx += remainder
        end_idx = start_idx + rows_per_part
    
    # Handle last part edge case
    if part_index == num_splits - 1:
        end_idx = total_rows
    
    subset = df.iloc[start_idx:end_idx].copy()
    
    print(f"📊 Dataset split {part_index + 1}/{num_splits}:")
    print(f"  - Rows: {start_idx} to {end_idx-1} ({len(subset)} articles)")
    print(f"  - Total dataset: {total_rows} articles")
    
    return subset

def process_single_article(pipeline, text, article_id, row_index, total, output_dir):
    """Process a single article and return success status"""
    try:
        # Clear GPU memory before processing each article to prevent OOM
        if row_index % 5 == 0:  # Clear every 5 articles
            clear_gpu_memory()
        
        # Create metadata
        metadata = {
            'article_id': article_id,
            'row_index': row_index,
            'text_length': len(text),
            'processing_timestamp': datetime.now().isoformat(),
            'text_preview': text[:200] + "..." if len(text) > 200 else text
        }
        
        print(f"📄 Processing {article_id} (row {row_index + 1}/{total}, {len(text)} chars)")
        
        # Generate knowledge graph
        start_time = time.time()
        knowledge_graph = pipeline.generate_graph(text)
        processing_time = time.time() - start_time
        
        # Clear GPU memory after processing
        clear_gpu_memory()
        
        # Add processing info to metadata
        metadata.update({
            'processing_time_seconds': processing_time,
            'nodes_count': knowledge_graph.number_of_nodes(),
            'edges_count': knowledge_graph.number_of_edges(),
            'success': True
        })
        
        # Save the graph using article ID
        success = save_graph(knowledge_graph, article_id, output_dir, metadata)
        
        if success:
            print(f"✅ {article_id}: {knowledge_graph.number_of_nodes()} nodes, "
                  f"{knowledge_graph.number_of_edges()} edges, {processing_time:.2f}s")
        else:
            print(f"⚠️ {article_id}: empty graph")
            
        return success, metadata
        
    except Exception as e:
        print(f"❌ Error processing {article_id}: {e}")
        
        error_metadata = {
            'article_id': article_id,
            'row_index': row_index,
            'text_length': len(text),
            'processing_timestamp': datetime.now().isoformat(),
            'error': str(e),
            'traceback': traceback.format_exc(),
            'success': False
        }
        
        # Save error info
        safe_id = str(article_id).replace('/', '_').replace('\\', '_').replace(':', '_')
        error_filename = os.path.join(output_dir, 'failed', f'graph_{safe_id}_processing_error.txt')
        os.makedirs(os.path.dirname(error_filename), exist_ok=True)
        
        with open(error_filename, 'w', encoding='utf-8') as f:
            f.write(f"Processing error for article {article_id}\n")
            f.write(f"Error: {e}\n")
            f.write(f"Text length: {len(text)}\n")
            f.write(f"Text preview: {text[:200]}...\n")
            f.write(f"Traceback:\n{traceback.format_exc()}\n")
            
        return False, error_metadata

def save_graph(graph, article_id, output_dir, metadata=None):
    """Save graph using networkx pickle format with article ID as filename"""
    try:
        # Sanitize article ID for filename
        safe_id = str(article_id).replace('/', '_').replace('\\', '_').replace(':', '_')
        
        # Ensure directories exist
        os.makedirs(os.path.join(output_dir, 'successful'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'failed'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'metadata'), exist_ok=True)
        
        if graph.number_of_nodes() == 0:
            # Save empty graph info
            filename = os.path.join(output_dir, 'failed', f'graph_{safe_id}_empty.txt')
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Empty graph for article {article_id}\n")
                f.write(f"Metadata: {metadata}\n")
            return False
        
        # Save successful graph
        filename = os.path.join(output_dir, 'successful', f'graph_{safe_id}.gpickle')
        success = save_graph_util(graph, filename)
        
        if not success:
            print(f"Failed to save graph for article {article_id}")
            return False
        
        # Save metadata separately
        if metadata:
            metadata_filename = os.path.join(output_dir, 'metadata', f'metadata_{safe_id}.json')
            import json
            with open(metadata_filename, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
        
        return True
        
    except Exception as e:
        print(f"Error saving graph {article_id}: {e}")
        return False

def get_already_processed_files(output_dir, include_errors=True):
    """Get list of already processed article IDs
    
    Args:
        output_dir: Directory containing processed files
        include_errors: If True, include processing errors as "already processed"
                       If False, only include successful and empty graphs
    """
    processed_ids = set()
    
    # Check successful files
    successful_pattern = os.path.join(output_dir, 'successful', 'graph_*.gpickle')
    for filepath in glob.glob(successful_pattern):
        filename = os.path.basename(filepath)
        article_id = filename[6:-9]  # Remove 'graph_' prefix and '.gpickle' suffix
        processed_ids.add(article_id)
    
    # Check failed files
    failed_pattern = os.path.join(output_dir, 'failed', 'graph_*')
    for filepath in glob.glob(failed_pattern):
        filename = os.path.basename(filepath)
        if filename.startswith('graph_'):
            if filename.endswith('_empty.txt'):
                article_id = filename[6:-10]
                processed_ids.add(article_id)
            elif filename.endswith('_processing_error.txt'):
                # Only include processing errors if include_errors is True
                if include_errors:
                    article_id = filename[6:-20]
                    processed_ids.add(article_id)
            elif filename.endswith('_error.txt'):
                if include_errors:
                    article_id = filename[6:-10]
                    processed_ids.add(article_id)
    
    return processed_ids


def get_error_files(output_dir):
    """Get list of article IDs that had processing errors (for retry)"""
    error_ids = set()
    
    failed_pattern = os.path.join(output_dir, 'failed', 'graph_*_processing_error.txt')
    for filepath in glob.glob(failed_pattern):
        filename = os.path.basename(filepath)
        if filename.startswith('graph_') and filename.endswith('_processing_error.txt'):
            # Remove 'graph_' prefix (6 chars) and '_processing_error.txt' suffix (21 chars)
            article_id = filename[6:-21]
            error_ids.add(article_id)
    
    return error_ids


def delete_error_files(output_dir, error_ids):
    """Delete error files for articles that will be retried"""
    deleted_count = 0
    for article_id in error_ids:
        safe_id = str(article_id).replace('/', '_').replace('\\', '_').replace(':', '_')
        error_file = os.path.join(output_dir, 'failed', f'graph_{safe_id}_processing_error.txt')
        if os.path.exists(error_file):
            os.remove(error_file)
            deleted_count += 1
    print(f"🗑️ Deleted {deleted_count} error files for retry")

def main():
    """Main processing function"""
    global USE_CPU_ONLY
    
    parser = argparse.ArgumentParser(description='Improved parallel processing for knowledge graphs')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel processing within this instance')
    parser.add_argument('--workers', type=int, default=2, help='Number of worker processes (recommend 2 for API limits)')
    parser.add_argument('--split', type=int, help='Split dataset into N parts for multiple script instances')
    parser.add_argument('--part', type=int, help='Process part number (0 to split-1)')
    parser.add_argument('--resume', action='store_true', help='Resume from already processed files')
    parser.add_argument('--retry-errors', action='store_true', help='Only retry articles that had processing errors')
    parser.add_argument('--cpu', action='store_true', help='Force CPU-only processing (avoid CUDA OOM)')
    
    args = parser.parse_args()
    
    # Set CPU-only mode globally
    USE_CPU_ONLY = args.cpu
    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print("⚠️ CPU-only mode enabled - CUDA disabled")
    
    # Validate split arguments
    if args.split is not None and args.part is None:
        print("❌ Error: --part must be specified when using --split")
        return 1
    
    if args.part is not None and args.split is None:
        print("❌ Error: --split must be specified when using --part")
        return 1
    
    print("🚀 Improved Dataset Preprocessing for Knowledge Graph Generation")
    print("=" * 70)
    
    if args.split:
        print(f"📊 Dataset splitting mode: Processing part {args.part + 1}/{args.split}")
        # Create unique log file for this part
        log_file = f"processing_log_part_{args.part}.txt"
        output_dir = f"{OUTPUT_DIR}_part_{args.part}"
    else:
        log_file = "processing_log.txt"
        output_dir = OUTPUT_DIR
        
    print(f"📁 Output directory: {output_dir}")
    print(f"📋 Log file: {log_file}")
    
    try:
        # Load and prepare dataset
        print("📖 Loading dataset...")
        df = pd.read_csv(DATASET_PATH)
        print(f"✅ Loaded {len(df)} articles")
        
        # Handle retry-errors mode FIRST (before splitting)
        # This is because error IDs need to be matched against the FULL dataset
        if args.retry_errors:
            error_ids = get_error_files(output_dir)
            if error_ids:
                print(f"🔄 Retry-errors mode: Found {len(error_ids)} articles with processing errors")
                
                # Debug: print sample error IDs
                sample_errors = list(error_ids)[:5]
                print(f"  📋 Sample error IDs: {sample_errors}")
                
                # Convert dataset IDs to safe format for matching
                df['safe_id'] = df['id'].astype(str).str.replace('/', '_', regex=False).str.replace('\\', '_', regex=False).str.replace(':', '_', regex=False)
                
                # Also create a row index column for matching (in case error files used row indices)
                df['row_idx'] = df.index.astype(str)
                
                # Debug: print sample dataset IDs
                sample_df_ids = df['safe_id'].head(5).tolist()
                print(f"  📋 Sample dataset 'id' column: {sample_df_ids}")
                print(f"  📋 Sample dataset row indices: {df['row_idx'].head(5).tolist()}")
                
                # Try to match by 'id' column first
                df_by_id = df[df['safe_id'].isin(error_ids)]
                
                # If no matches by ID, try matching by row index
                if len(df_by_id) == 0:
                    print("  ⚠️ No matches by 'id' column, trying row index...")
                    df_by_idx = df[df['row_idx'].isin(error_ids)]
                    
                    if len(df_by_idx) > 0:
                        print(f"  ✅ Found {len(df_by_idx)} matches by row index!")
                        df = df_by_idx
                    else:
                        print("  ❌ No matches by row index either!")
                        
                        # Check error ID range vs dataset range
                        error_ids_int = sorted([int(x) for x in error_ids if x.isdigit()])
                        if error_ids_int:
                            print(f"  📊 Error ID range: {min(error_ids_int)} - {max(error_ids_int)}")
                            print(f"  📊 Dataset row range: 0 - {len(df)-1}")
                            print(f"  📊 Dataset 'id' range: {df['id'].min()} - {df['id'].max()}")
                        
                        return 0
                else:
                    print(f"  ✅ Found {len(df_by_id)} matches by 'id' column!")
                    df = df_by_id
                
                print(f"📊 Will retry: {len(df)} articles")
                
                # Delete old error files so they can be reprocessed
                delete_error_files(output_dir, error_ids)
                
                # Skip splitting when retrying errors - we already filtered to specific articles
                print("  ℹ️ Skipping dataset split (retry-errors mode uses full dataset filter)")
            else:
                print("✅ No processing errors found to retry!")
                return 0
        else:
            # Split dataset if requested (only when NOT retrying errors)
            if args.split:
                df = split_dataset(df, args.split, args.part)
            
            # Filter already processed if resuming
            if args.resume:
                already_processed = get_already_processed_files(output_dir, include_errors=True)
                if already_processed:
                    print(f"📋 Resume mode: Found {len(already_processed)} already processed")
                    safe_ids = df['id'].astype(str).str.replace('/', '_', regex=False).str.replace('\\', '_', regex=False).str.replace(':', '_', regex=False)
                    df = df[~safe_ids.isin(already_processed)]
                    print(f"📊 Remaining to process: {len(df)} articles")
        
        if len(df) == 0:
            print("✅ No articles to process!")
            return 0
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process articles
        start_time = time.time()
        
        if args.parallel and len(df) > 1:
            print(f"🔄 Parallel processing with {args.workers} workers...")
            if args.cpu:
                print("  ⚠️ Using CPU-only mode for all workers")
            
            # Prepare arguments for workers
            worker_args = [
                (row['fulltext'], row['id'], idx, len(df), output_dir)
                for idx, (_, row) in enumerate(df.iterrows())
            ]
            
            # Process with multiprocessing
            with mp.Pool(processes=args.workers, initializer=initialize_worker, initargs=(GEMINI_API_KEY, args.cpu)) as pool:
                results = list(tqdm(
                    pool.imap(worker_process_article, worker_args),
                    total=len(worker_args),
                    desc="Processing articles"
                ))
        else:
            print("🔄 Sequential processing...")
            if args.cpu:
                print("  ⚠️ Using CPU-only mode")
            
            # Initialize pipeline once
            pipeline = GraphPipeline(gemini_api_key=GEMINI_API_KEY)
            
            results = []
            for idx, (_, row) in enumerate(tqdm(df.iterrows(), desc="Processing articles")):
                result = process_single_article(
                    pipeline, row['fulltext'], row['id'], idx, len(df), output_dir
                )
                results.append(result)
        
        # Calculate statistics
        total_time = time.time() - start_time
        successful = sum(1 for success, _ in results if success)
        failed = len(results) - successful
        
        print("\n" + "=" * 60)
        print("🎯 PROCESSING COMPLETED!")
        print("=" * 60)
        print(f"📊 Statistics:")
        print(f"  - Total articles: {len(results)}")
        print(f"  - Successful: {successful}")
        print(f"  - Failed: {failed}")
        print(f"  - Success rate: {(successful/len(results)*100):.1f}%")
        print(f"  - Total time: {total_time/3600:.2f} hours")
        print(f"  - Average per article: {total_time/len(results):.2f} seconds")
        print(f"📁 Output saved to: {output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
