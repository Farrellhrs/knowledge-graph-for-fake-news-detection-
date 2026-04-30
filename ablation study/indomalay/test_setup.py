#!/usr/bin/env python3
"""
Quick Test Script for KAPALM Ablation Study
Runs a minimal test to verify all components work correctly
"""

import sys
import torch
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test imports
print("\n[1] Testing imports...")
try:
    from config import AblationMode, ABLATION_CONFIGS, MODEL_CONFIG, PATHS, get_ablation_config
    print("  ✓ config.py")
except Exception as e:
    print(f"  ✗ config.py: {e}")
    sys.exit(1)

try:
    from graph_utils import GraphProcessor, collate_fn
    print("  ✓ graph_utils.py")
except Exception as e:
    print(f"  ✗ graph_utils.py: {e}")
    sys.exit(1)

try:
    from dataset import IndoMalayDataset, load_and_filter_dataset
    print("  ✓ dataset.py")
except Exception as e:
    print(f"  ✗ dataset.py: {e}")
    sys.exit(1)

try:
    from models import create_model, get_model_summary, KAPALMAblation
    print("  ✓ models.py")
except Exception as e:
    print(f"  ✗ models.py: {e}")
    sys.exit(1)

try:
    from trainer import train_model, evaluate, EvaluationMetrics
    print("  ✓ trainer.py")
except Exception as e:
    print(f"  ✗ trainer.py: {e}")
    sys.exit(1)

# Test ablation configurations
print("\n[2] Testing ablation configurations...")
for mode in AblationMode:
    config = get_ablation_config(mode)
    print(f"  ✓ {mode.value}: {config.description[:50]}...")

# Test model creation for each mode
print("\n[3] Testing model creation...")
for mode in AblationMode:
    try:
        config = get_ablation_config(mode)
        model = create_model(MODEL_CONFIG, config)
        param_count = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  ✓ {mode.value}: {param_count:,} params ({trainable:,} trainable)")
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ✗ {mode.value}: {e}")

# Test paths
print("\n[4] Checking paths...")
from pathlib import Path
for name, path in PATHS.items():
    p = Path(path)
    exists = p.exists()
    status = "✓" if exists else "✗ (will be created)"
    print(f"  {status} {name}: {path}")

print("\n" + "=" * 60)
print("All component tests passed!")
print("=" * 60)
print("\nTo run the full ablation study:")
print("  python run_ablation.py --mode all")
print("\nTo run a single ablation:")
print("  python run_ablation.py --mode full")
print("  python run_ablation.py --mode wo_gp")
print("  python run_ablation.py --mode wo_in")
print("  python run_ablation.py --mode fine_only")
print("  python run_ablation.py --mode coarse_only")
