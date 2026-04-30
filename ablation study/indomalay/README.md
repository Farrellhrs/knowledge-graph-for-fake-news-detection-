# KAPALM Ablation Study - Indo-Malay Fake News Detection

## Overview

This ablation study analyzes the contribution of different components in the KAPALM model for Indo-Malay fake news detection.

## Ablation Configurations

| Mode | Description | Final Representation | Components |
|------|-------------|---------------------|------------|
| `full` | Full KAPALM | concat(h, a, s) | Text + Coarse (IN) + Fine (GP) |
| `wo_gp` | w/o Graph Pooling | concat(h, a) | Remove fine-grained module |
| `wo_in` | w/o Interaction Node | concat(h, s) | Mean pooling instead of IN |
| `fine_only` | Fine-Grained Only | concat(h, s) | Pruned graph only |
| `coarse_only` | Coarse Only | concat(h, a) | Original graph, no pruning |

### Notation
- **h**: Text embedding from IndoBERT
- **a**: Coarse-grained knowledge (GAT + Interaction Node)
- **s**: Fine-grained knowledge (GAT + Attentive Graph Pooling)
- **IN**: Interaction Node
- **GP**: Graph Pooling (Attentive)

## Project Structure

```
ablation_study/indomalay/
├── config.py           # Configuration and ablation settings
├── dataset.py          # Dataset loading and preprocessing
├── graph_utils.py      # Graph processing utilities
├── models.py           # KAPALM model variants
├── trainer.py          # Training and evaluation utilities
├── run_ablation.py     # Main experiment runner
├── cache/              # Cached graph data
├── results/            # Experiment results
└── checkpoints/        # Model checkpoints
```

## Usage

### Run All Ablations
```bash
python run_ablation.py --mode all
```

### Run Specific Ablation
```bash
python run_ablation.py --mode full
python run_ablation.py --mode wo_gp
python run_ablation.py --mode wo_in
python run_ablation.py --mode fine_only
python run_ablation.py --mode coarse_only
```

### Custom Output
```bash
python run_ablation.py --mode all --output my_results.json
```

## Results Table Template

### F1 Score Results (5-Fold Cross-Validation)

| Model Variant | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean | Std |
|---------------|--------|--------|--------|--------|--------|------|-----|
| Full Model | | | | | | | |
| w/o Graph Pooling | | | | | | | |
| w/o Interaction Node | | | | | | | |
| Fine-Grained Only | | | | | | | |
| Coarse Only | | | | | | | |

### All Metrics Summary (Mean ± Std)

| Model Variant | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---------------|----------|-----------|--------|-----|---------|
| Full Model | | | | | |
| w/o Graph Pooling | | | | | |
| w/o Interaction Node | | | | | |
| Fine-Grained Only | | | | | |
| Coarse Only | | | | | |

## Interpretation Guide

### What Performance Differences Would Imply

#### 1. Full Model vs w/o Graph Pooling (w/o GP)
- **If Full > w/o GP**: Fine-grained knowledge via attentive pooling captures entity-specific relationships critical for Indo-Malay fake news detection. This suggests misinformation in Indo-Malay context often targets specific entities or makes false claims about particular relationships.
- **If Full ≈ w/o GP**: The fine-grained details are redundant; the coarse-grained global structure is sufficient for this dataset. Indo-Malay fake news may operate at a higher semantic level.

#### 2. Full Model vs w/o Interaction Node (w/o IN)
- **If Full > w/o IN**: The learnable interaction node provides valuable aggregated graph context beyond simple mean pooling. It captures complex interactions between knowledge graph elements that are important for Indo-Malay fake news patterns.
- **If Full ≈ w/o IN**: The interaction node doesn't add significant value; mean aggregation captures the necessary global structure.

#### 3. Fine-Grained Only vs Coarse Only
- **If Fine > Coarse**: Local entity neighborhoods (pruned subgraphs around key entities) are more informative. Indo-Malay fake news likely involves entity-centric misinformation patterns.
- **If Coarse > Fine**: The overall knowledge structure matters more than specific entity relationships. Fake news affects broader knowledge coherence rather than specific facts.

#### 4. Single vs Dual Representations
- **If Full > max(Fine, Coarse)**: Both scales of knowledge are complementary and necessary. Indo-Malay fake news both distorts local entity facts AND disrupts broader knowledge patterns.
- **If Full ≈ max(Fine, Coarse)**: One scale dominates; the other provides minimal additional information.

#### 5. Impact of Graph Pruning
- **Coarse Only (original graph) vs Full (with centrality pruning)**:
  - If pruning helps: Indo-Malay knowledge graphs contain noise that pruning removes.
  - If pruning hurts: Full graph structure contains important peripheral information.

### Indo-Malay Specific Considerations

For Indo-Malay fake news detection specifically:
- Indonesian/Malay news often involves regional entities and cultural contexts
- Knowledge graphs may capture relationships between political figures, institutions, and events unique to the region
- The dual-scale approach may help capture both:
  - **Coarse**: General topic/domain coherence
  - **Fine**: Specific claim-entity relationships

## Requirements

```
torch>=2.0.0
transformers>=4.30.0
adapters>=0.1.0
torch-geometric>=2.3.0
pandas>=1.5.0
scikit-learn>=1.2.0
networkx>=3.0
numpy>=1.24.0
```

## Citation

If you use this code, please cite the original KAPALM paper and this ablation study implementation.
