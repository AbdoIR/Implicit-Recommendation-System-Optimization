# Stochastic Optimization for an Implicit Recommendation System

Minimal project comparing stochastic optimizers for implicit Matrix Factorization on MovieLens 100K.

## Goal

Compare optimizers on an implicit recommendation task:

- SGD
- SGD with Momentum
- Adam
- Adagrad
- RMSprop
- FTRL

Metrics:

- Precision@10
- Recall@10
- Hit Rate@10
- NDCG@10
- Loss convergence
- Training time
- Memory usage
- Sparse user/item behavior

## Structure

```text
.
├── notebooks/
│   ├── 01_data_preparation_exploration.ipynb
│   ├── 02_training_baseline_sgd.ipynb
│   ├── 03_evaluation_metrics.ipynb
│   └── 04_optimizer_comparison.ipynb
├── src/
│   ├── data_utils.py
│   ├── model_utils.py
│   ├── training_utils.py
│   ├── evaluation_utils.py
│   └── plotting_utils.py
├── data/processed/
├── models/
├── results/
│   ├── figures/
│   └── tables/
└── requirements.txt
```

## Setup

```bash
python -m venv opt-venv
pip install -r requirements.txt
```

For GPU training, install a PyTorch build that matches your CUDA version from the official PyTorch instructions.

## Dataset

Download it with:

```bash
python scripts/download_movielens_100k.py
```

This creates:

```text
dataset/ml-100k/
```

Required files:

- `u.data`
- `u.user`
- `u.item`

## Run Order

Run notebooks in order:

1. `01_data_preparation_exploration.ipynb`
2. `02_training_baseline_sgd.ipynb`
3. `03_evaluation_metrics.ipynb`
4. `04_optimizer_comparison.ipynb`

Generated files are written to:

- `data/processed/`
- `models/`
- `results/figures/`
- `results/tables/`

## Notes

The project focuses only on implicit Matrix Factorization. Advanced recommenders such as NCF, autoencoders, and LightGCN are intentionally not used.
