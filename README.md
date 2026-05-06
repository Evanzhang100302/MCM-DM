# MCM-DM: Towards Better Spatio-Temporal Event Representation Learning via Discrete Morse Theory

## Requirements

```bash
pip install -r requirements.txt
```

## Datasets

Two sample datasets are included:

| Dataset   | Domain          | Zoom |
|-----------|-----------------|------|
| Crime2026 | Urban crime (Philadelphia, 2026) | 12 |
| Twitter   | Social media check-ins           | 14 |

Place additional datasets under `dataset/<DATASET_NAME>/` with files `data_train.pkl`, `data_val.pkl`, `data_test.pkl`. Each file is a list of sequences, where each sequence is a list of events `[time, lon, lat]`.

## Usage

```bash
bash run.sh [DATASET] [CUDA_ID] [SEED] [EPOCHS]
```

**Examples:**
```bash
bash run.sh Crime2026 0 42 100
bash run.sh Twitter   0 42 300
```

If using a custom Python environment:
```bash
PYTHON=/path/to/python bash run.sh Crime2026 0 42 100
```

## Pipeline

The script runs four steps automatically, skipping any step whose output already exists:

| Step | Script | Output |
|------|--------|--------|
| 1 | `extract_tile_embeddings.py` | `tile_embeddings_{DATASET}_z{ZOOM}.pt` |
| 2 | `build_morse_features.py` | `morse_features_{DATASET}_z{ZOOM}.pt` |
| 3 | `train_mm_morse_temporal.py` | `models_mm/{DATASET}_ts500_morse_temporal_seed{SEED}/` |
| 4 | `test_morse_pure.py` | MAE / RMSE printed to stdout |

## File Structure

```
MCM-DM/
├── run.sh                        # Full pipeline runner
├── extract_tile_embeddings.py    # Step 1: VLM tile embedding extraction
├── build_morse_features.py       # Step 2: Discrete Morse complex construction
├── train_mm_morse_temporal.py    # Step 3: Model training
├── test_morse_pure.py            # Step 4: Evaluation
├── model_mm.py                   # Model architecture
├── morse_function.py             # Morse theory utilities
├── transformer_st.py             # Spatio-temporal transformer encoder
├── dataset/
│   ├── Crime2026/
│   └── Twitter/
└── requirements.txt
```

