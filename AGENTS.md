# AGENTS.md — LSC Colombian Sign Language Interpreter

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements_inference.txt
```

Download the MediaPipe hand landmarker model (gitignored):
```bash
wget -O hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## Run real-time interpreter (webcam)

```bash
# Default: auto-selects best model from metadata
python lsc_inference_realtime_v2.py

# Specific model + explicit artifact dir
python lsc_inference_realtime_v2.py --model gcn --artifacts model_artifacts_v2
python lsc_inference_realtime_v2.py --model classic --artifacts model_artifacts_v2
python lsc_inference_realtime_v2.py --model gat     --artifacts model_artifacts_v2
python lsc_inference_realtime_v2.py --model gat_robusto --artifacts model_artifacts_v2
python lsc_inference_realtime_v2.py --model mlp     --artifacts model_artifacts_v2
python lsc_inference_realtime_v2.py --list-models   # list available artifacts
```

## Run stability benchmark (webcam)

```bash
# All 21 letters, 4 models (classic, gcn, gat, gat_robusto), 5s per letter
python lsc_stability_benchmark_v2.py --all-letters

# Single letter
python lsc_stability_benchmark_v2.py --letter A

# Plot from saved JSON (no camera)
python plot_benchmark_results.py
```

## Architecture

### Two generations (v1 → v2)

| | v1 (`lsc_inference_realtime.py`) | v2 (`lsc_inference_realtime_v2.py`) |
|---|---|---|
| Node features | 7-dim `[x,y,z, bone_vec, dist_center]` | 10-dim `[x,y,z, bone_vec, dist_center, polar_r, azimuth, elevation]` |
| Artifact dir | `model_artifacts/` | `model_artifacts_v2/` |
| Benchmark | `lsc_stability_benchmark.py` (v1 scripts) | `lsc_stability_benchmark_v2.py` (v2 scripts, evaluates all models simultaneously) |

**Always prefer v2** (`_v2` suffix) for new work. v1 is kept for reference.

### Model types in artifact dir

| Key | File | Input | Framework |
|-----|------|-------|-----------|
| `classic` | `mejor_modelo_clasico.joblib` or `modelos_clasicos.joblib` | tabular (118 features) | sklearn |
| `mlp` | `modelo_mlp.keras` | tabular (118 features) | TensorFlow/Keras |
| `gcn` | `modelo_gcn_pyg.pt` | graph (21 nodes, 10 features) | PyTorch Geometric |
| `gat` | `modelo_gat_mejorado.pt` | graph + edge_attr | PyTorch Geometric |
| `gat_robusto` | `modelo_gat_robusto.pt` | graph + edge_attr | PyTorch Geometric |

All require `label_encoder.joblib` and `metadata_inferencia.json` in the same directory.

### 21 LSC letters

`A B C D E F I K L M N O P Q R T U V W X Y`

Letters missing (require motion): G, H, J, Ñ, S, Z.

### Landmark processing pipeline

1. MediaPipe Hands extracts 21 landmarks (x,y,z)
2. Center at wrist (landmark 0), scale by max distance, rotate so landmark 9→+X
3. For tabular models: 63 coords + 21 anatomical distances + 10 fingertip distances + 5 wrist-tip distances + 15 finger angles + 4 palm angles = **118 features**
4. For graph models: 21-node graph with extra edges (fingertip pairs, neighbor MCPs, thumb-index)

### Artifact directories

- `model_artifacts/` — v1 (7-dim node features) — **outdated**
- `model_artifacts_v2/` — v2 (10-dim node features, HG-GCN style) — **current**
- `legacy/` — old notebooks and scripts, gitignored

## MLP requires TensorFlow

If TensorFlow isn't available for your Python version, use Python 3.10–3.12. See `requirements_inference_mlp_optional.txt`.

## No formal test suite, linting, or CI

This is a research/experimental repo. There are no tests (`tests/` contains only benchmark output images), no CI workflows, and no linter/formatter config.
