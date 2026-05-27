# Intérprete de Lengua de Señas Colombiana (LSC)

Reconocimiento en tiempo real de 21 letras del alfabeto manual estático de la Lengua de Señas Colombiana usando cámara web + MediaPipe Hands.

**Letras**: `A B C D E F I K L M N O P Q R T U V W X Y`
(Las que requieren movimiento — G, H, J, Ñ, S, Z — no están incluidas.)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements_inference.txt
wget -O hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## Modelos disponibles

### Landmark-based (V2/V3)

| Modelo | Tipo | Features |
|--------|------|----------|
| Classic (Random Forest) | sklearn | 118/126 tabulares |
| GAT (Graph Attention) | PyTorch Geometric | grafo 21 nodos, 10/18-dim |
| GCN | PyTorch Geometric | grafo 21 nodos, 10/18-dim |

### CNN Hybrid (V3)

| Modelo | Arquitectura | Input |
|--------|-------------|-------|
| CNN | MobileNetV3-Small | crop RGB 224×224 |

## Ejecutar

### Intérprete en tiempo real

```bash
# Landmarks
python inference/lsc_inference_realtime_v2.py --model classic --artifacts model_artifacts_v2

# CNN
python inference/lsc_inference_realtime_cnn.py --artifacts model_artifacts_cnn
```

### Benchmark de estabilidad

```bash
python benchmarks/lsc_stability_benchmark_v3.py --letter A
python benchmarks/lsc_stability_benchmark_v3.py --all-letters
```

### Ver resultados (sin cámara)

```bash
python benchmarks/plot_benchmark_results.py
```

## Notebooks

| Notebook | Descripción |
|----------|-------------|
| `LSC_INTERPRETER_v3.ipynb` | Entrenamiento V3: landmarks 18-dim + palm normal + signed volumes |
| `LSC_CNN_Hybrid_Pipeline.ipynb` | CNN híbrida: MediaPipe bbox + MobileNetV3-Small fine-tuning |

## Estructura

```
.
├── inference/               # Scripts de inferencia en tiempo real
├── benchmarks/              # Scripts y resultados de benchmark
├── model_artifacts_v2/      # Modelos V2 (10-dim node features)
├── model_artifacts_v3/      # Modelos V3 (18-dim node features)
├── model_artifacts_cnn/     # Modelo CNN (.keras + .onnx)
├── generated/               # hand_landmarker.task (gitignored)
└── requirements_inference.txt
```
