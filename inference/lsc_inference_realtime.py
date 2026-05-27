"""
LSC Real-Time Inference Script — V3
=====================================
Interprete de Lengua de Señas Colombiana usando camara web + MediaPipe.

Soporta los 5 modelos entrenados en el notebook:
  - Random Forest / SVM / KNN  (classic ML + features geometricas)
  - MLP                        (Keras .keras + features geometricas)
  - GCN (HandGCNv2)            (PyTorch Geometric, node_dim=18)
  - GAT (HandGAT)              (PyTorch Geometric, node_dim=18 + edge_attr)
  - GAT Robusto                (identico a GAT, entrenado con aumentacion)

Preprocesamiento exactamente igual al entrenamiento:
  1. Extraer 21 landmarks (x,y,z) con MediaPipe Hands.
  2. Centrar en la muñeca (landmark 0).
  3. Escalar por la distancia maxima al origen.
  4. Rotar para alinear el landmark 9 (metacarpo medio) al eje +X.
  5. Para modelos tabulares: 126 features (coords + distancias + angulos
     + vector normal de palma + volumenes con signo).
  6. Para modelos de grafo: node_features (18-dim por nodo): 10 HG-GCN
     + 3 palm normal + 5 signed volumes + edge_attr.

Requisitos:
  pip install opencv-python mediapipe numpy joblib tensorflow
  pip install torch torch-geometric
  (MediaPipe >= 0.10 con la API Tasks)

Uso:
  python lsc_inference_realtime.py
  python lsc_inference_realtime.py --model gcn --artifacts model_artifacts_v3
  python lsc_inference_realtime.py --model mlp --artifacts model_artifacts_v3
  python lsc_inference_realtime.py --model classic
  python lsc_inference_realtime.py --model gat
  python lsc_inference_realtime.py --model gat_robusto
  python lsc_inference_realtime.py --list-models
"""

import argparse
import json
import sys
import time
from collections import Counter, deque
from itertools import combinations
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# CONEXIONES ANATOMICAS DE MEDIAPIPE HANDS (21 landmarks)
# ──────────────────────────────────────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# Aristas extra usadas en la GCN/GAT del entrenamiento
TIP_INDICES = [4, 8, 12, 16, 20]
NEIGHBOR_TIP_EDGES  = [(8, 12), (12, 16), (16, 20)]
NEIGHBOR_MCP_EDGES  = [(5, 9), (9, 13), (13, 17)]
THUMB_INDEX_EDGES   = [(4, 8), (1, 5)]
EXTRA_TIP_EDGES     = list(combinations(TIP_INDICES, 2))
EXTRA_EDGES         = EXTRA_TIP_EDGES + NEIGHBOR_TIP_EDGES + NEIGHBOR_MCP_EDGES + THUMB_INDEX_EDGES

# Indices para features geometricas tabulares
FINGER_CHAINS = {
    'pulgar':  [0, 1, 2, 3, 4],
    'indice':  [0, 5, 6, 7, 8],
    'medio':   [0, 9, 10, 11, 12],
    'anular':  [0, 13, 14, 15, 16],
    'menique': [0, 17, 18, 19, 20],
}
PALM_ANGLE_TRIPLES = [
    ('palma_indice_medio',   5,  0,  9),
    ('palma_medio_anular',   9,  0, 13),
    ('palma_anular_menique', 13, 0, 17),
    ('palma_indice_menique',  5, 0, 17),
]


# ──────────────────────────────────────────────────────────────────────────────
# PREPROCESAMIENTO (replica exacta del notebook)
# ──────────────────────────────────────────────────────────────────────────────

def _to_numpy_coords(landmarks):
    """Acepta lista de landmarks de MediaPipe o numpy array."""
    if isinstance(landmarks, np.ndarray):
        return landmarks.astype(np.float32)
    return np.array(
        [[lm.x, lm.y, getattr(lm, 'z', 0.0)] for lm in landmarks],
        dtype=np.float32,
    )


def normalize_landmarks(landmarks):
    """
    Normaliza coordenadas de la mano:
      - Centra en la muñeca (landmark 0).
      - Escala por la distancia maxima al origen.
      - Rota para alinear el vector muñeca->MCP3 (landmark 9) al eje +X.
    Retorna array (21, 3) o (21, 2) segun la entrada.
    """
    coords = _to_numpy_coords(landmarks)
    is_2d = coords.shape[1] == 2
    if is_2d:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 1), dtype=np.float32)])

    wrist   = coords[0]
    centered = coords - wrist
    scale    = np.linalg.norm(centered, axis=1).max()
    if scale > 0:
        centered = centered / scale

    ref = centered[9]
    angle = np.arctan2(ref[1], ref[0]) if np.linalg.norm(ref[:2]) > 0 else 0.0
    c, s = np.cos(-angle), np.sin(-angle)
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0., 0., 1.]], dtype=np.float32)
    centered = (R @ centered.T).T

    return centered[:, :2] if is_2d else centered


def build_adjacency(num_nodes, base_connections, extra_edges=None):
    """Construye aristas sin duplicados y las ordena."""
    edges = set()
    def add(a, b):
        if 0 <= a < num_nodes and 0 <= b < num_nodes and a != b:
            edges.add(tuple(sorted((a, b))))
    for a, b in base_connections:
        add(a, b)
    for a, b in (extra_edges or []):
        add(a, b)
    return sorted(edges)


def build_edge_index_torch(edges_undirected):
    """Convierte lista de aristas (undirected) a edge_index de PyTorch Geometric."""
    import torch
    directed = []
    for i, j in edges_undirected:
        directed.append((i, j))
        directed.append((j, i))
    return torch.tensor(directed, dtype=torch.long).t().contiguous()


def compute_palm_normal(coords_normalized):
    """
    Vector normal de la palma con wrist(0), index_MCP(5), pinky_MCP(17).
    Producto cruz v1 × v2. Z positivo = palma hacia camara.
    """
    c = coords_normalized.astype(np.float32)
    v1 = c[5] - c[0]
    v2 = c[17] - c[0]
    normal = np.cross(v1, v2)
    norm_len = np.linalg.norm(normal)
    if norm_len > 1e-9:
        normal = normal / norm_len
    return float(normal[0]), float(normal[1]), float(normal[2])


def compute_signed_volumes(coords_normalized):
    """
    Volumenes de tetraedros con signo (quiralidad).
    Base: wrist(0), index_MCP(5), pinky_MCP(17).
    Apex: cada punta de dedo (4,8,12,16,20).
    V = (1/6) * w · (u × v). Cambia de signo al voltear la mano.
    """
    c = coords_normalized.astype(np.float32)
    u = c[5] - c[0]
    v = c[17] - c[0]
    cross_uv = np.cross(u, v)
    volumes = []
    for tip_idx in [4, 8, 12, 16, 20]:
        w = c[tip_idx] - c[0]
        vol = np.dot(w, cross_uv) / 6.0
        volumes.append(float(vol))
    return np.array(volumes, dtype=np.float32)


def build_node_features(coords_normalized, edges_undirected, base_connections=None):
    """
    Construye node_features de dimension 18 por nodo:
      3 coords + 3 bone_vec + 1 dist_center + 1 polar_r + 1 azimuth
      + 1 elevation + 3 palm_normal + 5 signed_volumes
    """
    c = coords_normalized.astype(np.float32)           # (21,3)
    parent_map = {}
    for parent, child in (base_connections or edges_undirected):
        if child not in parent_map:
            parent_map[child] = parent

    bone_vecs = np.zeros_like(c)
    for idx in range(c.shape[0]):
        p = parent_map.get(idx)
        if p is not None:
            bone_vecs[idx] = c[idx] - c[p]

    dist_center = np.linalg.norm(c, axis=1, keepdims=True)   # (21,1)

    x, y, z = c[:, 0], c[:, 1], c[:, 2]
    eps = np.finfo(np.float32).eps
    polar_r  = np.sqrt(np.maximum(x**2 + y**2, eps)).reshape(-1, 1)
    azimuth  = np.arctan2(y, x).reshape(-1, 1)
    elevation = np.arctan2(z, np.sqrt(np.maximum(x**2 + y**2, eps))).reshape(-1, 1)

    palm_nx, palm_ny, palm_nz = compute_palm_normal(coords_normalized)
    signed_vols = compute_signed_volumes(coords_normalized)
    palm_normal_features = np.tile(
        np.array([[palm_nx, palm_ny, palm_nz]], dtype=np.float32), (c.shape[0], 1))
    signed_vol_features = np.tile(
        signed_vols.reshape(1, -1), (c.shape[0], 1))

    node_features = np.concatenate(
        [c, bone_vecs, dist_center, polar_r, azimuth, elevation,
         palm_normal_features, signed_vol_features], axis=1)  # (21,18)
    return node_features.astype(np.float32)


def build_edge_features(coords_normalized, edges_undirected):
    """
    Distancia euclidiana por arista (undirected → duplicada para directed).
    Retorna numpy array (2*num_edges, 1) – igual que el notebook.
    """
    c = coords_normalized
    edge_feats = []
    for i, j in edges_undirected:
        dist = float(np.linalg.norm(c[i] - c[j]))
        edge_feats.append([dist])
    # Duplicar para aristas dirigidas (i→j y j→i)
    return np.array(edge_feats + edge_feats, dtype=np.float32)  # (2*E, 1)


def build_tabular_features(coords_normalized):
    """
    Construye el vector de features tabulares enriquecidas:
      63 coords + distancias anatomicas + distancias entre puntas +
      distancias muñeca-puntas + angulos articulares.
    Identico a construir_features_geometricas() del notebook.
    """
    c = coords_normalized  # (21,3)

    bone_pairs     = list(HAND_CONNECTIONS)
    tip_pairs      = list(combinations(TIP_INDICES, 2))
    wrist_tip_pairs = [(0, tip) for tip in TIP_INDICES]

    finger_angle_triples = []
    for chain in FINGER_CHAINS.values():
        for a, b, cc in zip(chain[:-2], chain[1:-1], chain[2:]):
            finger_angle_triples.append((a, b, cc))

    def dist(i, j):
        return float(np.linalg.norm(c[i] - c[j]))

    def angle(a, b, cc):
        v1 = c[a] - c[b]
        v2 = cc - c[b] if isinstance(cc, np.ndarray) else c[cc] - c[b]
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom == 0:
            return 0.0
        return float(np.arccos(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)) / np.pi)

    features = []
    features.extend(c.reshape(-1).tolist())                              # 63 coords
    features.extend(dist(i, j) for i, j in bone_pairs)                  # 21 dist. anatomicas
    features.extend(dist(i, j) for i, j in tip_pairs)                   # 10 dist. puntas
    features.extend(dist(i, j) for i, j in wrist_tip_pairs)             # 5  dist. muñeca-puntas
    features.extend(angle(a, b, cc) for a, b, cc in finger_angle_triples)  # angulos dedos
    features.extend(angle(a, b, cc) for _, a, b, cc in PALM_ANGLE_TRIPLES) # angulos palma

    palm_nx, palm_ny, palm_nz = compute_palm_normal(c)
    vol_features = compute_signed_volumes(c)
    features.extend([palm_nx, palm_ny, palm_nz])          # 3 palm normal XYZ
    for v in vol_features:
        features.append(float(v))                           # 5 signed tetra volumes

    return np.array(features, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# ARQUITECTURAS DE RED (deben coincidir con las del entrenamiento)
# ──────────────────────────────────────────────────────────────────────────────

def _import_torch_geometric():
    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.data import Data
        from torch_geometric.nn import GCNConv, GATConv, GraphNorm
        from torch_geometric.utils import to_dense_batch
        return torch, F, Data, GCNConv, GATConv, GraphNorm, to_dense_batch
    except ImportError:
        return None


class HandGCNv2:
    """Wrapper de la arquitectura HandGCNv2 del notebook."""

    @staticmethod
    def build(torch, F, GCNConv, GraphNorm, to_dense_batch,
              in_channels, hidden_dims, num_layers, num_classes, num_nodes, dropout):
        import torch.nn as nn

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims] * num_layers

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj  = nn.Linear(in_channels, hidden_dims[0])
                self.convs = nn.ModuleList()
                self.norms = nn.ModuleList()
                self.num_nodes = num_nodes
                self.dropout   = dropout

                for idx in range(num_layers):
                    in_d  = hidden_dims[idx - 1] if idx > 0 else hidden_dims[0]
                    out_d = hidden_dims[idx]
                    self.convs.append(GCNConv(in_d, out_d))
                    self.norms.append(GraphNorm(out_d))

                self.classifier = nn.Sequential(
                    nn.Linear(num_nodes * hidden_dims[-1], 256),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, num_classes),
                )

            def forward(self, data):
                x, edge_index, batch = data.x, data.edge_index, data.batch
                x = F.relu(self.proj(x))
                for conv, norm in zip(self.convs, self.norms):
                    h = conv(x, edge_index)
                    h = norm(h, batch)
                    h = F.relu(h)
                    h = F.dropout(h, p=self.dropout, training=self.training)
                    x = x + h if x.shape == h.shape else h
                x_dense, _ = to_dense_batch(x, batch, max_num_nodes=self.num_nodes)
                return self.classifier(x_dense.reshape(x_dense.size(0), -1))

        return _Model()


class HandGAT:
    """Wrapper de la arquitectura HandGAT del notebook."""

    @staticmethod
    def build(torch, F, GATConv, GraphNorm, to_dense_batch,
              in_channels, hidden_channels, num_classes, num_nodes, heads=4, dropout=0.3):
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_nodes = num_nodes
                self.dropout   = dropout
                self.conv1 = GATConv(in_channels, hidden_channels,
                                     heads=heads, dropout=dropout, edge_dim=1)
                self.norm1 = GraphNorm(hidden_channels * heads)
                self.conv2 = GATConv(hidden_channels * heads, hidden_channels * 2,
                                     heads=1, concat=False, dropout=dropout, edge_dim=1)
                self.norm2 = GraphNorm(hidden_channels * 2)
                self.classifier = nn.Sequential(
                    nn.Linear(num_nodes * hidden_channels * 2, 128),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(128, num_classes),
                )

            def forward(self, data):
                x, edge_index, edge_attr, batch = \
                    data.x, data.edge_index, data.edge_attr, data.batch
                x = F.elu(self.conv1(x, edge_index, edge_attr))
                x = self.norm1(x, batch)
                x = F.elu(self.conv2(x, edge_index, edge_attr))
                x = self.norm2(x, batch)
                x_dense, _ = to_dense_batch(x, batch, max_num_nodes=self.num_nodes)
                return self.classifier(x_dense.reshape(x_dense.size(0), -1))

        return _Model()


# ──────────────────────────────────────────────────────────────────────────────
# CARGA DE MODELOS
# ──────────────────────────────────────────────────────────────────────────────

def load_model_package(model_key: str, artifacts_dir: Path):
    """
    Carga el modelo indicado y devuelve un dict con todo lo necesario
    para hacer predicciones.

    model_key: 'classic' | 'mlp' | 'gcn' | 'gat' | 'gat_robusto' | 'auto'
    """
    artifacts_dir = Path(artifacts_dir)
    meta_path     = artifacts_dir / "metadata_inferencia.json"
    encoder_path  = artifacts_dir / "label_encoder.joblib"

    if not meta_path.exists():
        raise FileNotFoundError(f"No se encontro metadata_inferencia.json en {artifacts_dir}")
    if not encoder_path.exists():
        raise FileNotFoundError(f"No se encontro label_encoder.joblib en {artifacts_dir}")

    with meta_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    encoder     = joblib.load(encoder_path)
    class_names = np.array(metadata.get("class_names", list(encoder.classes_)))

    # Determinar key real cuando es 'auto'
    if model_key == "auto":
        best = metadata.get("best_model_by_macro_f1", "")
        if best and "gcn" in best.lower():
            model_key = "gcn"
        elif best and ("gat" in best.lower()):
            model_key = "gat"
        elif best and "mlp" in best.lower():
            model_key = "mlp"
        else:
            model_key = "classic"
        print(f"[auto] Mejor modelo segun metadata: '{best}' → usando key '{model_key}'")

    pkg = dict(
        key         = model_key,
        class_names = class_names,
        metadata    = metadata,
        encoder     = encoder,
    )

    # ── Modelos clasicos (sklearn) ────────────────────────────────────────────
    if model_key == "classic":
        path = artifacts_dir / "mejor_modelo_clasico.joblib"
        if not path.exists():
            path = artifacts_dir / "modelos_clasicos.joblib"
        pkg["model"]      = joblib.load(path)
        pkg["model_type"] = "classic"
        pkg["input"]      = "tabular"
        print(f"Modelo clasico cargado desde {path}")
        return pkg

    # ── MLP (Keras) ───────────────────────────────────────────────────────────
    if model_key == "mlp":
        import tensorflow as tf
        path = artifacts_dir / "modelo_mlp.keras"
        pkg["model"]      = tf.keras.models.load_model(str(path))
        pkg["model_type"] = "mlp"
        pkg["input"]      = "tabular"
        print(f"MLP cargado desde {path}")
        return pkg

    # ── GCN / GAT (PyTorch Geometric) ────────────────────────────────────────
    pyg = _import_torch_geometric()
    if pyg is None:
        raise ImportError("PyTorch / PyTorch Geometric no están disponibles.")
    torch, F, Data, GCNConv, GATConv, GraphNorm, to_dense_batch = pyg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_key == "gcn":
        path = artifacts_dir / "modelo_gcn_pyg.pt"
        ckpt = torch.load(path, map_location=device, weights_only=False)
        hidden = ckpt.get("hidden_channels", 32)
        layers = 3  # siempre 3 en el notebook (GCN_LAYERS=3)
        model  = HandGCNv2.build(
            torch, F, GCNConv, GraphNorm, to_dense_batch,
            in_channels  = int(ckpt.get("in_channels", 18)),
            hidden_dims  = hidden,
            num_layers   = layers,
            num_classes  = int(ckpt.get("num_classes", len(class_names))),
            num_nodes    = 21,
            dropout      = float(ckpt.get("dropout", 0.3)),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval().to(device)

        gcn_edges   = build_adjacency(21, HAND_CONNECTIONS, EXTRA_EDGES)
        edge_index  = build_edge_index_torch(gcn_edges).to(device)

        pkg["model"]      = model
        pkg["model_type"] = "gcn"
        pkg["input"]      = "graph"
        pkg["device"]     = device
        pkg["edge_index"] = edge_index
        pkg["gcn_edges"]  = gcn_edges
        pkg["torch"]      = torch
        pkg["F"]          = F
        pkg["Data"]       = Data
        pkg["to_dense"]   = to_dense_batch
        pkg["needs_edge_attr"] = False
        print(f"HandGCNv2 cargado desde {path} en {device}")
        return pkg

    if model_key in ("gat", "gat_robusto"):
        fname = "modelo_gat_robusto.pt" if model_key == "gat_robusto" \
                else "modelo_gat_mejorado.pt"
        path  = artifacts_dir / fname
        ckpt  = torch.load(path, map_location=device, weights_only=False)
        model = HandGAT.build(
            torch, F, GATConv, GraphNorm, to_dense_batch,
            in_channels      = int(ckpt.get("in_channels", 18)),
            hidden_channels  = int(ckpt.get("hidden_channels", 32)),
            num_classes      = int(ckpt.get("num_classes", len(class_names))),
            num_nodes        = 21,
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval().to(device)

        gcn_edges  = build_adjacency(21, HAND_CONNECTIONS, EXTRA_EDGES)
        edge_index = build_edge_index_torch(gcn_edges).to(device)

        pkg["model"]      = model
        pkg["model_type"] = "gat"
        pkg["input"]      = "graph"
        pkg["device"]     = device
        pkg["edge_index"] = edge_index
        pkg["gcn_edges"]  = gcn_edges
        pkg["torch"]      = torch
        pkg["F"]          = F
        pkg["Data"]       = Data
        pkg["to_dense"]   = to_dense_batch
        pkg["needs_edge_attr"] = True
        print(f"HandGAT cargado desde {path} en {device}")
        return pkg

    raise ValueError(f"model_key '{model_key}' no reconocido. "
                     "Opciones: classic | mlp | gcn | gat | gat_robusto | auto")


# ──────────────────────────────────────────────────────────────────────────────
# PREDICCION
# ──────────────────────────────────────────────────────────────────────────────

def softmax_numpy(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return (e / e.sum()).astype(np.float32)


def predict(pkg: dict, coords_norm: np.ndarray) -> tuple[str, float, np.ndarray]:
    """
    Hace una prediccion a partir de landmarks ya normalizados (21,3).
    Retorna (label, confidence, probabilities).
    """
    class_names = pkg["class_names"]
    model_type  = pkg["model_type"]

    # ── Tabular (classic / mlp) ───────────────────────────────────────────────
    if pkg["input"] == "tabular":
        tabular = build_tabular_features(coords_norm).reshape(1, -1)

        if model_type == "mlp":
            probs = pkg["model"].predict(tabular, verbose=0)[0]
        else:
            if hasattr(pkg["model"], "predict_proba"):
                probs = pkg["model"].predict_proba(tabular)[0]
            elif hasattr(pkg["model"], "decision_function"):
                scores = pkg["model"].decision_function(tabular)
                probs  = softmax_numpy(scores[0] if scores.ndim > 1 else scores)
            else:
                idx   = int(pkg["model"].predict(tabular)[0])
                probs = np.zeros(len(class_names), dtype=np.float32)
                probs[idx] = 1.0

    # ── Grafo (gcn / gat) ─────────────────────────────────────────────────────
    else:
        torch      = pkg["torch"]
        F          = pkg["F"]
        Data       = pkg["Data"]
        device     = pkg["device"]
        edge_index = pkg["edge_index"]
        gcn_edges  = pkg["gcn_edges"]
        model      = pkg["model"]

        node_feats = build_node_features(coords_norm, gcn_edges, HAND_CONNECTIONS)
        x_tensor   = torch.tensor(node_feats, dtype=torch.float32, device=device)
        batch      = torch.zeros(21, dtype=torch.long, device=device)

        data_kwargs = dict(
            x          = x_tensor,
            edge_index = edge_index,
            batch      = batch,
        )

        if pkg["needs_edge_attr"]:
            ef = build_edge_features(coords_norm, gcn_edges)
            data_kwargs["edge_attr"] = torch.tensor(ef, dtype=torch.float32, device=device)

        data = Data(**data_kwargs)

        with torch.no_grad():
            logits = model(data)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    idx        = int(np.argmax(probs))
    label      = str(class_names[idx])
    confidence = float(probs[idx])
    return label, confidence, probs


# ──────────────────────────────────────────────────────────────────────────────
# BUFFER DE ESTABILIZACION TEMPORAL
# ──────────────────────────────────────────────────────────────────────────────

class PredictionBuffer:
    """
    Suaviza la salida del clasificador frame a frame.

    Parametros:
      maxlen         – ventana temporal (numero de frames).
      min_votes      – votos minimos de la misma clase para considerarla estable.
      min_confidence – umbral de confianza minima para agregar al buffer.
    """

    def __init__(self, maxlen: int = 10, min_votes: int = 6,
                 min_confidence: float = 0.50):
        self.buffer           = deque(maxlen=maxlen)
        self.min_votes        = min_votes
        self.min_confidence   = min_confidence
        self.last_stable      = None
        self.stable_conf_avg  = 0.0

    def update(self, label: str | None, confidence: float) -> str | None:
        if label is None or confidence < self.min_confidence:
            return self.last_stable

        self.buffer.append((label, confidence))
        counts = Counter(l for l, _ in self.buffer)
        top_label, votes = counts.most_common(1)[0]

        if votes >= self.min_votes:
            self.last_stable     = top_label
            self.stable_conf_avg = float(np.mean(
                [c for l, c in self.buffer if l == top_label]
            ))

        return self.last_stable

    def reset(self):
        self.buffer.clear()
        self.last_stable     = None
        self.stable_conf_avg = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# DETECTOR MEDIAPIPE
# ──────────────────────────────────────────────────────────────────────────────

def create_mp_detector(model_path: str,
                       min_detection_confidence: float = 0.5,
                       min_presence_confidence:  float = 0.5,
                       min_tracking_confidence:  float = 0.5):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontro el modelo MediaPipe en: {model_path}\n"
            "Descargalo con:\n"
            "  wget https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )
    opts = mp_vision.HandLandmarkerOptions(
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path)),
        num_hands    = 1,
        min_hand_detection_confidence = min_detection_confidence,
        min_hand_presence_confidence  = min_presence_confidence,
        min_tracking_confidence       = min_tracking_confidence,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


def extract_from_frame(frame_bgr: np.ndarray, detector) -> dict | None:
    """
    Detecta landmarks en el frame y devuelve un dict con coords normalizadas.
    Retorna None si no se detecta mano.
    """
    from mediapipe.tasks.python import vision as mp_vision

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)

    if not result.hand_landmarks:
        return None

    lms        = result.hand_landmarks[0]
    coords_raw = _to_numpy_coords(lms)
    coords_norm = normalize_landmarks(lms)

    return {
        "coords_norm"      : coords_norm,       # (21,3) normalizados
        "coords_raw"       : coords_raw,         # (21,3) en espacio imagen [0,1]
        "detection_result" : result,
    }


# ──────────────────────────────────────────────────────────────────────────────
# VISUALIZACION
# ──────────────────────────────────────────────────────────────────────────────

# Paleta de colores (BGR)
_CLR_LINE      = (70, 130, 180)
_CLR_NODE      = (0, 140, 255)
_CLR_TEXT_W    = (255, 255, 255)
_CLR_TEXT_G    = (0, 220, 120)
_CLR_TEXT_Y    = (0, 220, 220)
_CLR_BG        = (20, 20, 20)


def draw_landmarks(frame_bgr: np.ndarray, detection_result,
                   connections=HAND_CONNECTIONS) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    if not detection_result or not detection_result.hand_landmarks:
        return out

    lms = detection_result.hand_landmarks[0]

    for i, j in connections:
        p1 = (int(lms[i].x * w), int(lms[i].y * h))
        p2 = (int(lms[j].x * w), int(lms[j].y * h))
        cv2.line(out, p1, p2, _CLR_LINE, 2, cv2.LINE_AA)

    for idx, lm in enumerate(lms):
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(out, (cx, cy), 5, _CLR_NODE, -1)
        cv2.putText(out, str(idx), (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, _CLR_TEXT_W, 1, cv2.LINE_AA)
    return out


def draw_overlay(frame_bgr: np.ndarray,
                 instant_label: str | None,
                 instant_conf:  float,
                 stable_label:  str | None,
                 stable_conf:   float,
                 model_name:    str,
                 fps:           float,
                 status:        str = "") -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    # Panel superior
    cv2.rectangle(out, (0, 0), (w, 100), _CLR_BG, -1)

    # Titulo
    cv2.putText(out, "LSC Real-Time Interpreter", (14, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, _CLR_TEXT_W, 2, cv2.LINE_AA)

    # Prediccion instantanea
    inst_txt = (f"Instant: {instant_label}  ({instant_conf:.0%})"
                if instant_label else "Instant: sin mano")
    cv2.putText(out, inst_txt, (14, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200, 200, 200), 1, cv2.LINE_AA)

    # Prediccion estable
    stable_txt = (f"Estable: {stable_label}  ({stable_conf:.0%})"
                  if stable_label else "Estable: esperando...")
    cv2.putText(out, stable_txt, (14, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, _CLR_TEXT_G, 2, cv2.LINE_AA)

    # Letra grande (esquina derecha)
    if stable_label:
        txt_size = cv2.getTextSize(stable_label, cv2.FONT_HERSHEY_SIMPLEX, 3.5, 5)[0]
        cv2.putText(out, stable_label, (w - txt_size[0] - 16, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.5, _CLR_TEXT_Y, 5, cv2.LINE_AA)

    # FPS + modelo (esquina inferior)
    info = f"{model_name}  |  {fps:.1f} FPS"
    if status:
        info += f"  |  {status}"
    cv2.putText(out, info, (14, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)

    # Instruccion
    cv2.putText(out, "Presiona 'q' para salir  |  'r' para reiniciar buffer",
                (14, h - 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# BUCLE PRINCIPAL DE CAMARA
# ──────────────────────────────────────────────────────────────────────────────

def run_camera(pkg: dict,
               mp_model_path: str,
               camera_index:  int   = 0,
               buffer_len:    int   = 10,
               min_votes:     int   = 6,
               min_conf:      float = 0.50,
               show_debug:    bool  = False):
    """
    Bucle de inferencia en tiempo real con camara web.

    Teclas:
      q – salir
      r – reiniciar buffer de estabilizacion
      d – alternar debug (mostrar confianzas)
    """
    detector = create_mp_detector(mp_model_path)
    buf      = PredictionBuffer(maxlen=buffer_len,
                                min_votes=min_votes,
                                min_confidence=min_conf)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la camara (indice {camera_index}).")

    # Intentar forzar 30 fps
    cap.set(cv2.CAP_PROP_FPS, 30)

    model_label = pkg["key"].upper()
    prev_time   = time.perf_counter()
    fps         = 0.0
    debug       = show_debug

    print("=" * 60)
    print(f"  Modelo : {model_label}  |  clases: {len(pkg['class_names'])}")
    print(f"  Buffer : ventana={buffer_len}, votos={min_votes}, conf>={min_conf:.0%}")
    print("  Teclas : q=salir  r=reiniciar buffer  d=debug")
    print("=" * 60)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error: no se pudo leer el frame.")
                break

            frame = cv2.flip(frame, 1)   # espejo horizontal

            # ── Deteccion y prediccion ─────────────────────────────────────────
            entry = extract_from_frame(frame, detector)
            instant_label, instant_conf = None, 0.0
            stable_conf = buf.stable_conf_avg

            if entry is not None:
                try:
                    instant_label, instant_conf, probs = predict(pkg, entry["coords_norm"])
                    buf.update(instant_label, instant_conf)
                    stable_conf = buf.stable_conf_avg

                    if debug:
                        top5 = np.argsort(probs)[::-1][:5]
                        status = "  ".join(
                            f"{pkg['class_names'][i]}:{probs[i]:.2f}"
                            for i in top5
                        )
                    else:
                        status = ""

                    frame = draw_landmarks(frame, entry["detection_result"])

                except Exception as exc:
                    status = f"Error prediccion: {exc}"
            else:
                status = ""

            # ── FPS ───────────────────────────────────────────────────────────
            now      = time.perf_counter()
            fps      = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            # ── Overlay ───────────────────────────────────────────────────────
            frame = draw_overlay(
                frame,
                instant_label, instant_conf,
                buf.last_stable, stable_conf,
                model_label, fps, status,
            )

            cv2.imshow("LSC Interpreter", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                buf.reset()
                print("Buffer reiniciado.")
            elif key == ord("d"):
                debug = not debug
                print(f"Debug {'ON' if debug else 'OFF'}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camara liberada.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="LSC Real-Time Interpreter – inferencia desde camara.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", "-m",
        default="auto",
        choices=["auto", "classic", "mlp", "gcn", "gat", "gat_robusto"],
        help="Modelo a usar (default: auto → mejor segun metadata).",
    )
    parser.add_argument(
        "--artifacts", "-a",
        default="model_artifacts_kaggle",
        help="Directorio con los artefactos guardados (default: model_artifacts_kaggle).",
    )
    parser.add_argument(
        "--mediapipe", "-mp",
        default="hand_landmarker.task",
        help="Ruta al archivo hand_landmarker.task de MediaPipe.",
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="Indice de la camara (default: 0).",
    )
    parser.add_argument(
        "--buffer-len",
        type=int,
        default=10,
        help="Longitud del buffer temporal (default: 10 frames).",
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=6,
        help="Votos minimos para que una letra se considere estable (default: 6).",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.50,
        help="Confianza minima para agregar al buffer (default: 0.50).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Mostrar top-5 probabilidades en pantalla.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Listar los archivos de modelos encontrados y salir.",
    )
    return parser.parse_args()


def list_models(artifacts_dir: Path):
    print(f"\nArtefactos en: {artifacts_dir.resolve()}\n")
    files = {
        "classic"     : "mejor_modelo_clasico.joblib",
        "mlp"         : "modelo_mlp.keras",
        "gcn"         : "modelo_gcn_pyg.pt",
        "gat"         : "modelo_gat_mejorado.pt",
        "gat_robusto" : "modelo_gat_robusto.pt",
    }
    for key, fname in files.items():
        path   = artifacts_dir / fname
        status = "✓ encontrado" if path.exists() else "✗ NO encontrado"
        print(f"  [{key:12s}]  {fname:<35s}  {status}")
    print()


def main():
    args         = parse_args()
    artifacts    = Path(args.artifacts)

    if args.list_models:
        list_models(artifacts)
        sys.exit(0)

    print(f"\nCargando modelo '{args.model}' desde '{artifacts}'...")
    pkg = load_model_package(args.model, artifacts)

    run_camera(
        pkg            = pkg,
        mp_model_path  = args.mediapipe,
        camera_index   = args.camera,
        buffer_len     = args.buffer_len,
        min_votes      = args.min_votes,
        min_conf       = args.min_conf,
        show_debug     = args.debug,
    )


if __name__ == "__main__":
    main()