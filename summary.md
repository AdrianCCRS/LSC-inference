# Proyecto: Intérprete de Lengua de Señas Colombiana (LSC) con MediaPipe Hands

## Qué se está haciendo

Este proyecto implementa un **intérprete de Lengua de Señas Colombiana (LSC)** que reconoce 21 letras del alfabeto manual estático (A, B, C, D, E, F, I, K, L, M, N, O, P, Q, R, T, U, V, W, X, Y) a partir de imágenes de la mano. La entrada visual se procesa con **MediaPipe Hands** para extraer 21 landmarks 3D `(x, y, z)` de la mano, y sobre esos landmarks se entrena un clasificador que predice la letra correspondiente. El pipeline completo incluye extracción de landmarks, normalización (centrado en muñeca, escala, alineación rotacional), extracción de features geométricas, entrenamiento, inferencia en tiempo real con cámara web, y un benchmark de estabilidad temporal.

### Pipeline completo

1. **Detección**: MediaPipe Hands extrae 21 landmarks `(x, y, z)` de la mano
2. **Normalización**: centrar en muñeca (landmark 0), escalar por distancia máxima, rotar para alinear landmark 9 al eje +X
3. **Feature engineering**:
   - Vector plano: 63 coordenadas concatenadas
   - Features tabulares enriquecidas: 63 coords + 21 distancias anatómicas + 10 distancias entre puntas + 5 distancias muñeca-puntas + ángulos articulares (15 de dedos + 4 de palma) = **118 features**
   - Representación de grafo: `(21, 7)` node features `[x, y, z, bone_vec_x, bone_vec_y, bone_vec_z, dist_center]`
4. **Entrenamiento**: split 80/20 estratificado (3057 train, 765 test), con balanceo de clases (oversampling + undersampling) a 182 muestras/clase
5. **Inferencia**: cámara en tiempo real con buffer temporal de estabilización (ventana de 10 frames, 6 votos mínimos)
6. **Benchmark**: métricas de estabilidad temporal (confianza media, entropía, flip rate, stability ratio, top-1 agreement)

---

## Modelos comparados

Todos los modelos comparten el mismo split de datos y se evalúan con las mismas métricas (macro-F1, weighted-F1, accuracy, precision, recall).

### 1. Modelos clásicos de ML + features geométricas (representación tabular 118D)

| Modelo | Macro-F1 | Accuracy | Observaciones |
|--------|----------|----------|---------------|
| **Random Forest** (300 árboles) | **0.986** | **0.986** | Mejor modelo global |
| SVM RBF (C=1.0) | 0.967 | 0.967 | Escalado estándar requerido |
| KNN (k=5) | 0.943 | 0.944 | Escalado + distancia euclidiana |

- **Entrada**: vector tabular enriquecido con distancias y ángulos explícitos (118 features)
- **Fortaleza**: Random Forest captura relaciones no lineales con features geométricas explícitas; no requiere GPU

### 2. MLP (red neuronal densa) + features geométricas (118D)

| Modelo | Macro-F1 | Accuracy | Observaciones |
|--------|----------|----------|---------------|
| MLP (128→64→21) | 0.959 | 0.959 | BatchNorm, Dropout 0.3/0.2, Adam |

- Arquitectura: `Input(118) → Dense(128) → BatchNorm → Dropout(0.3) → Dense(64) → Dropout(0.2) → Dense(21, softmax)`
- 25,365 parámetros, early stopping con paciencia 10
- **Entrada**: features tabulares enriquecidas
- **Fortaleza**: aprende representaciones no lineales sobre features geométricas

### 3. GCN (Graph Convolutional Network) — PyTorch Geometric

| Modelo | Macro-F1 | Accuracy | Observaciones |
|--------|----------|----------|---------------|
| HandGCNv2 | 0.983 | 0.983 | 7-dim node features, 3 capas GCN |

- Arquitectura: proyección lineal + 3 × (GCNConv + GraphNorm + ReLU + dropout 0.3) con skip connections → to_dense_batch → Flatten → `Linear(21×32 → 256) → ReLU → Dropout → Linear(256 → 128) → ReLU → Linear(128 → 21)`
- Grafo anatómico: 21 nodos, aristas base de MediaPipe + aristas extra (conexiones entre puntas, MCP vecinos, pulgar-índice)
- 150 épocas con early stopping (paciencia 15), label smoothing 0.1
- **Entrada**: grafo `(21, 7)`, donde cada nodo contiene `[x, y, z, bone_vec, dist_center]`
- **Fortaleza**: conserva topología anatómica explícita; agrega información entre landmarks vecinos

### 4. HandGAT (Graph Attention Network)

| Modelo | Macro-F1 | Accuracy | Observaciones |
|--------|----------|----------|---------------|
| GAT Mejorada | 0.980 | 0.980 | 2 capas GAT, 4 heads, edge_dim=1 |

- Arquitectura: `GATConv(in→32×4, edge_dim=1) → GraphNorm → GATConv(128→64, heads=1, edge_dim=1) → GraphNorm → Linear(21×64→128) → ReLU → Dropout → Linear(128→21)`
- Pesos de clase balanceados, label smoothing 0.1, scheduler ReduceLROnPlateau
- **Entrada**: mismo grafo que la GCN, pero el edge_attr son distancias euclidianas entre nodos conectados
- **Fortaleza**: la atención permite que cada nodo pese diferencialmente la información de sus vecinos

### 5. HandGAT Robusto (con aumentación de datos)

| Modelo | Macro-F1 | Accuracy | Observaciones |
|--------|----------|----------|---------------|
| HandGAT Robusto (Augment) | 0.974 | 0.974 | Aumentación en entrenamiento |

- Misma arquitectura que HandGAT
- Aumentación: ruido gaussiano (σ=0.005), rotación (±15°), escala (0.9–1.1), dropout de nodos (p=0.08)
- **Objetivo**: robustez a variaciones de cámara y ruido de MediaPipe
- Entrenado con datos aumentados, evaluado sin aumentación

---

## Resultados del benchmark de estabilidad temporal (cámara real)

La estabilidad en tiempo real se evaluó manteniendo la letra "T" fija ante la cámara durante ~5 segundos. Métricas no capturadas por accuracy/F1 offline:

| Modelo | Conf μ | Entropía | Flip/s | Estabilidad | Top-1 Agree |
|--------|--------|----------|--------|-------------|-------------|
| GCN | 0.682 | 0.430 | 3.24 | 0.910 | 0.910 |
| GAT | 0.556 | 0.545 | 3.24 | 0.923 | 0.923 |
| GAT Robusto | 0.477 | 0.619 | 9.38 | 0.644 | 0.678 |
| Classic (RF) | 0.431 | 0.545 | 8.63 | 0.490 | 0.610 |

**Observación clave**: aunque Random Forest lidera en métricas offline (macro-F1), la **GCN es superior en estabilidad temporal**: mayor confianza media, menor flip rate, mayor stability ratio. La GAT también mantiene alta estabilidad. El modelo clásico sufre más flickering (cambios frecuentes de letra entre frames).

---

## Datasets utilizados

1. **Roboflow** (`LSC.v1i.multiclass`): dataset con split train/valid/test, formato YOLO multiclase, 2906 imágenes
2. **Kaggle** (`dataset-lsc-modelo`): 21 carpetas por letra, 200 imágenes/clase; después de filtrar con MediaPipe quedan 2548 muestras válidas (1652 descartadas por no detectar mano)

---

## Propuesta de alternativas y modelos a explorar

A continuación se proponen enfoques que podrían complementar o superar los resultados actuales, organizados por categoría.

### A. Modelos basados en imágenes (end-to-end, sin MediaPipe)

Estos modelos procesan la imagen directamente, eliminando la dependencia de MediaPipe como paso intermedio (y el 40% de imágenes descartadas por no detección).

#### A.1 YOLO (detección + clasificación conjunta)

- **YOLOv8-cls o YOLOv8-pose**: entrenar extremo a extremo para clasificar la letra a partir de la imagen, posiblemente con keypoints de mano como tarea auxiliar
- **Ventaja**: un solo modelo reemplaza MediaPipe + clasificador; inferencia más rápida (optimizado para CPU/GPU/NPU); no descarta imágenes sin mano detectable por MediaPipe
- **Desventaja**: requiere muchas más imágenes etiquetadas; el dataset actual (200 img/clase) puede ser insuficiente
- **Estrategia**: usar el dataset actual con strong data augmentation (rotación, escala, traslación, brillo, contraste, recorte aleatorio) o pre-entrenar en un dataset grande de manos (HaGRID, FreiHAND) y hacer fine-tuning

#### A.2 CNNs clásicas (transfer learning)

- **Arquitecturas**: ResNet-18/50, MobileNetV3, EfficientNet-B0, ConvNeXt-Tiny
- **Estrategia**: cargar pesos pre-entrenados en ImageNet, reemplazar la cabeza clasificadora por 21 clases LSC, fine-tuning con learning rate diferencial
- **Ventaja**: se benefician de features visuales genéricas; MobileNet/EfficientNet son eficientes para despliegue en móviles
- **Desventaja**: ImageNet no contiene manos; puede ser mejor pre-entrenar en datasets de gestos/ manos (HaGRID: ~700K imágenes de gestos, 18 clases)
- **Alternativa**: usar **MediaPipe como preprocesador de recorte** (crop de la región de la mano detectada) + CNN sobre la región recortada, combinando lo mejor de ambos enfoques

#### A.3 Vision Transformers (ViT)

- **Arquitecturas**: ViT-B/16, Swin Transformer, DeiT (Data-efficient Image Transformer)
- **Ventaja**: DeiT fue diseñado para funcionar con datasets pequeños (~5000 imágenes); los transformers capturan relaciones globales que pueden ser útiles para distinguir configuraciones de dedos
- **Estrategia**: pre-entrenar en ImageNet-21k o usar modelos destilados; evaluar si la atención multi-cabeza captura relaciones entre regiones de la mano sin necesidad de grafo explícito

### B. Mejoras sobre el enfoque actual de landmarks

Si se mantiene MediaPipe como extractor de landmarks, hay espacio para mejorar el modelado.

#### B.1 Mejores arquitecturas de GNN

- **GraphSAGE**: sampling de vecinos para escalar a grafos más grandes (menos relevante para 21 nodos, pero puede generalizar mejor)
- **GIN (Graph Isomorphism Network)**: teóricamente más expresiva que GCN/GAT para distinguir estructuras de grafo
- **EdgeConv (DGCNN)**: opera en el espacio de aristas, capturando relaciones geométricas entre pares de puntos; particularmente relevante para landmarks de mano porque modela explícitamente la geometría de las conexiones
- **Transformer sobre grafo (Graphormer, GPS)**: combina atención global con paso de mensajes local; puede capturar dependencias de largo alcance entre dedos no adyacentes

#### B.2 Mejores features para los modelos tabulares

- Agregar **features de curvatura** de los dedos
- Incorporar **momentos de Hu** o descriptores de forma sobre la nube de puntos
- Añadir **features de velocidad/aceleración** entre frames consecutivos (aprovechando la naturaleza temporal del video)
- Usar **PCA** sobre las 63 coordenadas para reducir dimensionalidad y eliminar correlaciones

#### B.3 Enfoques secuenciales/temporales

El proyecto actual trata cada frame como independiente. Modelar la secuencia temporal podría mejorar la estabilidad:

- **LSTM / GRU sobre secuencias de vectores de landmarks**: aprende patrones de movimiento entre frames
- **TCN (Temporal Convolutional Network)**: convoluciones 1D sobre secuencias de features; más rápido de entrenar que LSTM
- **Transformer temporal**: atención sobre la dimensión temporal para modelar dependencias de largo plazo
- **Two-stream**: combinar features espaciales del frame actual + features de movimiento (optical flow o diferencia entre frames)

### C. Modelos multimodales

Combinar la imagen cruda con los landmarks de MediaPipe para obtener lo mejor de ambos mundos:

- **Late fusion**: CNN sobre imagen + GNN sobre landmarks, concatenar embeddings antes del clasificador
- **Early fusion**: usar landmarks como atención espacial sobre feature maps de la CNN
- **Cross-attention**: los landmarks atienden a features visuales y viceversa

### D. Pre-entrenamiento y datos sintéticos

#### D.1 Pre-entrenamiento en datasets masivos de manos

- **HaGRID** (Hand Gesture Recognition Image Dataset): ~700K imágenes, 18 clases de gestos, múltiples personas y fondos. Pre-entrenar aquí antes de fine-tuning en LSC
- **FreiHAND** / **HO-3D**: datasets 3D de mano con landmarks; útiles para pre-entrenar el extractor de features del grafo
- **InterHand2.6M**: dataset masivo de manos interactuando; pre-entrenamiento self-supervised para representaciones de mano

#### D.2 Aumentación de datos más agresiva

- **MixUp / CutMix**: mezclar landmarks de diferentes letras para crear ejemplos sintéticos
- **Generación de poses sintéticas**: muestrear configuraciones de mano plausibles usando un modelo paramétrico (MANO) y renderizar
- **Domain randomization**: variar iluminación, fondo, tono de piel, oclusión parcial en las imágenes de entrenamiento

#### D.3 Aprendizaje auto-supervisado

- **SimCLR / MoCo / BYOL** sobre imágenes de mano recortadas por MediaPipe: aprender representaciones visuales sin etiquetas, luego fine-tuning con pocas muestras etiquetadas
- **Masked Autoencoder (MAE)** sobre landmarks: enmascarar nodos del grafo y predecirlos, forzando al modelo a aprender la estructura anatómica

### E. Comparación sistemática propuesta

Para evaluar rigurosamente las alternativas, se propone:

1. **Misma métrica principal**: macro-F1 (porque el dataset está balanceado) + benchmark de estabilidad temporal
2. **Mismo split de datos**: train/val/test idénticos para todos los modelos
3. **Métricas adicionales**: tiempo de inferencia (ms), tamaño del modelo (MB), FPS en cámara real, tasa de frames sin detección
4. **Ablación de features**: comparar modelo con solo 63 coords vs. con features geométricas enriquecidas para cada arquitectura
5. **Validación cruzada**: k-fold (k=5) para estimar varianza del rendimiento
6. **Test de significancia estadística**: McNemar o Wilcoxon para determinar si diferencias entre modelos son significativas

---

## Infraestructura del proyecto

- **Notebook principal**: `LSC_INTERPRETER_up (1).ipynb` — experimento completo en Colab
- **Inferencia en tiempo real**: `lsc_inference_realtime.py` — script standalone con soporte para los 5 modelos (classic, mlp, gcn, gat, gat_robusto)
- **Benchmark de estabilidad**: `lsc_stability_benchmark.py` — comparación de métricas temporales entre modelos con cámara real
- **Artefactos guardados**: `model_artifacts_kaggle/` — modelos entrenados, encoder, metadatos
- **Dataset**: Roboflow (formato YOLO multiclase) + Kaggle (carpetas por clase)
