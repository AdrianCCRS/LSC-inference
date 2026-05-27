"""
LSC Real-Time Inference Script — CNN Hybrid Pipeline (ONNX)
=============================================================
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import json
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# ──────────────────────────────────────────────────────────────────────────────
# DETECTOR MEDIAPIPE (solo para bounding box)
# ──────────────────────────────────────────────────────────────────────────────

def create_mp_detector(model_path, min_det_conf=0.3, min_pres_conf=0.3,
                       min_track_conf=0.3):
    """Crea el detector MediaPipe Hands. Umbrales bajos para maximizar detecciones."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    model_path = Path(model_path)
    if not model_path.exists():
        # Intentar ruta relativa al directorio del script
        script_dir = Path(__file__).resolve().parent
        fallback_path = (script_dir / model_path).resolve()
        if fallback_path.exists():
            model_path = fallback_path

    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontro el modelo MediaPipe en: {model_path}\n"
            "Descargalo con:\n"
            "  wget https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )
    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        num_hands=1,
        min_hand_detection_confidence=min_det_conf,
        min_hand_presence_confidence=min_pres_conf,
        min_tracking_confidence=min_track_conf,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


def extract_hand_bbox(landmarks, img_h, img_w, padding=0.10):
    """
    Extrae el bounding box de la mano a partir de 21 landmarks MediaPipe.
    Anade un padding del 10% para no cortar puntas de dedos.
    Devuelve (x1, y1, x2, y2) en pixeles.
    """
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]

    x_min = max(0.0, min(xs))
    y_min = max(0.0, min(ys))
    x_max = min(1.0, max(xs))
    y_max = min(1.0, max(ys))

    w_box = x_max - x_min
    h_box = y_max - y_min
    pad_w = w_box * padding
    pad_h = h_box * padding

    x_min = max(0.0, x_min - pad_w)
    y_min = max(0.0, y_min - pad_h)
    x_max = min(1.0, x_max + pad_w)
    y_max = min(1.0, y_max + pad_h)

    x1 = int(x_min * img_w)
    y1 = int(y_min * img_h)
    x2 = int(x_max * img_w)
    y2 = int(y_max * img_h)
    return x1, y1, x2, y2


def preprocess_frame(frame_rgb, detector):
    """
    Pipeline hibrido de preprocesamiento:
      1. Detectar landmarks con MediaPipe.
      2. Si detecta: recortar bounding box con padding.
      3. Si NO detecta (fallback): usar frame completo reescalado.
      4. Redimensionar a 224x224, normalizar a [0,1].

    Devuelve (crop_rgb, detection_result) donde crop_rgb es (224,224,3)
    y detection_result puede ser None si fallback.
    """
    import mediapipe as mp

    h, w = frame_rgb.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=frame_rgb)
    result = detector.detect(mp_image)

    crop = None
    if result.hand_landmarks:
        lms = result.hand_landmarks[0]
        x1, y1, x2, y2 = extract_hand_bbox(lms, h, w)
        if (x2 - x1) > 10 and (y2 - y1) > 10:
            crop = frame_rgb[y1:y2, x1:x2]

    if crop is None or crop.size == 0:
        crop = frame_rgb  # fallback: imagen completa

    crop_resized = cv2.resize(crop, IMG_SIZE, interpolation=cv2.INTER_AREA)
    crop_norm = crop_resized.astype(np.float32) / 255.0
    return crop_norm, result


# ──────────────────────────────────────────────────────────────────────────────
# CARGA DEL MODELO CNN (ONNX Runtime — sin TensorFlow)
# ──────────────────────────────────────────────────────────────────────────────

def load_cnn_package(artifacts_dir):
    """
    Carga el modelo CNN via ONNX Runtime y el LabelEncoder.
    Prueba modelo_cnn_hybrid.onnx primero. Si no existe, intenta
    modelo_cnn_hybrid.keras (requiere TensorFlow).

    Devuelve un dict con session, encoder, class_names, metadata.
    """
    import onnxruntime as ort

    artifacts_dir = Path(artifacts_dir)
    if not artifacts_dir.exists():
        # Intentar ruta relativa al directorio del script
        script_dir = Path(__file__).resolve().parent
        fallback_path = (script_dir / artifacts_dir).resolve()
        if fallback_path.exists():
            artifacts_dir = fallback_path

    meta_path = artifacts_dir / "metadata_inferencia.json"
    encoder_path = artifacts_dir / "label_encoder.joblib"
    onnx_path = artifacts_dir / "modelo_cnn_hybrid.onnx"
    keras_path = artifacts_dir / "modelo_cnn_hybrid.keras"

    if not encoder_path.exists():
        raise FileNotFoundError(
            f"No se encontro label_encoder.joblib en {artifacts_dir}")

    metadata = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

    encoder = joblib.load(encoder_path)
    class_names = np.array(metadata.get("class_names", list(encoder.classes_)))

    # ── Cargar modelo ONNX (preferido, no necesita TF) ──
    if onnx_path.exists():
        session = ort.InferenceSession(
            str(onnx_path),
            providers=['CPUExecutionProvider'],
        )
        input_name = session.get_inputs()[0].name
        print(f"Modelo ONNX cargado: {onnx_path}")
        model_source = "onnx"
    elif keras_path.exists():
        import tensorflow as tf
        model = tf.keras.models.load_model(str(keras_path), compile=False)
        session = model
        input_name = None
        print(f"Modelo Keras cargado (fallback): {keras_path}")
        model_source = "keras"
    else:
        raise FileNotFoundError(
            f"No se encontro modelo (.onnx ni .keras) en {artifacts_dir}")

    print(f"  Arquitectura: {metadata.get('architecture', 'desconocida')}")
    print(f"  Accuracy test: {metadata.get('test_accuracy', 'N/A')}")
    print(f"  Macro-F1 test: {metadata.get('test_macro_f1', 'N/A')}")
    print(f"  Clases: {len(class_names)}")

    return {
        "session": session,
        "input_name": input_name,
        "model_source": model_source,
        "encoder": encoder,
        "class_names": class_names,
        "metadata": metadata,
        "model_type": "cnn_hybrid",
        "input": "image",
    }


def predict_cnn(pkg, crop_rgb):
    """
    Predice a partir de un crop RGB (224, 224, 3) en [0, 1].
    Usa ONNX Runtime (o Keras como fallback).
    Devuelve (label, confidence, probabilities).
    """
    batch = np.expand_dims(crop_rgb, axis=0).astype(np.float32)

    if pkg["model_source"] == "onnx":
        outputs = pkg["session"].run(None, {pkg["input_name"]: batch})
        probs = outputs[0][0]
    else:
        probs = pkg["session"].predict(batch, verbose=0)[0]

    idx = int(np.argmax(probs))
    label = str(pkg["class_names"][idx])
    confidence = float(probs[idx])
    return label, confidence, probs


# ──────────────────────────────────────────────────────────────────────────────
# BUFFER DE ESTABILIZACION TEMPORAL
# ──────────────────────────────────────────────────────────────────────────────

class PredictionBuffer:
    """
    Suaviza la salida del clasificador frame a frame.

    Parametros:
      maxlen         — ventana temporal (numero de frames).
      min_votes      — votos minimos de la misma clase para considerarla estable.
      min_confidence — umbral de confianza minima para agregar al buffer.
    """

    def __init__(self, maxlen=10, min_votes=6, min_confidence=0.50):
        self.buffer = deque(maxlen=maxlen)
        self.min_votes = min_votes
        self.min_confidence = min_confidence
        self.last_stable = None
        self.stable_conf_avg = 0.0

    def update(self, label, confidence):
        if label is None or confidence < self.min_confidence:
            return self.last_stable

        self.buffer.append((label, confidence))
        counts = Counter(l for l, _ in self.buffer)
        top_label, votes = counts.most_common(1)[0]

        if votes >= self.min_votes:
            self.last_stable = top_label
            self.stable_conf_avg = float(np.mean(
                [c for l, c in self.buffer if l == top_label]
            ))

        return self.last_stable

    def reset(self):
        self.buffer.clear()
        self.last_stable = None
        self.stable_conf_avg = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# VISUALIZACION
# ──────────────────────────────────────────────────────────────────────────────

_CLR_LINE   = (70, 130, 180)
_CLR_NODE   = (0, 140, 255)
_CLR_TEXT_W = (255, 255, 255)
_CLR_TEXT_G = (0, 220, 120)
_CLR_TEXT_Y = (0, 220, 220)
_CLR_BG     = (20, 20, 20)


def draw_landmarks(frame_bgr, detection_result):
    """Dibuja landmarks de MediaPipe sobre el frame."""
    out = frame_bgr.copy()
    if not detection_result or not detection_result.hand_landmarks:
        return out

    h, w = out.shape[:2]
    lms = detection_result.hand_landmarks[0]

    for i, j in HAND_CONNECTIONS:
        p1 = (int(lms[i].x * w), int(lms[i].y * h))
        p2 = (int(lms[j].x * w), int(lms[j].y * h))
        cv2.line(out, p1, p2, _CLR_LINE, 2, cv2.LINE_AA)

    for idx, lm in enumerate(lms):
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(out, (cx, cy), 5, _CLR_NODE, -1)
        cv2.putText(out, str(idx), (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, _CLR_TEXT_W, 1, cv2.LINE_AA)
    return out


def draw_overlay(frame_bgr, instant_label, instant_conf, stable_label,
                 stable_conf, model_name, fps, status=""):
    """Superpone HUD con prediccion y FPS."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    cv2.rectangle(out, (0, 0), (w, 100), _CLR_BG, -1)
    cv2.putText(out, "LSC CNN Hybrid Interpreter", (14, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, _CLR_TEXT_W, 2, cv2.LINE_AA)

    inst_txt = (f"Instant: {instant_label}  ({instant_conf:.0%})"
                if instant_label else "Instant: sin mano")
    cv2.putText(out, inst_txt, (14, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200, 200, 200), 1, cv2.LINE_AA)

    stable_txt = (f"Estable: {stable_label}  ({stable_conf:.0%})"
                  if stable_label else "Estable: esperando...")
    cv2.putText(out, stable_txt, (14, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, _CLR_TEXT_G, 2, cv2.LINE_AA)

    if stable_label:
        txt_size = cv2.getTextSize(stable_label, cv2.FONT_HERSHEY_SIMPLEX,
                                    3.5, 5)[0]
        cv2.putText(out, stable_label, (w - txt_size[0] - 16, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.5, _CLR_TEXT_Y, 5, cv2.LINE_AA)

    info = f"{model_name}  |  {fps:.1f} FPS"
    if status:
        info += f"  |  {status}"
    cv2.putText(out, info, (14, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(out, "q=salir  |  r=reiniciar buffer  |  d=debug",
                (14, h - 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# BUCLE PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

def run_camera(pkg, mp_model_path, camera_index=0, buffer_len=10,
               min_votes=6, min_conf=0.50, show_debug=False):
    detector = create_mp_detector(mp_model_path)
    buf = PredictionBuffer(maxlen=buffer_len, min_votes=min_votes,
                           min_confidence=min_conf)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la camara (indice {camera_index}).")

    cap.set(cv2.CAP_PROP_FPS, 30)

    model_label = "CNN_HYBRID"
    prev_time = time.perf_counter()
    fps = 0.0
    debug = show_debug

    print("=" * 60)
    print(f"  Modelo : {model_label}  |  clases: {len(pkg['class_names'])}")
    print(f"  Buffer : ventana={buffer_len}, votos={min_votes}, "
          f"conf>={min_conf:.0%}")
    print("  Teclas : q=salir  r=reiniciar buffer  d=debug")
    print("=" * 60)

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("Error: no se pudo leer el frame.")
                break

            frame_bgr = cv2.flip(frame_bgr, 1)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # ── Preprocesamiento + prediccion ──────────────────────────
            crop_rgb, det_result = preprocess_frame(frame_rgb, detector)
            instant_label, instant_conf = None, 0.0
            stable_conf = buf.stable_conf_avg

            try:
                instant_label, instant_conf, probs = predict_cnn(pkg, crop_rgb)
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

                # Dibujar landmarks si se detecto mano
                frame_bgr = draw_landmarks(frame_bgr, det_result)

            except Exception as exc:
                status = f"Error prediccion: {exc}"

            # ── FPS ────────────────────────────────────────────────────
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            # ── Overlay ─────────────────────────────────────────────────
            frame_bgr = draw_overlay(
                frame_bgr,
                instant_label, instant_conf,
                buf.last_stable, stable_conf,
                model_label, fps, status,
            )

            cv2.imshow("LSC CNN Hybrid Interpreter", frame_bgr)

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
        description="LSC CNN Hybrid Real-Time Interpreter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--artifacts", "-a", default="model_artifacts_cnn",
                        help="Directorio de artefactos (default: model_artifacts_cnn).")
    parser.add_argument("--mediapipe", "-mp", default="hand_landmarker.task",
                        help="Ruta al archivo hand_landmarker.task.")
    parser.add_argument("--camera", "-c", type=int, default=0,
                        help="Indice de la camara (default: 0).")
    parser.add_argument("--buffer-len", type=int, default=10,
                        help="Longitud del buffer temporal (default: 10).")
    parser.add_argument("--min-votes", type=int, default=6,
                        help="Votos minimos para estabilidad (default: 6).")
    parser.add_argument("--min-conf", type=float, default=0.50,
                        help="Confianza minima para el buffer (default: 0.50).")
    parser.add_argument("--debug", action="store_true",
                        help="Mostrar top-5 probabilidades.")
    return parser.parse_args()


def main():
    args = parse_args()
    artifacts = Path(args.artifacts)

    print(f"\nCargando modelo CNN desde '{artifacts}'...")
    pkg = load_cnn_package(artifacts)

    run_camera(
        pkg=pkg,
        mp_model_path=args.mediapipe,
        camera_index=args.camera,
        buffer_len=args.buffer_len,
        min_votes=args.min_votes,
        min_conf=args.min_conf,
        show_debug=args.debug,
    )


if __name__ == "__main__":
    main()
