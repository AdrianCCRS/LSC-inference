






























































"""
LSC Stability Benchmark
========================
Compara la estabilidad temporal de los modelos LSC en condiciones reales
de camara, capturando metricas que las metricas offline (accuracy/F1) no miden.

Metricas capturadas por modelo:
  - confidence_mean / confidence_std  : confianza promedio y su variacion
  - entropy_mean                      : entropia de Shannon promedio (lower = mas seguro)
  - flip_rate                         : cambios de letra por segundo
  - stability_ratio                   : % de frames donde la prediccion == moda de la ventana
  - top1_agreement                    : acuerdo del frame con la prediccion estable
  - confidence_p25/p50/p75            : percentiles de confianza

Uso:
  # Modo interactivo: muestra la camara y graba datos
  python lsc_stability_benchmark.py --letter A --duration 5

  # Graficar resultados ya guardados
  python lsc_stability_benchmark.py --plot-only --results resultados_benchmark.json

  # Comparar todos los modelos automaticamente (sin camara, sobre video grabado)
  python lsc_stability_benchmark.py --video test_video.mp4 --letter A

Flujo tipico:
  1. Ejecuta el benchmark para cada modelo (o usa --all-models).
  2. Mantén la seña fija en camara durante el tiempo indicado.
  3. Al terminar genera graficas y guarda JSON con los datos.
"""

import argparse
import json
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

# Importar funciones del script principal
sys.path.insert(0, str(Path(__file__).parent))
try:
    from lsc_inference_realtime import (
        PredictionBuffer,
        build_tabular_features,
        build_node_features,
        build_edge_features,
        build_adjacency,
        build_edge_index_torch,
        create_mp_detector,
        draw_landmarks,
        extract_from_frame,
        load_model_package,
        normalize_landmarks,
        predict,
        softmax_numpy,
        HAND_CONNECTIONS,
        EXTRA_EDGES,
    )
except ImportError as e:
    print(f"ERROR: No se pudo importar lsc_inference_realtime.py\n  {e}")
    print("Asegurate de que lsc_inference_realtime.py este en la misma carpeta.")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# COLECCION DE METRICAS
# ──────────────────────────────────────────────────────────────────────────────

def shannon_entropy(probs: np.ndarray) -> float:
    """Entropia de Shannon normalizada [0, 1]."""
    p = np.clip(probs, 1e-9, 1.0)
    raw = -np.sum(p * np.log(p))
    max_entropy = np.log(len(p))
    return float(raw / max_entropy) if max_entropy > 0 else 0.0


class FrameMetricsCollector:
    """Acumula metricas frame a frame durante la grabacion."""

    def __init__(self, window: int = 10):
        self.window       = window
        self.confidences  = []
        self.entropies    = []
        self.labels       = []
        self.timestamps   = []
        self._label_win   = deque(maxlen=window)

    def record(self, label: str, confidence: float, probs: np.ndarray):
        t = time.perf_counter()
        self.confidences.append(confidence)
        self.entropies.append(shannon_entropy(probs))
        self.labels.append(label)
        self.timestamps.append(t)
        self._label_win.append(label)

    def compute_summary(self, model_key: str, target_letter: str) -> dict:
        if not self.confidences:
            return {}

        confs  = np.array(self.confidences)
        ents   = np.array(self.entropies)
        labels = self.labels
        ts     = np.array(self.timestamps)

        # Flip rate: cambios de clase por segundo
        flips = sum(1 for a, b in zip(labels[:-1], labels[1:]) if a != b)
        duration = float(ts[-1] - ts[0]) if len(ts) > 1 else 1.0
        flip_rate = flips / duration if duration > 0 else 0.0

        # Stability ratio: % frames cuya prediccion == la moda de toda la sesion
        modal_label = Counter(labels).most_common(1)[0][0]
        stability   = sum(1 for l in labels if l == modal_label) / len(labels)

        # Acuerdo con el buffer estable (ventana deslizante)
        agreements = []
        for i in range(len(labels)):
            start = max(0, i - self.window + 1)
            win   = labels[start:i + 1]
            stable = Counter(win).most_common(1)[0][0]
            agreements.append(1 if labels[i] == stable else 0)
        top1_agreement = float(np.mean(agreements))

        return {
            "model"           : model_key,
            "target_letter"   : target_letter,
            "n_frames"        : len(confs),
            "duration_s"      : round(duration, 2),
            "fps_effective"   : round(len(confs) / max(duration, 1e-6), 1),
            "confidence_mean" : round(float(confs.mean()), 4),
            "confidence_std"  : round(float(confs.std()), 4),
            "confidence_p25"  : round(float(np.percentile(confs, 25)), 4),
            "confidence_p50"  : round(float(np.percentile(confs, 50)), 4),
            "confidence_p75"  : round(float(np.percentile(confs, 75)), 4),
            "confidence_min"  : round(float(confs.min()), 4),
            "entropy_mean"    : round(float(ents.mean()), 4),
            "entropy_std"     : round(float(ents.std()), 4),
            "flip_rate"       : round(flip_rate, 3),
            "stability_ratio" : round(stability, 4),
            "top1_agreement"  : round(top1_agreement, 4),
            "modal_label"     : modal_label,
            # Series temporales completas (para graficas)
            "series_confidence": [round(float(c), 4) for c in confs],
            "series_entropy"   : [round(float(e), 4) for e in ents],
            "series_labels"    : labels,
            "series_timestamps": [round(float(t - ts[0]), 4) for t in ts],
        }


# ──────────────────────────────────────────────────────────────────────────────
# BUCLE DE GRABACION
# ──────────────────────────────────────────────────────────────────────────────

def _countdown(frame, seconds_left: int, model_key: str, letter: str):
    """Overlay de cuenta regresiva antes de grabar."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    msg1 = f"Modelo: {model_key.upper()}"
    msg2 = f"Seña objetivo: {letter.upper()}"
    msg3 = f"Grabando en {seconds_left}..."
    msg4 = "Mantén la seña fija frente a la camara"

    for i, (msg, y, scale, color) in enumerate([
        (msg1, h//2 - 100, 0.9,  (200, 200, 200)),
        (msg2, h//2 - 50,  1.2,  (0, 220, 120)),
        (msg3, h//2 + 20,  2.0,  (0, 220, 255)),
        (msg4, h//2 + 90,  0.65, (200, 200, 200)),
    ]):
        sz = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
        x  = (w - sz[0]) // 2
        cv2.putText(frame, msg, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
    return frame


def record_session(pkg: dict,
                   mp_model_path: str,
                   target_letter: str,
                   duration_s: float = 5.0,
                   camera_index: int = 0,
                   countdown_s: int = 3) -> dict:
    """
    Abre la camara, cuenta regresiva y graba metricas durante duration_s segundos.
    Retorna el summary de metricas.
    """
    detector  = create_mp_detector(mp_model_path)
    collector = FrameMetricsCollector(window=10)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la camara {camera_index}")
    cap.set(cv2.CAP_PROP_FPS, 30)

    model_key = pkg["key"]
    phase     = "countdown"     # → "recording" → "done"
    rec_start = None
    prev_cd   = countdown_s + 1

    print(f"\n[{model_key.upper()}] Sesion para letra '{target_letter.upper()}' "
          f"– {countdown_s}s cuenta regresiva + {duration_s}s grabacion")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            now = time.perf_counter()

            if phase == "countdown":
                secs_left = countdown_s - int(now - (rec_start or now))
                if rec_start is None:
                    rec_start = now           # inicio de cuenta atras
                    secs_left = countdown_s

                elapsed_cd = now - rec_start
                secs_left  = max(0, countdown_s - int(elapsed_cd))

                if secs_left != prev_cd:
                    print(f"  {secs_left}...")
                    prev_cd = secs_left

                frame = _countdown(frame, secs_left, model_key, target_letter)

                if elapsed_cd >= countdown_s:
                    phase     = "recording"
                    rec_start = now
                    print("  ¡GRABANDO!")

            elif phase == "recording":
                elapsed = now - rec_start
                remaining = duration_s - elapsed

                # Deteccion y prediccion
                entry = extract_from_frame(frame, detector)
                if entry is not None:
                    try:
                        label, conf, probs = predict(pkg, entry["coords_norm"])
                        collector.record(label, conf, probs)
                        frame = draw_landmarks(frame, entry["detection_result"])
                    except Exception as exc:
                        pass

                # HUD de grabacion
                bar_w = int((elapsed / duration_s) * w)
                cv2.rectangle(frame, (0, h - 8), (bar_w, h), (0, 200, 100), -1)
                cv2.putText(frame,
                            f"GRABANDO [{model_key.upper()}]  letra: {target_letter.upper()}"
                            f"  {remaining:.1f}s restantes",
                            (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                            (0, 220, 100), 2, cv2.LINE_AA)

                n = len(collector.confidences)
                if n > 0:
                    mean_c = np.mean(collector.confidences[-30:])
                    cv2.putText(frame,
                                f"Frames: {n}  Conf avg: {mean_c:.2%}",
                                (14, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                                (200, 200, 200), 1, cv2.LINE_AA)

                if elapsed >= duration_s:
                    phase = "done"
                    break

            cv2.imshow("LSC Stability Benchmark", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("  Interrumpido por el usuario.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    summary = collector.compute_summary(model_key, target_letter.upper())
    print(f"  Frames grabados: {summary.get('n_frames', 0)}")
    print(f"  Confianza media: {summary.get('confidence_mean', 0):.2%}")
    print(f"  Flip rate:       {summary.get('flip_rate', 0):.2f} cambios/s")
    print(f"  Estabilidad:     {summary.get('stability_ratio', 0):.2%}")
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# GRAFICAS
# ──────────────────────────────────────────────────────────────────────────────

def plot_results(results: list[dict], save_path: str = "benchmark_plots.png"):
    """
    Genera un panel de graficas comparativas entre modelos.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib no disponible. Instala con: pip install matplotlib")
        return

    if not results:
        print("No hay resultados para graficar.")
        return

    models   = [r["model"] for r in results]
    colors   = plt.cm.Set2(np.linspace(0, 1, len(models)))
    col_map  = {m: c for m, c in zip(models, colors)}

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        "Comparación de Estabilidad Temporal entre Modelos LSC\n"
        "(Métricas en inferencia con cámara real – no capturadas por accuracy/F1 offline)",
        fontsize=14, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Confianza media con barras de error (std) ───────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    means = [r["confidence_mean"] for r in results]
    stds  = [r["confidence_std"]  for r in results]
    bars  = ax1.bar(models, means, yerr=stds, capsize=6,
                    color=[col_map[m] for m in models], edgecolor="black", linewidth=0.8)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Confianza")
    ax1.set_title("Confianza Media ± Desv. Est.\n(más alto y más estrecho = mejor)")
    ax1.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    for bar, mean in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{mean:.2%}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax1.axhline(0.5, color="red", linestyle="--", alpha=0.4, linewidth=1, label="Umbral 50%")
    ax1.legend(fontsize=8)

    # ── 2. Entropia promedio ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    entropies = [r["entropy_mean"] for r in results]
    ent_stds  = [r["entropy_std"]  for r in results]
    ax2.bar(models, entropies, yerr=ent_stds, capsize=6,
            color=[col_map[m] for m in models], edgecolor="black", linewidth=0.8)
    ax2.set_ylim(0, max(entropies) * 1.35 if entropies else 1)
    ax2.set_ylabel("Entropía normalizada")
    ax2.set_title("Entropía de Shannon (normalizada)\n(más baja = más seguro y enfocado)")
    ax2.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    for i, (e, m) in enumerate(zip(entropies, models)):
        ax2.text(i, e + 0.005, f"{e:.3f}", ha="center", va="bottom", fontsize=8)

    # ── 3. Flip rate ───────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    flips = [r["flip_rate"] for r in results]
    ax3.bar(models, flips, color=[col_map[m] for m in models],
            edgecolor="black", linewidth=0.8)
    ax3.set_ylabel("Cambios de letra / segundo")
    ax3.set_title("Tasa de Cambio (Flip Rate)\n(más bajo = predicción más estable)")
    ax3.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    for i, f in enumerate(flips):
        ax3.text(i, f + 0.02, f"{f:.2f}", ha="center", va="bottom", fontsize=8)

    # ── 4. Stability ratio ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    stab = [r["stability_ratio"] for r in results]
    ax4.bar(models, stab, color=[col_map[m] for m in models],
            edgecolor="black", linewidth=0.8)
    ax4.set_ylim(0, 1.05)
    ax4.set_ylabel("Stability Ratio")
    ax4.set_title("Stability Ratio\n(% frames con la letra modal de la sesión)")
    ax4.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    for i, s in enumerate(stab):
        ax4.text(i, s + 0.01, f"{s:.2%}", ha="center", va="bottom", fontsize=8)

    # ── 5. Box plots de confianza ──────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    conf_series = [r["series_confidence"] for r in results]
    bp = ax5.boxplot(conf_series, patch_artist=True, notch=True,
                     medianprops=dict(color="black", linewidth=2))
    for patch, m in zip(bp["boxes"], models):
        patch.set_facecolor(col_map[m])
        patch.set_alpha(0.7)
    ax5.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    ax5.set_ylabel("Confianza")
    ax5.set_ylim(0, 1.05)
    ax5.set_title("Distribución de Confianza\n(mediana alta + caja estrecha = ideal)")
    ax5.axhline(0.5, color="red", linestyle="--", alpha=0.4, linewidth=1)

    # ── 6. Top-1 agreement (consistencia con ventana estable) ──────────────
    ax6 = fig.add_subplot(gs[1, 2])
    agree = [r["top1_agreement"] for r in results]
    ax6.bar(models, agree, color=[col_map[m] for m in models],
            edgecolor="black", linewidth=0.8)
    ax6.set_ylim(0, 1.05)
    ax6.set_ylabel("Acuerdo con buffer")
    ax6.set_title("Top-1 Agreement\n(acuerdo frame-buffer deslizante)")
    ax6.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    for i, a in enumerate(agree):
        ax6.text(i, a + 0.01, f"{a:.2%}", ha="center", va="bottom", fontsize=8)

    # ── 7. Series temporales de confianza ──────────────────────────────────
    ax7 = fig.add_subplot(gs[2, :2])
    for r in results:
        ts   = r["series_timestamps"]
        conf = r["series_confidence"]
        # Suavizado con media movil
        window = min(15, len(conf))
        smoothed = np.convolve(conf, np.ones(window)/window, mode="valid")
        ts_sm    = ts[window//2: window//2 + len(smoothed)]
        ax7.plot(ts_sm, smoothed, label=r["model"],
                 color=col_map[r["model"]], linewidth=2)
        ax7.fill_between(ts_sm, smoothed, alpha=0.1, color=col_map[r["model"]])

    ax7.set_xlabel("Tiempo (s)")
    ax7.set_ylabel("Confianza (suavizada)")
    ax7.set_ylim(0, 1.05)
    ax7.set_title("Evolución Temporal de la Confianza (media móvil)\n"
                  "(señal plana y alta = modelo estable en tiempo real)")
    ax7.axhline(0.5, color="red", linestyle="--", alpha=0.3, linewidth=1)
    ax7.legend(fontsize=9)
    ax7.grid(alpha=0.3)

    # ── 8. Radar / spider chart de metricas normalizadas ──────────────────
    ax8 = fig.add_subplot(gs[2, 2], polar=True)
    metrics_labels = [
        "Conf\nmedia", "Estabilidad", "Top-1\nAcuerdo",
        "1-Flip\nRate", "1-Entropia",
    ]
    N = len(metrics_labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    ax8.set_xticks(angles[:-1])
    ax8.set_xticklabels(metrics_labels, fontsize=8)
    ax8.set_ylim(0, 1)
    ax8.set_title("Radar de Métricas\n(área mayor = mejor)", pad=20, fontsize=10)

    # Normalizar flip_rate: invertir (menos flips = mejor) y normalizar al max
    max_flip = max(r["flip_rate"] for r in results) or 1.0

    for r in results:
        values = [
            r["confidence_mean"],
            r["stability_ratio"],
            r["top1_agreement"],
            1.0 - min(r["flip_rate"] / (max_flip + 1e-6), 1.0),
            1.0 - r["entropy_mean"],
        ]
        values += values[:1]
        ax8.plot(angles, values, "o-", linewidth=2,
                 color=col_map[r["model"]], label=r["model"])
        ax8.fill(angles, values, alpha=0.1, color=col_map[r["model"]])

    ax8.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nGraficas guardadas en: {save_path}")
    plt.show()


def print_comparison_table(results: list[dict]):
    """Imprime tabla comparativa en consola."""
    if not results:
        return

    cols = [
        ("Modelo",       "model",            "s",  16),
        ("Conf μ",       "confidence_mean",  ".2%", 8),
        ("Conf σ",       "confidence_std",   ".2%", 8),
        ("Entropía",     "entropy_mean",     ".3f", 9),
        ("Flip/s",       "flip_rate",        ".2f", 8),
        ("Estabilidad",  "stability_ratio",  ".2%", 12),
        ("Top-1 Agree",  "top1_agreement",   ".2%", 12),
        ("Frames",       "n_frames",         "d",   7),
    ]

    header = "  ".join(f"{name:<{w}}" for name, _, _, w in cols)
    sep    = "  ".join("-" * w for _, _, _, w in cols)
    print("\n" + "=" * len(header))
    print("TABLA COMPARATIVA DE ESTABILIDAD EN TIEMPO REAL")
    print("=" * len(header))
    print(header)
    print(sep)

    for r in sorted(results, key=lambda x: x["stability_ratio"], reverse=True):
        row_parts = []
        for name, key, fmt, w in cols:
            val = r.get(key, 0)
            if fmt == "s":
                cell = f"{str(val):<{w}}"
            elif fmt == ".2%":
                cell = f"{float(val):.2%}".rjust(w)
            elif fmt == ".3f":
                cell = f"{float(val):.3f}".rjust(w)
            elif fmt == ".2f":
                cell = f"{float(val):.2f}".rjust(w)
            elif fmt == "d":
                cell = f"{int(val)}".rjust(w)
            else:
                cell = str(val).rjust(w)
            row_parts.append(cell)
        print("  ".join(row_parts))

    print("=" * len(header))
    print("\nInterpretacion:")
    print("  Conf μ alta     → el modelo es más seguro en sus predicciones")
    print("  Conf σ baja     → la confianza es consistente (no oscila)")
    print("  Entropía baja   → distribución de probabilidad concentrada en una clase")
    print("  Flip/s bajo     → la letra predicha cambia menos (más estable)")
    print("  Estabilidad alta → más % de frames concuerdan con la letra modal")
    print("  Top-1 Agree alto → cada frame coincide con el consenso de la ventana\n")


# ──────────────────────────────────────────────────────────────────────────────
# MODO VIDEO (sin camara)
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_on_video(video_path: str, pkg: dict, mp_model_path: str,
                       target_letter: str) -> dict:
    """Corre el benchmark sobre un archivo de video en lugar de la camara."""
    detector  = create_mp_detector(mp_model_path)
    collector = FrameMetricsCollector(window=10)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Procesando {total} frames del video...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        entry = extract_from_frame(frame, detector)
        if entry is not None:
            try:
                label, conf, probs = predict(pkg, entry["coords_norm"])
                collector.record(label, conf, probs)
            except Exception:
                pass

    cap.release()
    return collector.compute_summary(pkg["key"], target_letter.upper())


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="LSC Stability Benchmark – compara estabilidad real de modelos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--letter", "-l", default="A",
                   help="Letra/seña que vas a mostrar en camara (default: A).")
    p.add_argument("--duration", "-d", type=float, default=5.0,
                   help="Segundos de grabacion por modelo (default: 5).")
    p.add_argument("--countdown", type=int, default=3,
                   help="Segundos de cuenta regresiva antes de grabar (default: 3).")
    p.add_argument("--models", "-m", nargs="+",
                   default=["classic", "mlp", "gcn", "gat", "gat_robusto"],
                   choices=["classic", "mlp", "gcn", "gat", "gat_robusto"],
                   help="Modelos a comparar.")
    p.add_argument("--artifacts", "-a", default="model_artifacts_kaggle",
                   help="Directorio de artefactos.")
    p.add_argument("--mediapipe", "-mp", default="hand_landmarker.task",
                   help="Ruta al hand_landmarker.task.")
    p.add_argument("--camera", "-c", type=int, default=0,
                   help="Indice de la camara.")
    p.add_argument("--video", "-v", default=None,
                   help="Video de entrada (en lugar de camara).")
    p.add_argument("--results", "-r", default="resultados_benchmark.json",
                   help="Archivo JSON donde guardar/leer resultados.")
    p.add_argument("--plot-only", action="store_true",
                   help="Solo graficar resultados existentes (sin camara).")
    p.add_argument("--no-plot", action="store_true",
                   help="No mostrar graficas al final.")
    p.add_argument("--plot-out", default="benchmark_plots.png",
                   help="Archivo de salida para las graficas.")
    return p.parse_args()


def main():
    args      = parse_args()
    artifacts = Path(args.artifacts)
    results   = []

    # ── Solo graficar resultados previos ──────────────────────────────────
    if args.plot_only:
        results_path = Path(args.results)
        if not results_path.exists():
            print(f"ERROR: No se encontro {results_path}")
            sys.exit(1)
        with results_path.open() as f:
            results = json.load(f)
        print(f"Cargados {len(results)} resultados desde {results_path}")
        print_comparison_table(results)
        if not args.no_plot:
            plot_results(results, args.plot_out)
        return

    # ── Grabar / procesar cada modelo ─────────────────────────────────────
    for model_key in args.models:
        print(f"\n{'='*60}")
        print(f"  Modelo: {model_key.upper()}")
        print(f"{'='*60}")

        try:
            pkg = load_model_package(model_key, artifacts)
        except Exception as e:
            print(f"  [SKIP] No se pudo cargar {model_key}: {e}")
            continue

        if args.video:
            summary = benchmark_on_video(
                args.video, pkg, args.mediapipe, args.letter
            )
        else:
            summary = record_session(
                pkg          = pkg,
                mp_model_path= args.mediapipe,
                target_letter= args.letter,
                duration_s   = args.duration,
                camera_index = args.camera,
                countdown_s  = args.countdown,
            )

        results.append(summary)

        # Guardar resultados incrementalmente
        with open(args.results, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Guardado en {args.results}")

    # ── Tabla y graficas ───────────────────────────────────────────────────
    if results:
        print_comparison_table(results)
        if not args.no_plot:
            plot_results(results, args.plot_out)
    else:
        print("\nNo se capturaron datos. Verifica que los modelos existen en la carpeta de artefactos.")


if __name__ == "__main__":
    main()