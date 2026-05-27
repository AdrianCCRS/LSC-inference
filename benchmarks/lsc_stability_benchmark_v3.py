"""
LSC Stability Benchmark v3 — Classic + GAT + CNN Simultaneo
=============================================================
Evalua 3 modelos en paralelo sobre la misma grabacion de camara:
  - classic (Random Forest, 126 features tabulares)
  - GAT (Graph Attention Network, 18-dim node features)
  - CNN (MobileNetV3-Small via ONNX, crop RGB 224x224)

Flujo:
  1. Carga los 3 modelos (2 via lsc_inference_realtime_v2 + 1 via lsc_inference_realtime_cnn).
  2. Por cada letra: graba 5s con la camara.
  3. En cada frame, ejecuta predict() para los 3 modelos simultaneamente.
  4. Calcula metricas por modelo y agrega al final.

Uso:
  python benchmarks/lsc_stability_benchmark_v3.py --all-letters
  python benchmarks/lsc_stability_benchmark_v3.py --letter A
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

# ── Forzar CPU y silenciar TF/CUDA ────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Forzar PyTorch a CPU (evita que probe drivers CUDA inexistentes)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ.setdefault("TORCH_DEVICE", "cpu")

import cv2
import numpy as np

# ── Pre-importar PyTorch Geometric ANTES de MediaPipe ─────────────
# IMPORTANTE: torch_geometric.nn debe importarse ANTES de que MediaPipe
# inicialice su contexto EGL/OpenGL (via create_mp_detector).
# Si se importa despues, se produce un segfault por conflicto entre
# las bibliotecas compartidas de OpenGL.
try:
    import torch
    torch.set_num_threads(1)  # evitar sobreuso de CPU en benchmark
    from torch_geometric.nn import GATConv, GCNConv, GraphNorm  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ── Añadir directorios al path ────────────────────────────────────
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "inference"))
sys.path.insert(0, str(_repo_root))

from lsc_inference_realtime_v2 import (
    PredictionBuffer,
    create_mp_detector,
    draw_landmarks,
    extract_from_frame,
    load_model_package,
    predict,
    HAND_CONNECTIONS,
    normalize_landmarks,
    _to_numpy_coords,
)
from lsc_inference_realtime_cnn import (
    load_cnn_package,
    predict_cnn,
    preprocess_frame,
    extract_hand_bbox,
    IMG_SIZE as CNN_IMG_SIZE,
)

LSC_LETTERS = list("ABCDEFIKLMNOPQRTVUWXY")  # 21 letras

# ── Rutas por defecto (relativas al repo) ────────────────────────
_DEFAULT_LANDMARK_ARTIFACTS = _repo_root / "model_artifacts_v2"
_DEFAULT_CNN_ARTIFACTS      = _repo_root / "model_artifacts_cnn"
_DEFAULT_MEDIAPIPE_PATH     = _repo_root / "generated" / "hand_landmarker.task"
_DEFAULT_RESULTS_JSON       = _repo_root / "benchmarks" / "resultados_benchmark_v3.json"
_DEFAULT_PLOT_PNG           = _repo_root / "benchmarks" / "benchmark_plots_v3.png"


# ──────────────────────────────────────────────────────────────────
# COLECCION DE METRICAS
# ──────────────────────────────────────────────────────────────────

def shannon_entropy(probs):
    p = np.clip(probs, 1e-9, 1.0)
    raw = -np.sum(p * np.log(p))
    mx = np.log(len(p))
    return float(raw / mx) if mx > 0 else 0.0


class FrameMetricsCollector:
    def __init__(self, window=10):
        self.window = window
        self.confidences = []
        self.entropies = []
        self.labels = []
        self.timestamps = []

    def record(self, label, confidence, probs):
        self.confidences.append(confidence)
        self.entropies.append(shannon_entropy(probs))
        self.labels.append(label)
        self.timestamps.append(time.perf_counter())

    def compute_summary(self, model_key, target_letter):
        if not self.confidences:
            return {}
        confs = np.array(self.confidences)
        ents = np.array(self.entropies)
        labels = self.labels
        ts = np.array(self.timestamps)

        flips = sum(1 for a, b in zip(labels[:-1], labels[1:]) if a != b)
        duration = float(ts[-1] - ts[0]) if len(ts) > 1 else 1.0
        flip_rate = flips / duration if duration > 0 else 0.0

        modal_label = Counter(labels).most_common(1)[0][0]
        stability = sum(1 for ll in labels if ll == modal_label) / len(labels)

        agreements = []
        for i in range(len(labels)):
            start = max(0, i - self.window + 1)
            win = labels[start:i + 1]
            stable = Counter(win).most_common(1)[0][0]
            agreements.append(1 if labels[i] == stable else 0)
        top1_agreement = float(np.mean(agreements))

        accuracy = sum(1 for ll in labels if ll == target_letter) / len(labels)

        return {
            "model": model_key, "target_letter": target_letter,
            "n_frames": len(confs), "duration_s": round(duration, 2),
            "fps_effective": round(len(confs) / max(duration, 1e-6), 1),
            "confidence_mean": round(float(confs.mean()), 4),
            "confidence_std": round(float(confs.std()), 4),
            "confidence_p25": round(float(np.percentile(confs, 25)), 4),
            "confidence_p50": round(float(np.percentile(confs, 50)), 4),
            "confidence_p75": round(float(np.percentile(confs, 75)), 4),
            "entropy_mean": round(float(ents.mean()), 4),
            "entropy_std": round(float(ents.std()), 4),
            "flip_rate": round(flip_rate, 3),
            "stability_ratio": round(stability, 4),
            "top1_agreement": round(top1_agreement, 4),
            "accuracy": round(accuracy, 4),
            "modal_label": modal_label,
        }


# ──────────────────────────────────────────────────────────────────
# AGREGACION MULTI-LETRA
# ──────────────────────────────────────────────────────────────────

def aggregate_model_results(per_letter_summaries):
    summaries = [s for s in per_letter_summaries if s]  # filtrar vacios
    if not summaries:
        return {}
    keys_agg = ["confidence_mean", "confidence_std", "entropy_mean",
                "flip_rate", "stability_ratio", "top1_agreement", "accuracy"]
    agg = {
        "model": summaries[0]["model"],
        "n_letters": len(summaries),
        "letters_tested": sorted([s["target_letter"] for s in summaries]),
    }
    for key in keys_agg:
        vals = np.array([s[key] for s in summaries])
        agg[f"{key}_avg"] = round(float(vals.mean()), 4)
        agg[f"{key}_sem"] = round(float(vals.std(ddof=1) / np.sqrt(len(vals))), 4)
    agg["per_letter"] = [
        {
            "letter": s["target_letter"], "confidence_mean": s["confidence_mean"],
            "stability_ratio": s["stability_ratio"], "top1_agreement": s["top1_agreement"],
            "flip_rate": s["flip_rate"], "entropy_mean": s["entropy_mean"],
            "accuracy": s["accuracy"],
        }
        for s in summaries
    ]
    return agg


# ──────────────────────────────────────────────────────────────────
# UNA SESION = 3 MODELOS A LA VEZ
# ──────────────────────────────────────────────────────────────────

def record_session_all_models(pkgs, cnn_pkg, detector, target_letter,
                              duration_s=5.0, camera_index=0, countdown_s=1):
    """
    Graba UNA sesion de camara y evalua los 3 modelos en cada frame.
    pkgs: dict con modelos landmark (classic + gat)
    cnn_pkg: dict con modelo CNN
    detector: MediaPipe detector YA inicializado (para evitar segfault con TF)
    """
    all_model_keys = list(pkgs.keys()) + ["cnn"]
    collectors = {mk: FrameMetricsCollector(window=10) for mk in all_model_keys}

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la camara {camera_index}")
    cap.set(cv2.CAP_PROP_FPS, 30)

    phase = "countdown"
    rec_start = None
    prev_cd = countdown_s + 1

    print(f"\n  [3 MODELOS] Letra '{target_letter}' — {countdown_s}s cuenta + {duration_s}s grab.")

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_bgr = cv2.flip(frame_bgr, 1)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = frame_bgr.shape[:2]
            now = time.perf_counter()

            if phase == "countdown":
                if rec_start is None:
                    rec_start = now
                elapsed_cd = now - rec_start
                secs_left = max(0, countdown_s - int(elapsed_cd))
                if secs_left != prev_cd:
                    print(f"    {secs_left}...")
                    prev_cd = secs_left

                overlay = frame_bgr.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, frame_bgr, 0.5, 0, frame_bgr)
                msg = f"Letra: {target_letter}  |  Grabando en {secs_left}..."
                sz = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 2)[0]
                cv2.putText(frame_bgr, msg, ((w - sz[0]) // 2, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 220, 255), 2, cv2.LINE_AA)

                if elapsed_cd >= countdown_s:
                    phase = "recording"
                    rec_start = now
                    print("    GRABANDO!")

            elif phase == "recording":
                elapsed = now - rec_start
                remaining = duration_s - elapsed

                # ── Preprocesamiento compartido: un solo MediaPipe call ──
                crop_rgb, det_result = preprocess_frame(frame_rgb, detector)

                # ── Landmark models (classic + gat): extraer coords ──
                if det_result and det_result.hand_landmarks:
                    lms = det_result.hand_landmarks[0]
                    coords_norm = normalize_landmarks(lms)
                    frame_bgr = draw_landmarks(frame_bgr, det_result)

                    for mk, pkg in pkgs.items():
                        try:
                            label, conf, probs = predict(pkg, coords_norm)
                            collectors[mk].record(label, conf, probs)
                        except Exception:
                            pass
                else:
                    # Sin landmarks, los modelos de grafo/tabular no pueden predecir
                    coords_norm = None

                # ── CNN model: usa directamente el crop RGB ──
                try:
                    label_cnn, conf_cnn, probs_cnn = predict_cnn(cnn_pkg, crop_rgb)
                    collectors["cnn"].record(label_cnn, conf_cnn, probs_cnn)
                except Exception:
                    pass

                # HUD
                bar_w = int((elapsed / duration_s) * w)
                cv2.rectangle(frame_bgr, (0, h - 8), (bar_w, h), (0, 200, 100), -1)
                cv2.putText(frame_bgr, f"{target_letter}  {remaining:.1f}s",
                            (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 220, 100), 2)

                if elapsed >= duration_s:
                    break

            cv2.imshow("LSC Benchmark v3 (Classic + GAT + CNN)", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("  Interrumpido por el usuario.")
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # Calcular summaries
    summaries = {}
    for mk in all_model_keys:
        s = collectors[mk].compute_summary(mk, target_letter.upper())
        summaries[mk] = s
        print(f"    {mk:10s}  Conf={s.get('confidence_mean', 0):.2%}  "
              f"Acc={s.get('accuracy', 0):.2%}  Stab={s.get('stability_ratio', 0):.2%}")
    return summaries


# ──────────────────────────────────────────────────────────────────
# GRAFICAS (mismo formato que v2)
# ──────────────────────────────────────────────────────────────────

def plot_results(aggregates, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    aggregates = [r for r in aggregates if r.get("model") and r.get("n_letters", 0) > 0]
    if not aggregates:
        return
    models = [r["model"] for r in aggregates]
    n = len(models)
    colors = plt.cm.Set2(np.linspace(0, 1, max(n, 1)))
    col_map = {m: c for m, c in zip(models, colors)}
    n_letters = aggregates[0].get("n_letters", "?")

    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(
        f"Comparacion de Estabilidad Temporal — v3 (simult., mismos frames)\n"
        f"({n_letters} letras x {n} modelos — media ± SEM)",
        fontsize=14, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)

    # 1. Confianza
    ax1 = fig.add_subplot(gs[0, 0])
    means = [r.get("confidence_mean_avg", 0) for r in aggregates]
    sems = [r.get("confidence_mean_sem", 0) for r in aggregates]
    bars = ax1.bar(models, means, yerr=sems, capsize=6,
                    color=[col_map[m] for m in models], edgecolor="black", linewidth=0.8)
    ax1.set_ylim(0, 1.05); ax1.set_ylabel("Confianza")
    ax1.set_title("Confianza Media ± SEM"); ax1.tick_params(axis="x", rotation=20, labelsize=9)
    for bar, mean in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{mean:.2%}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax1.axhline(0.5, color="red", linestyle="--", alpha=0.4)

    # 2. Entropia
    ax2 = fig.add_subplot(gs[0, 1])
    e_vals = [r.get("entropy_mean_avg", 0) for r in aggregates]
    e_sems = [r.get("entropy_mean_sem", 0) for r in aggregates]
    ax2.bar(models, e_vals, yerr=e_sems, capsize=6,
            color=[col_map[m] for m in models], edgecolor="black", linewidth=0.8)
    ax2.set_ylim(0, max(e_vals)*1.4 if max(e_vals)>0 else 1)
    ax2.set_ylabel("Entropia"); ax2.set_title("Entropia de Shannon\n(mas baja = mas seguro)")
    ax2.tick_params(axis="x", rotation=20, labelsize=9)

    # 3. Flip rate
    ax3 = fig.add_subplot(gs[0, 2])
    f_vals = [r.get("flip_rate_avg", 0) for r in aggregates]
    f_sems = [r.get("flip_rate_sem", 0) for r in aggregates]
    ax3.bar(models, f_vals, yerr=f_sems, capsize=6,
            color=[col_map[m] for m in models], edgecolor="black", linewidth=0.8)
    ax3.set_ylabel("Cambios/s"); ax3.set_title("Flip Rate ± SEM\n(mas bajo = mas estable)")
    ax3.tick_params(axis="x", rotation=20, labelsize=9)

    # 4. Stability ratio
    ax4 = fig.add_subplot(gs[1, 0])
    s_vals = [r.get("stability_ratio_avg", 0) for r in aggregates]
    s_sems = [r.get("stability_ratio_sem", 0) for r in aggregates]
    ax4.bar(models, s_vals, yerr=s_sems, capsize=6,
            color=[col_map[m] for m in models], edgecolor="black", linewidth=0.8)
    ax4.set_ylim(0, 1.05); ax4.set_ylabel("Stability Ratio")
    ax4.set_title("Stability Ratio ± SEM"); ax4.tick_params(axis="x", rotation=20, labelsize=9)
    for i, s in enumerate(s_vals):
        ax4.text(i, s + 0.01, f"{s:.2%}", ha="center", va="bottom", fontsize=8)

    # 5. Accuracy
    ax5 = fig.add_subplot(gs[1, 1])
    a_vals = [r.get("accuracy_avg", 0) for r in aggregates]
    a_sems = [r.get("accuracy_sem", 0) for r in aggregates]
    ax5.bar(models, a_vals, yerr=a_sems, capsize=6,
            color=[col_map[m] for m in models], edgecolor="black", linewidth=0.8)
    ax5.set_ylim(0, 1.05); ax5.set_ylabel("Accuracy")
    ax5.set_title("Accuracy vs Target ± SEM"); ax5.tick_params(axis="x", rotation=20, labelsize=9)
    for i, a in enumerate(a_vals):
        ax5.text(i, a + 0.01, f"{a:.2%}", ha="center", va="bottom", fontsize=8)

    # 6. Top-1 agreement
    ax6 = fig.add_subplot(gs[1, 2])
    t_vals = [r.get("top1_agreement_avg", 0) for r in aggregates]
    t_sems = [r.get("top1_agreement_sem", 0) for r in aggregates]
    ax6.bar(models, t_vals, yerr=t_sems, capsize=6,
            color=[col_map[m] for m in models], edgecolor="black", linewidth=0.8)
    ax6.set_ylim(0, 1.05); ax6.set_ylabel("Acuerdo")
    ax6.set_title("Top-1 Agreement ± SEM"); ax6.tick_params(axis="x", rotation=20, labelsize=9)

    # 7. Heatmap accuracy por letra
    ax7 = fig.add_subplot(gs[2, :2])
    all_letters = aggregates[0].get("letters_tested", [])
    if all_letters:
        heatmap = np.zeros((len(models), len(all_letters)))
        for i, r in enumerate(aggregates):
            per_l = {pl["letter"]: pl.get("accuracy", 0) for pl in r.get("per_letter", [])}
            for j, lt in enumerate(all_letters):
                heatmap[i, j] = per_l.get(lt, np.nan)
        im = ax7.imshow(heatmap, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
        ax7.set_xticks(range(len(all_letters))); ax7.set_xticklabels(all_letters, fontsize=9)
        ax7.set_yticks(range(len(models))); ax7.set_yticklabels(models, fontsize=9)
        ax7.set_title("Accuracy por Letra y Modelo\n(mismos frames de entrada para todos)")
        plt.colorbar(im, ax=ax7, shrink=0.8)
        for i in range(len(models)):
            for j in range(len(all_letters)):
                val = heatmap[i, j]
                if not np.isnan(val):
                    ax7.text(j, i, f"{val:.0%}", ha="center", va="center",
                             fontsize=7, fontweight="bold",
                             color="white" if val < 0.5 else "black")

    # 8. Radar
    ax8 = fig.add_subplot(gs[2, 2], polar=True)
    mlabels = ["Conf\nmedia", "Accuracy", "Top-1\nAgree", "1-Flip\nRate", "1-Entropia"]
    N = len(mlabels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]; angles += angles[:1]
    ax8.set_xticks(angles[:-1]); ax8.set_xticklabels(mlabels, fontsize=8)
    ax8.set_ylim(0, 1); ax8.set_title("Radar de Metricas\n(area mayor = mejor)", pad=20, fontsize=10)
    max_flip = max(r.get("flip_rate_avg", 0) for r in aggregates) or 1.0
    for r in aggregates:
        vals = [r.get("confidence_mean_avg", 0), r.get("accuracy_avg", 0),
                r.get("top1_agreement_avg", 0),
                1.0 - min(r.get("flip_rate_avg", 0) / (max_flip + 1e-6), 1.0),
                1.0 - r.get("entropy_mean_avg", 0)]
        vals += vals[:1]
        ax8.plot(angles, vals, "o-", linewidth=2, color=col_map[r["model"]], label=r["model"])
        ax8.fill(angles, vals, alpha=0.1, color=col_map[r["model"]])
    ax8.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nGraficas guardadas en: {save_path}")


# ──────────────────────────────────────────────────────────────────
# TABLA
# ──────────────────────────────────────────────────────────────────

def print_table(aggregates):
    aggregates = [r for r in aggregates if r.get("model") and r.get("n_letters", 0) > 0]
    if not aggregates:
        return
    cols = [
        ("Modelo",       "model",                "s",   10),
        ("Conf mu",      "confidence_mean_avg",  ".2%", 8),
        ("±SEM",         "confidence_mean_sem",  ".3f", 7),
        ("Acc mu",       "accuracy_avg",         ".2%", 8),
        ("Entropia",     "entropy_mean_avg",     ".3f", 9),
        ("Flip/s",       "flip_rate_avg",        ".2f", 8),
        ("Estabilidad",  "stability_ratio_avg",  ".2%", 12),
        ("Top-1",        "top1_agreement_avg",   ".2%", 8),
        ("Letras",       "n_letters",            "d",   7),
    ]
    hdr = "  ".join(f"{name:<{w}}" for name, _, _, w in cols)
    sep = "  ".join("-" * w for _, _, _, w in cols)
    print("\n" + "=" * len(hdr))
    print("TABLA COMPARATIVA DE ESTABILIDAD (v3 — Classic + GAT + CNN)")
    print("=" * len(hdr)); print(hdr); print(sep)
    for r in sorted(aggregates, key=lambda x: x.get("stability_ratio_avg", 0), reverse=True):
        row = []
        for _, key, fmt, w in cols:
            val = r.get(key, 0)
            if fmt == "s":    cell = f"{str(val):<{w}}"
            elif fmt == ".2%": cell = f"{float(val):.2%}".rjust(w)
            elif fmt == ".3f": cell = f"{float(val):.3f}".rjust(w)
            elif fmt == ".2f": cell = f"{float(val):.2f}".rjust(w)
            elif fmt == "d":   cell = f"{int(val)}".rjust(w)
            else:              cell = str(val).rjust(w)
            row.append(cell)
        print("  ".join(row))
    print("=" * len(hdr))


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="LSC Stability Benchmark v3 — Classic + GAT + CNN")
    p.add_argument("--letter", "-l", default=None, help="Letra individual.")
    p.add_argument("--letters", nargs="+", default=None, help="Lista de letras.")
    p.add_argument("--all-letters", action="store_true", help="21 letras del alfabeto LSC.")
    p.add_argument("--duration", "-d", type=float, default=5.0, help="Segundos por letra (default: 5).")
    p.add_argument("--countdown", type=int, default=1, help="Segundos cuenta regresiva (default: 1).")
    p.add_argument("--artifacts-landmark", default=str(_DEFAULT_LANDMARK_ARTIFACTS),
                   help="Dir de artefactos para classic + GAT.")
    p.add_argument("--artifacts-cnn", default=str(_DEFAULT_CNN_ARTIFACTS),
                   help="Dir de artefactos para CNN.")
    p.add_argument("--mediapipe", "-mp", default=str(_DEFAULT_MEDIAPIPE_PATH))
    p.add_argument("--camera", "-c", type=int, default=0)
    p.add_argument("--results", "-r", default=str(_DEFAULT_RESULTS_JSON))
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--plot-out", default=str(_DEFAULT_PLOT_PNG))
    return p.parse_args()


def main():
    args = parse_args()

    if args.all_letters:
        letters = LSC_LETTERS
    elif args.letters:
        letters = [l.upper() for l in args.letters if l.upper() in LSC_LETTERS]
    elif args.letter:
        letters = [args.letter.upper()]
    else:
        letters = ["A"]

    print(f"Letras: {letters}  |  Modelos: classic, GAT, CNN  |  "
          f"{len(letters)} sesion(es) × {args.duration}s cada una\n")

    # Plot-only mode
    if args.plot_only:
        rp = Path(args.results)
        if not rp.exists():
            print(f"ERROR: {rp} no existe."); sys.exit(1)
        with rp.open() as f:
            data = json.load(f)
        aggs = data.get("aggregates", data if isinstance(data, list) else [])
        print_table(aggs)
        if not args.no_plot:
            plot_results(aggs, args.plot_out)
        return

    # ── 1. Pre-inicializar MediaPipe ANTES de cargar modelos ────
    #    NOTA: torch_geometric ya fue importado a nivel de modulo
    #    (arriba) para evitar segfault por conflicto EGL/OpenGL.
    print("Inicializando detector MediaPipe...")
    mp_detector = create_mp_detector(args.mediapipe)
    print("MediaPipe listo.\n")

    # ── 2. Cargar modelos landmark (classic + GAT) ANTES del CNN ─
    #    IMPORTANTE: cargar PyTorch primero. Si cargamos TF (Keras
    #    fallback del CNN) antes de PyTorch, se produce un segfault
    #    por conflicto entre runtimes TF y PyTorch en el mismo proceso.
    landmark_artifacts = Path(args.artifacts_landmark)
    pkgs = {}
    for mk in ["classic", "gat"]:
        try:
            pkgs[mk] = load_model_package(mk, landmark_artifacts)
        except Exception as e:
            print(f"  [SKIP] {mk}: {e}")
    if not pkgs:
        print("ERROR: Ningun modelo landmark cargado."); sys.exit(1)

    # ── 3. Cargar modelo CNN (preferiblemente via ONNX) ─────────
    cnn_artifacts = Path(args.artifacts_cnn)
    onnx_path = cnn_artifacts / "modelo_cnn_hybrid.onnx"
    if not onnx_path.exists():
        print(f"ADVERTENCIA: No se encontro {onnx_path}")
        print("  El fallback a Keras/TensorFlow puede causar segfault")
        print("  si PyTorch ya esta cargado en el mismo proceso.")
        print("  Ejecuta:  conda activate tf_env && pip install tf2onnx onnx")
        print("            python convert_keras_to_onnx.py")
    try:
        cnn_pkg = load_cnn_package(cnn_artifacts)
    except Exception as e:
        print(f"ERROR cargando CNN: {e}"); sys.exit(1)
    print(f"Modelos cargados: {list(pkgs.keys())} + CNN\n")

    # ── Una sesion por letra, todos los modelos a la vez ────
    all_model_keys = list(pkgs.keys()) + ["cnn"]
    per_model = {mk: [] for mk in all_model_keys}
    all_per_letter = []

    for letter in letters:
        print(f"{'=' * 60}\n  Letra: {letter}\n{'=' * 60}")
        summaries = record_session_all_models(
            pkgs, cnn_pkg, mp_detector, letter,
            args.duration, args.camera, args.countdown,
        )
        for mk, s in summaries.items():
            per_model[mk].append(s)
            all_per_letter.append(s)

        # Guardar incremental
        aggregates = [aggregate_model_results(per_model[mk]) for mk in all_model_keys]
        with open(args.results, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": {"letters": letters, "models": all_model_keys,
                             "duration_s": args.duration, "simultaneous": True},
                "per_letter": all_per_letter, "aggregates": aggregates,
            }, f, indent=2, ensure_ascii=False)
        print(f"  Guardado en {args.results}")

    # ── Final ────────────────────────────────────────────────
    aggregates = [aggregate_model_results(per_model[mk]) for mk in all_model_keys]
    if aggregates:
        print("\n" + "=" * 70)
        print("  RESULTADOS FINALES (3 MODELOS SOBRE LOS MISMOS FRAMES)")
        print("=" * 70)
        print_table(aggregates)
        if not args.no_plot:
            plot_results(aggregates, args.plot_out)


if __name__ == "__main__":
    main()
