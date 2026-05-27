"""
Genera graficas y tabla desde resultados_benchmark_v2.json (sin camara).
Uso:  python plot_benchmark_results.py
"""
import json
import sys
from pathlib import Path

import numpy as np

RESULTS_FILE = Path(__file__).parent / "resultados_benchmark_v2.json"
PLOT_OUT     = Path(__file__).parent / "benchmark_plots_v2.png"


def print_table(aggregates: list[dict]):
    cols = [
        ("Modelo",        "model",                "s",   15),
        ("Conf μ",        "confidence_mean_avg",  ".2%", 8),
        ("±SEM",          "confidence_mean_sem",  ".3f", 7),
        ("Acc μ",         "accuracy_avg",         ".2%", 8),
        ("±SEM",          "accuracy_sem",         ".3f", 7),
        ("Entropia",      "entropy_mean_avg",     ".3f", 9),
        ("Flip/s",        "flip_rate_avg",        ".2f", 8),
        ("Estabilidad",   "stability_ratio_avg",  ".2%", 12),
        ("Top-1",         "top1_agreement_avg",   ".2%", 8),
        ("Letras",        "n_letters",            "d",   7),
    ]
    header = "  ".join(f"{name:<{w}}" for name, _, _, w in cols)
    sep    = "  ".join("-" * w for _, _, _, w in cols)
    print("\n" + "=" * len(header))
    print("TABLA COMPARATIVA DE ESTABILIDAD EN TIEMPO REAL (v2 - HG-GCN)")
    print("=" * len(header))
    print(header)
    print(sep)

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
    print("=" * len(header))


def plot_from_json(results_file: str, save_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    with open(results_file) as f:
        data = json.load(f)

    aggregates = data.get("aggregates", data if isinstance(data, list) else [])
    if not aggregates:
        print("No hay datos agregados en el JSON.")
        return

    models  = [r["model"] for r in aggregates]
    n       = len(models)
    colors  = plt.cm.Set2(np.linspace(0, 1, max(n, 1)))
    col_map = {m: c for m, c in zip(models, colors)}
    n_letters = aggregates[0].get("n_letters", "?")

    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(
        f"Comparacion de Estabilidad Temporal — Agregado Multi-Letra (v2 - HG-GCN)\n"
        f"({n_letters} letras × {n} modelos — media ± SEM sobre letras)",
        fontsize=14, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)

    # 1. Confianza media con SEM
    ax1 = fig.add_subplot(gs[0, 0])
    means = [r.get("confidence_mean_avg", 0) for r in aggregates]
    sems  = [r.get("confidence_mean_sem", 0) for r in aggregates]
    bars  = ax1.bar(models, means, yerr=sems, capsize=6,
                    color=[col_map[m] for m in models], edgecolor="black", linewidth=0.8)
    ax1.set_ylim(0, 1.05); ax1.set_ylabel("Confianza")
    ax1.set_title("Confianza Media ± SEM\n(mas alto = mejor)")
    ax1.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    for bar, mean in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{mean:.2%}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax1.axhline(0.5, color="red", linestyle="--", alpha=0.4, linewidth=1)

    # 2. Entropia
    ax2 = fig.add_subplot(gs[0, 1])
    evals = [r.get("entropy_mean_avg", 0) for r in aggregates]
    esems = [r.get("entropy_mean_sem", 0) for r in aggregates]
    ax2.bar(models, evals, yerr=esems, capsize=6,
            color=[col_map[m] for m in models], edgecolor="black", linewidth=0.8)
    ax2.set_ylim(0, max(evals)*1.4 if max(evals)>0 else 1)
    ax2.set_ylabel("Entropia"); ax2.set_title("Entropia de Shannon\n(mas baja = mas seguro)")
    ax2.set_xticklabels(models, rotation=20, ha="right", fontsize=9)

    # 3. Flip rate
    ax3 = fig.add_subplot(gs[0, 2])
    fvals = [r.get("flip_rate_avg", 0) for r in aggregates]
    fsems = [r.get("flip_rate_sem", 0) for r in aggregates]
    ax3.bar(models, fvals, yerr=fsems, capsize=6,
            color=[col_map[m] for m in models], edgecolor="black", linewidth=0.8)
    ax3.set_ylabel("Cambios/s"); ax3.set_title("Flip Rate ± SEM\n(mas bajo = mas estable)")
    ax3.set_xticklabels(models, rotation=20, ha="right", fontsize=9)

    # 4. Stability ratio
    ax4 = fig.add_subplot(gs[1, 0])
    svals = [r.get("stability_ratio_avg", 0) for r in aggregates]
    ssems = [r.get("stability_ratio_sem", 0) for r in aggregates]
    ax4.bar(models, svals, yerr=ssems, capsize=6,
            color=[col_map[m] for m in models], edgecolor="black", linewidth=0.8)
    ax4.set_ylim(0, 1.05); ax4.set_ylabel("Stability Ratio")
    ax4.set_title("Stability Ratio ± SEM")
    ax4.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    for i, s in enumerate(svals):
        ax4.text(i, s + 0.01, f"{s:.2%}", ha="center", va="bottom", fontsize=8)

    # 5. Accuracy
    ax5 = fig.add_subplot(gs[1, 1])
    avals = [r.get("accuracy_avg", 0) for r in aggregates]
    asems = [r.get("accuracy_sem", 0) for r in aggregates]
    ax5.bar(models, avals, yerr=asems, capsize=6,
            color=[col_map[m] for m in models], edgecolor="black", linewidth=0.8)
    ax5.set_ylim(0, 1.05); ax5.set_ylabel("Accuracy")
    ax5.set_title("Accuracy vs Target Letter ± SEM\n(% frames correctos)")
    ax5.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    for i, a in enumerate(avals):
        ax5.text(i, a + 0.01, f"{a:.2%}", ha="center", va="bottom", fontsize=8)

    # 6. Top-1 agreement
    ax6 = fig.add_subplot(gs[1, 2])
    tvals = [r.get("top1_agreement_avg", 0) for r in aggregates]
    tsems = [r.get("top1_agreement_sem", 0) for r in aggregates]
    ax6.bar(models, tvals, yerr=tsems, capsize=6,
            color=[col_map[m] for m in models], edgecolor="black", linewidth=0.8)
    ax6.set_ylim(0, 1.05); ax6.set_ylabel("Acuerdo")
    ax6.set_title("Top-1 Agreement ± SEM\n(acuerdo frame-buffer)")
    ax6.set_xticklabels(models, rotation=20, ha="right", fontsize=9)

    # 7. Heatmap de accuracy por letra
    ax7 = fig.add_subplot(gs[2, :2])
    all_letters = aggregates[0].get("letters_tested", [])
    if all_letters:
        heatmap = np.zeros((len(models), len(all_letters)))
        for i, r in enumerate(aggregates):
            per_l = {pl["letter"]: pl.get("accuracy", 0) for pl in r.get("per_letter", [])}
            for j, l in enumerate(all_letters):
                heatmap[i, j] = per_l.get(l, np.nan)
        im = ax7.imshow(heatmap, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
        ax7.set_xticks(range(len(all_letters)))
        ax7.set_xticklabels(all_letters, fontsize=9)
        ax7.set_yticks(range(len(models)))
        ax7.set_yticklabels(models, fontsize=9)
        ax7.set_title("Accuracy por Letra y Modelo\n(verde = mejor)")
        plt.colorbar(im, ax=ax7, shrink=0.8)
        for i in range(len(models)):
            for j in range(len(all_letters)):
                val = heatmap[i, j]
                if not np.isnan(val):
                    ax7.text(j, i, f"{val:.0%}", ha="center", va="center",
                             fontsize=7, fontweight="bold",
                             color="white" if val < 0.5 else "black")

    # 8. Radar chart
    ax8 = fig.add_subplot(gs[2, 2], polar=True)
    mlabels = ["Conf\nmedia", "Accuracy", "Top-1\nAgree", "1-Flip\nRate", "1-Entropia"]
    N = len(mlabels)
    angles = [n/float(N)*2*np.pi for n in range(N)]; angles += angles[:1]
    ax8.set_xticks(angles[:-1]); ax8.set_xticklabels(mlabels, fontsize=8)
    ax8.set_ylim(0, 1); ax8.set_title("Radar de Metricas\n(area mayor = mejor)", pad=20, fontsize=10)

    max_flip = max(r.get("flip_rate_avg", 0) for r in aggregates) or 1.0
    for r in aggregates:
        vals = [
            r.get("confidence_mean_avg", 0),
            r.get("accuracy_avg", 0),
            r.get("top1_agreement_avg", 0),
            1.0 - min(r.get("flip_rate_avg", 0) / (max_flip+1e-6), 1.0),
            1.0 - r.get("entropy_mean_avg", 0),
        ]
        vals += vals[:1]
        ax8.plot(angles, vals, "o-", linewidth=2, color=col_map[r["model"]], label=r["model"])
        ax8.fill(angles, vals, alpha=0.1, color=col_map[r["model"]])
    ax8.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=8)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nGraficas guardadas en: {save_path}")


if __name__ == "__main__":
    if not RESULTS_FILE.exists():
        print(f"ERROR: No se encontro {RESULTS_FILE}")
        sys.exit(1)

    with open(RESULTS_FILE) as f:
        data = json.load(f)

    aggregates = data.get("aggregates", [])
    if not aggregates:
        print("No hay datos agregados.")
        sys.exit(1)

    print(f"Resultados cargados: {len(aggregates)} modelos × {aggregates[0].get('n_letters','?')} letras = {len(data.get('per_letter',[]))} sesiones\n")
    print_table(aggregates)
    plot_from_json(str(RESULTS_FILE), str(PLOT_OUT))
