from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_training_runs(path: Path) -> List[List[Dict]]:
    runs: List[List[Dict]] = []
    current: List[Dict] = []

    if not path.exists():
        return runs

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            epoch = int(rec.get("epoch", 0))

            if current and epoch <= int(current[-1].get("epoch", 0)):
                runs.append(current)
                current = [rec]
            else:
                current.append(rec)

    if current:
        runs.append(current)

    return runs


def get_class_rows(report: Dict) -> List[Dict]:
    rows: List[Dict] = []
    skip = {"micro avg", "macro avg", "weighted avg"}

    for name, vals in report.items():
        if name in skip:
            continue
        if not isinstance(vals, dict):
            continue
        if not {"precision", "recall", "f1-score", "support"}.issubset(vals.keys()):
            continue

        rows.append(
            {
                "class": name,
                "precision": float(vals["precision"]),
                "recall": float(vals["recall"]),
                "f1": float(vals["f1-score"]),
                "support": int(vals["support"]),
            }
        )

    rows.sort(key=lambda x: x["support"], reverse=True)
    return rows


def compute_ovr_confusion(rows: List[Dict], total_support: int) -> List[Dict]:
    out: List[Dict] = []

    for r in rows:
        support = float(r["support"])
        precision = float(r["precision"])
        recall = float(r["recall"])

        tp = recall * support
        fn = support - tp

        if precision > 0:
            pred_pos = tp / precision
            fp = max(pred_pos - tp, 0.0)
        else:
            fp = 0.0

        tn = max(total_support - support - fp, 0.0)

        out.append(
            {
                "class": r["class"],
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "tp_i": int(round(tp)),
                "fp_i": int(round(fp)),
                "fn_i": int(round(fn)),
                "tn_i": int(round(tn)),
            }
        )

    return out


def save_confusion_csv(path: Path, conf_rows: List[Dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "TP", "FP", "FN", "TN", "TP_float", "FP_float", "FN_float", "TN_float"])
        for r in conf_rows:
            writer.writerow(
                [
                    r["class"],
                    r["tp_i"],
                    r["fp_i"],
                    r["fn_i"],
                    r["tn_i"],
                    f"{r['tp']:.3f}",
                    f"{r['fp']:.3f}",
                    f"{r['fn']:.3f}",
                    f"{r['tn']:.3f}",
                ]
            )


def plot_overall_metrics(metrics: Dict, out_path: Path) -> None:
    keys = ["token_accuracy", "entity_precision", "entity_recall", "entity_f1"]
    labels = ["Token Acc", "Entity Precision", "Entity Recall", "Entity F1"]
    vals = [float(metrics.get(k, 0.0)) * 100 for k in keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, vals, color=["#4E79A7", "#59A14F", "#F28E2B", "#E15759"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage")
    ax.set_title("Hybrid Evaluation: Overall Metrics")

    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.2f}%", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_class_prf1(rows: List[Dict], out_path: Path) -> None:
    labels = [r["class"] for r in rows]
    p = np.array([r["precision"] for r in rows]) * 100
    r = np.array([r["recall"] for r in rows]) * 100
    f = np.array([r["f1"] for r in rows]) * 100

    x = np.arange(len(labels))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, p, width=w, label="Precision", color="#4E79A7")
    ax.bar(x, r, width=w, label="Recall", color="#59A14F")
    ax.bar(x + w, f, width=w, label="F1", color="#E15759")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage")
    ax.set_title("Per-Class Precision / Recall / F1")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_class_support(rows: List[Dict], out_path: Path) -> None:
    labels = [r["class"] for r in rows]
    supports = [r["support"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(labels, supports, color="#76B7B2")
    ax.set_xlabel("Entity Count (Support)")
    ax.set_title("Class Support Distribution")

    for b, v in zip(bars, supports):
        ax.text(v + max(supports) * 0.01, b.get_y() + b.get_height() / 2, str(v), va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_ovr_confusion(conf_rows: List[Dict], out_path: Path) -> None:
    n = len(conf_rows)
    cols = 3
    rows_n = math.ceil(n / cols)

    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 4.2, rows_n * 4.0))
    if isinstance(axes, np.ndarray):
        axes = axes.ravel()
    else:
        axes = np.array([axes])

    for i, row in enumerate(conf_rows):
        ax = axes[i]
        mat = np.array([[row["tp_i"], row["fn_i"]], [row["fp_i"], row["tn_i"]]], dtype=float)

        im = ax.imshow(mat, cmap="Blues")
        ax.set_title(f"{row['class']} (OvR)")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pred +", "Pred -"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["True +", "True -"])

        for rr in range(2):
            for cc in range(2):
                ax.text(cc, rr, str(int(mat[rr, cc])), ha="center", va="center", color="black", fontsize=10)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("One-vs-Rest Confusion Matrices (Derived from Precision/Recall/Support)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_training_curves(training_runs: List[List[Dict]], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for run_idx, run in enumerate(training_runs, start=1):
        epochs = [int(r.get("epoch", 0)) for r in run]
        train_loss = [float(r.get("train_loss", 0.0)) for r in run]
        val_f1 = [float(r.get("val_f1", 0.0)) * 100 for r in run]
        val_precision = [float(r.get("val_precision", 0.0)) * 100 for r in run]
        val_recall = [float(r.get("val_recall", 0.0)) * 100 for r in run]

        axes[0].plot(epochs, train_loss, marker="o", label=f"Run {run_idx}")
        axes[1].plot(epochs, val_f1, marker="o", label=f"Run {run_idx} F1")
        axes[1].plot(epochs, val_precision, marker="^", linestyle="--", label=f"Run {run_idx} Prec")
        axes[1].plot(epochs, val_recall, marker="s", linestyle=":", label=f"Run {run_idx} Rec")

    axes[0].set_title("Training Loss by Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Validation Metrics by Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Percentage")
    axes[1].set_ylim(80, 100)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate confusion matrix and metric graphs from saved stats.")
    parser.add_argument(
        "--metrics_json",
        default="hybrid_test_metrics_2k_conservative.json",
        help="Path to metrics JSON produced by evaluation.",
    )
    parser.add_argument(
        "--training_log",
        default="models/muril-hiner/training_log.jsonl",
        help="Path to training log JSONL.",
    )
    parser.add_argument(
        "--out_dir",
        default="plots",
        help="Output directory for generated plots.",
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics_json)
    training_path = Path(args.training_log)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_json(metrics_path)
    report = metrics.get("classification_report", {})
    class_rows = get_class_rows(report)

    total_support = int(round(sum(r["support"] for r in class_rows)))
    conf_rows = compute_ovr_confusion(class_rows, total_support)

    save_confusion_csv(out_dir / "hybrid_ovr_confusion_counts.csv", conf_rows)
    plot_overall_metrics(metrics, out_dir / "hybrid_overall_metrics.png")
    plot_class_prf1(class_rows, out_dir / "hybrid_class_prf1.png")
    plot_class_support(class_rows, out_dir / "hybrid_class_support.png")
    plot_ovr_confusion(conf_rows, out_dir / "hybrid_ovr_confusion_matrix.png")

    training_runs = load_training_runs(training_path)
    if training_runs:
        plot_training_curves(training_runs, out_dir / "training_curves.png")

    print("Saved outputs:")
    for p in sorted(out_dir.glob("*")):
        print(f" - {p}")


if __name__ == "__main__":
    main()
