"""
Compares trained runs side by side using the history JSON (from train.py)
and classification report JSON (from evaluate.py) that each run saves
under outputs/. Run this after training + evaluating whichever models you
want to compare.

Usage:
    python compare_models.py
    python compare_models.py --runs custom_cnn resnet50_frozen resnet50_finetuned
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

import config


def load_run(run_name: str):
    out_dir = Path(config.OUTPUT_DIR)
    history_path = out_dir / f"{run_name}_history.json"
    report_path = out_dir / f"{run_name}_classification_report.json"

    history = json.loads(history_path.read_text()) if history_path.exists() else None
    report = json.loads(report_path.read_text()) if report_path.exists() else None

    return history, report


def compare(run_names):
    print(f"{'Run':<22}{'Test Acc':<12}{'Macro F1':<12}{'Weighted F1':<12}")
    print("-" * 58)

    plotted_any = False
    fig, ax = plt.subplots(figsize=(8, 5))

    for run_name in run_names:
        history, report = load_run(run_name)

        if report is None:
            print(f"{run_name:<22}{'(no evaluate.py results found)':<40}")
        else:
            acc = report["accuracy"]
            macro_f1 = report["macro avg"]["f1-score"]
            weighted_f1 = report["weighted avg"]["f1-score"]
            print(f"{run_name:<22}{acc:<12.4f}{macro_f1:<12.4f}{weighted_f1:<12.4f}")

        if history is not None:
            ax.plot(history["val_acc"], label=run_name)
            plotted_any = True
        else:
            print(f"  (no train.py history found for {run_name}, skipping in plot)")

    if plotted_any:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Accuracy")
        ax.set_title("Validation Accuracy by Model")
        ax.legend()
        fig.tight_layout()
        plot_path = Path(config.OUTPUT_DIR) / "model_comparison_val_acc.png"
        fig.savefig(plot_path, dpi=150)
        print(f"\nSaved comparison plot to: {plot_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs", type=str, nargs="+",
        default=["custom_cnn", "resnet50_frozen", "resnet50_finetuned"],
        help="Run names to compare (matches the {run_name}_history.json / "
             "{run_name}_classification_report.json files in outputs/)",
    )
    args = parser.parse_args()
    compare(args.runs)
