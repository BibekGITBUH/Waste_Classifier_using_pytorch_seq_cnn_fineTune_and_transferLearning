"""
Loads a trained checkpoint and evaluates it on the held-out test set:
accuracy, precision, recall, F1-score (per-class + macro/weighted avg),
and a confusion matrix plot.

Usage:
    python evaluate.py --model custom_cnn
    python evaluate.py --model resnet50 --freeze-backbone
    python evaluate.py --model resnet50 --no-freeze-backbone
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

import config
from dataset import get_dataloaders
from train import build_model, get_run_name


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix plot to: {save_path}")


def evaluate_model(model_name: str = "custom_cnn", freeze_backbone: bool = True):
    device = config.DEVICE
    _, _, test_loader = get_dataloaders()

    run_name = get_run_name(model_name, freeze_backbone)

    model = build_model(model_name, freeze_backbone=freeze_backbone).to(device)
    ckpt_path = Path(config.CHECKPOINT_DIR) / f"{run_name}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}. Run train.py first.")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    y_true, y_pred = get_predictions(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(config.NUM_CLASSES), zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    print(f"\n=== {run_name} — Test Set Results ===")
    print(f"Accuracy: {acc:.4f}\n")
    print(f"{'Class':<12}{'Precision':<12}{'Recall':<12}{'F1':<12}{'Support':<10}")
    for i, cls in enumerate(config.CLASS_NAMES):
        print(f"{cls:<12}{precision[i]:<12.4f}{recall[i]:<12.4f}{f1[i]:<12.4f}{support[i]:<10}")
    print(f"\n{'Macro avg':<12}{precision_macro:<12.4f}{recall_macro:<12.4f}{f1_macro:<12.4f}")

    # Full sklearn report (includes weighted avg too), saved for the report deliverable
    report_dict = classification_report(
        y_true, y_pred, target_names=config.CLASS_NAMES, output_dict=True, zero_division=0
    )
    report_path = Path(config.OUTPUT_DIR) / f"{run_name}_classification_report.json"
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"\nSaved full classification report to: {report_path}")

    cm = confusion_matrix(y_true, y_pred, labels=range(config.NUM_CLASSES))
    cm_plot_path = Path(config.OUTPUT_DIR) / f"{run_name}_confusion_matrix.png"
    plot_confusion_matrix(cm, config.CLASS_NAMES, cm_plot_path)

    return {
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="custom_cnn",
        choices=["custom_cnn", "resnet50"],
        help="Which trained model to evaluate",
    )
    parser.add_argument(
        "--freeze-backbone", dest="freeze_backbone", action="store_true",
        help="Evaluate the frozen-backbone resnet50 checkpoint (resnet50 only). Default.",
    )
    parser.add_argument(
        "--no-freeze-backbone", dest="freeze_backbone", action="store_false",
        help="Evaluate the fine-tuned resnet50 checkpoint (resnet50 only).",
    )
    parser.set_defaults(freeze_backbone=True)
    args = parser.parse_args()
    evaluate_model(args.model, freeze_backbone=args.freeze_backbone)
