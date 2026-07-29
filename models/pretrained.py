"""
Transfer-learning model(s) built on torchvision's pretrained weights.

Currently supports ResNet50. The backbone can either be frozen (only the
new classifier head trains) or fully fine-tuned (all layers train, usually
with a lower learning rate than a from-scratch model needs).
"""

import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def build_resnet50(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """
    Load ImageNet-pretrained ResNet50 and replace the final FC layer.

    Args:
        num_classes: number of output classes for the new head.
        freeze_backbone: if True, all conv layers are frozen (requires_grad=False)
            and only the new FC head is trainable. If False, every parameter
            stays trainable for full fine-tuning.
    """
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the classifier head. New layers are trainable by default
    # (requires_grad=True on creation), so this works whether or not the
    # backbone above was frozen.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import torch

    for freeze in (True, False):
        model = build_resnet50(num_classes=5, freeze_backbone=freeze)
        dummy = torch.randn(2, 3, 224, 224)
        out = model(dummy)
        mode = "frozen backbone" if freeze else "full fine-tune"
        print(f"[{mode}] Output shape: {out.shape}, "
              f"Trainable params: {count_trainable_parameters(model):,}")
