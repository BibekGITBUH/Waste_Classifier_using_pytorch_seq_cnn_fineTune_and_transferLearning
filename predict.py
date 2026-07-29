"""
Predict waste class for images using a trained model (custom CNN or
ResNet50 transfer learning).

Usage:
    # Single image
    python predict.py --image path/to/image.jpg

    # Folder of images (supports .jpg, .jpeg, .png)
    python predict.py --folder path/to/images/
    python predict.py --folder path/to/images/ --model custom_cnn
    python predict.py --folder path/to/images/ --model resnet50 --no-freeze-backbone
"""

import argparse
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
import csv
from datetime import datetime

import config
from train import build_model, get_run_name


def get_model(model_name: str = "custom_cnn", freeze_backbone: bool = True):
    """Load and return the model."""
    device = config.DEVICE
    model = build_model(model_name, freeze_backbone=freeze_backbone).to(device)
    run_name = get_run_name(model_name, freeze_backbone)
    ckpt_path = Path(config.CHECKPOINT_DIR) / f"{run_name}_best.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model, device


def predict_single_image(image_path: str, model=None, device=None,
                          model_name: str = "custom_cnn", freeze_backbone: bool = True):
    """
    Predict class for a single image.

    Args:
        image_path: Path to the image file
        model: Pre-loaded model (optional)
        device: PyTorch device (optional)
        model_name: Name of the model (default: custom_cnn)
        freeze_backbone: Whether to load the frozen-backbone resnet50 checkpoint
            (only relevant if model is None and model_name == "resnet50")

    Returns:
        Tuple of (class_name, confidence_score, all_probabilities_dict)
    """
    if model is None or device is None:
        model, device = get_model(model_name, freeze_backbone)

    # Load and transform image
    if not Path(image_path).exists():
        return None, 0.0, None  # Return None for missing file

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return None, 0.0, None  # Return None for invalid image

    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = outputs.softmax(dim=1)[0]
        confidence, pred_idx = probs.max(dim=0)

    pred_class = config.CLASS_NAMES[pred_idx.item()]
    confidence_val = confidence.item()

    # Create probability dict
    probs_dict = {config.CLASS_NAMES[i]: probs[i].item() for i in range(len(config.CLASS_NAMES))}

    return pred_class, confidence_val, probs_dict


def predict_folder(folder_path: str, model_name: str = "custom_cnn", freeze_backbone: bool = True):
    """
    Predict classes for all images in a folder.

    Args:
        folder_path: Path to folder containing images
        model_name: Name of the model (default: custom_cnn)
        freeze_backbone: Whether to load the frozen-backbone resnet50 checkpoint

    Supported formats: .jpg, .jpeg, .png
    """
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found at {folder_path}")

    if not folder_path.is_dir():
        raise NotADirectoryError(f"{folder_path} is not a directory")

    # Load model once
    model, device = get_model(model_name, freeze_backbone)
    run_name = get_run_name(model_name, freeze_backbone)
    print(f"✓ Loaded model from: {Path(config.CHECKPOINT_DIR) / f'{run_name}_best.pt'}\n")

    # Find all supported image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = [f for f in folder_path.rglob('*') if f.suffix in image_extensions]

    if not image_files:
        print(f"❌ No images found in {folder_path}")
        return

    print(f"Found {len(image_files)} images. Processing...\n")
    print(f"{'Image Path':<60} {'Predicted':<12} {'Confidence':<12}")
    print("=" * 84)

    results = []

    for i, image_path in enumerate(image_files, 1):
        pred_class, confidence, probs = predict_single_image(
            image_path, model, device, model_name, freeze_backbone
        )

        if pred_class is None:
            print(f"{str(image_path):<60} {'ERROR':<12} {'---':<12}")
            results.append({
                'index': i,
                'image_path': str(image_path),
                'predicted_class': 'ERROR',
                'confidence': 0.0
            })
        else:
            rel_path = image_path.relative_to(folder_path)
            print(f"{str(rel_path):<60} {pred_class:<12} {confidence:.2%}")
            results.append({
                'index': i,
                'image_path': str(image_path),
                'predicted_class': pred_class,
                'confidence': confidence
            })

    print("=" * 84)
    print(f"\n✓ Processing complete! Total: {len(results)} images")

    # Save results to CSV
    csv_path = folder_path / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['index', 'image_path', 'predicted_class', 'confidence'])
        writer.writeheader()
        writer.writerows(results)

    print(f"✓ Results saved to: {csv_path}\n")

    # Summary statistics
    successful = [r for r in results if r['predicted_class'] != 'ERROR']
    if successful:
        class_counts = {}
        for r in successful:
            cls = r['predicted_class']
            class_counts[cls] = class_counts.get(cls, 0) + 1

        print("Summary by Class:")
        for cls in config.CLASS_NAMES:
            count = class_counts.get(cls, 0)
            print(f"  {cls:<12} {count:>4} images")

        avg_confidence = sum(r['confidence'] for r in successful) / len(successful)
        print(f"\nAverage Confidence: {avg_confidence:.2%}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict waste class for image(s)")
    parser.add_argument(
        "--image",
        type=str,
        help="Path to a single image file (.jpg, .jpeg, .png)"
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="Path to folder containing multiple images"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="custom_cnn",
        choices=["custom_cnn", "resnet50"],
        help="Which model to use (default: custom_cnn)"
    )
    parser.add_argument(
        "--freeze-backbone", dest="freeze_backbone", action="store_true",
        help="Use the frozen-backbone resnet50 checkpoint (resnet50 only). Default.",
    )
    parser.add_argument(
        "--no-freeze-backbone", dest="freeze_backbone", action="store_false",
        help="Use the fine-tuned resnet50 checkpoint (resnet50 only).",
    )
    parser.set_defaults(freeze_backbone=True)
    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.print_help()
        print("\n❌ Error: Provide either --image or --folder")
        exit(1)

    if args.image and args.folder:
        print("❌ Error: Provide either --image or --folder, not both")
        exit(1)

    try:
        if args.image:
            # Single image mode
            model, device = get_model(args.model, args.freeze_backbone)
            run_name = get_run_name(args.model, args.freeze_backbone)
            print(f"✓ Loaded model from: {Path(config.CHECKPOINT_DIR) / f'{run_name}_best.pt'}\n")

            if not Path(args.image).exists():
                raise FileNotFoundError(f"Image not found at {args.image}")

            pred_class, confidence, probs = predict_single_image(
                args.image, model, device, args.model, args.freeze_backbone
            )

            if pred_class is None:
                print(f"❌ Could not process image: {args.image}")
            else:
                print(f"{'='*50}")
                print(f"Image: {args.image}")
                print(f"Predicted Class: {pred_class}")
                print(f"Confidence: {confidence:.2%}")
                print(f"{'='*50}\n")

                print("Class Probabilities:")
                for cls in config.CLASS_NAMES:
                    prob = probs[cls]
                    bar = "█" * int(prob * 30)
                    print(f"  {cls:<12} {prob:.2%} {bar}")

        else:
            # Folder mode
            predict_folder(args.folder, args.model, args.freeze_backbone)

    except Exception as e:
        print(f"❌ Error: {e}")
