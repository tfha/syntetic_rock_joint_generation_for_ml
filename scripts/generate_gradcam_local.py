"""
Generate GradCAM visualizations locally from a trained model.

Usage:
    poetry run python scripts/generate_gradcam_local.py \
        --model-path path/to/best_model.pth \
        --images-dir path/to/images \
        --masks-dir path/to/masks \
        --output-dir outputs/gradcam_local \
        --num-samples 20

This is easier than debugging GradCAM in Azure ML jobs.
You can run it after downloading job artifacts.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ml_segmentation.data_loading import SegmentationDataset
from ml_segmentation.models import create_model
from ml_segmentation.plotting.gradcam import generate_gradcam_visualizations
from ml_segmentation.transforms import get_transforms


def main():
    parser = argparse.ArgumentParser(
        description="Generate GradCAM visualizations locally"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to trained model checkpoint (e.g., best_model.pth)",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing validation/test images",
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        required=True,
        help="Directory containing validation/test masks",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/gradcam_local"),
        help="Output directory for GradCAM visualizations",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of samples to visualize",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="unet",
        help="Model architecture (unet, unetplusplus, deeplabv3plus)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="resnet34",
        help="Encoder backbone",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Prediction threshold",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu or cuda)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=768,
        help="Image size for model input",
    )

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model_path}")
    print(f"Device: {device}")

    # Create model
    model = create_model(
        model_name=args.model_name,
        encoder_name=args.encoder,
        encoder_weights=None,  # We're loading trained weights
        in_channels=3,
        classes=1,
    )

    # Load trained weights
    checkpoint = torch.load(args.model_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print("Model loaded successfully")

    # Get image files
    image_files = sorted(args.images_dir.glob("*.png"))
    if not image_files:
        image_files = sorted(args.images_dir.glob("*.jpg"))

    if not image_files:
        print(f"ERROR: No images found in {args.images_dir}")
        return

    # Limit to num_samples
    image_files = list(image_files[: args.num_samples])
    print(f"Found {len(image_files)} images")

    # Create file list (just filenames)
    file_list = [img.name for img in image_files]

    # Create dataset
    transforms_dict = get_transforms(image_size=args.image_size, augment=False)
    dataset = SegmentationDataset(
        images_dir=args.images_dir,
        labels_dir=args.masks_dir,
        file_list=file_list,
        transform=transforms_dict,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process one at a time for visualization
        shuffle=False,
        num_workers=0,  # Single thread for simplicity
    )

    print(f"\n{'=' * 60}")
    print("Generating GradCAM Visualizations")
    print(f"{'=' * 60}\n")

    # Generate GradCAM
    try:
        generate_gradcam_visualizations(
            model=model,
            dataloader=dataloader,
            device=device,
            save_dir=args.output_dir,
            num_samples=args.num_samples,
            threshold=args.threshold,
        )
        print(f"\n✓ GradCAM visualizations saved to: {args.output_dir / 'gradcam'}")

    except Exception as e:
        print(f"\nERROR: GradCAM generation failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
