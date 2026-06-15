"""
Visual feature statistics extraction from images.

Extracts visual features (color, texture, edges) from images organized by dataset.
Saves per-image statistics to CSV for later analysis/plotting.

Output:
  - {dataset}_per_image_statistics.csv: Raw statistics for each image
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import sobel
from skimage.measure import shannon_entropy

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

logger = logging.getLogger(__name__)


# ================= CONFIGURATION LOADING =================


def load_config_from_yaml(yaml_path: Path | None = None) -> dict:
    """Load image_folder and output_folder from visual_statistics.yaml config."""
    if yaml is None:
        return {}

    if yaml_path is None:
        script_dir = Path(__file__).parent
        yaml_path = script_dir / "config" / "visual_statistics.yaml"

    yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        print(f"[INFO] Config file not found: {yaml_path}")
        return {}

    try:
        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        if config and isinstance(config, dict):
            result = {}
            if "image_folder" in config:
                result["image_folder"] = config["image_folder"]
            if "output_folder" in config:
                result["output_folder"] = config["output_folder"]

            if result:
                print(f"[INFO] Loaded config from: {yaml_path}")
            return result
    except Exception as e:
        print(f"[WARNING] Failed to load config from {yaml_path}: {e}")

    return {}


# ================= FUNCTIONS =================


def find_images_by_prefix(
    folder: Path, prefix: str, extensions: list[str] | None = None
) -> list[Path]:
    """Find images by filename prefix or partial match (case-insensitive)."""
    if extensions is None:
        extensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]

    files: list[Path] = []
    # Support both prefix matching and substring matching
    # This allows "cardboard" to match "Box-drone-cardboard-..."
    prefix_lower = prefix.lower()

    for ext in extensions:
        # First try prefix match
        files.extend(folder.glob(f"{prefix}*{ext}"))
        files.extend(folder.glob(f"{prefix}*{ext.upper()}"))

        # Then try substring match (case-insensitive)
        for img_file in folder.glob(f"*{ext}"):
            if prefix_lower in img_file.name.lower() and img_file not in files:
                files.append(img_file)

        for img_file in folder.glob(f"*{ext.upper()}"):
            if prefix_lower in img_file.name.lower() and img_file not in files:
                files.append(img_file)

    return sorted(set(files))


def compute_image_statistics(image_path: Path, dataset_name: str) -> dict:
    """
    Compute visual statistics for a single image.

    Returns dictionary with all features for this image.
    """
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb_float = img_rgb.astype(np.float32) / 255.0

    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    # Keep HSV in original ranges (H: 0-179, S: 0-255, V: 0-255)
    # Note: OpenCV uses H: 0-179 (not 360), we'll denormalize for clarity
    h_channel = img_hsv[:, :, 0] * 2  # Scale to 0-360 range
    s_channel = img_hsv[:, :, 1]  # Keep 0-255 range
    v_channel = img_hsv[:, :, 2]  # Keep 0-255 range

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_float = gray.astype(np.float32) / 255.0

    # Contrast as standard deviation of grayscale luminance (denormalized to 0-255)
    contrast_std = float(np.std(gray))

    # Entropy
    entropy_val = float(shannon_entropy(gray))

    # Edge density using Canny
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    edge_density = float(np.count_nonzero(edges) / edges.size)

    # Sobel mean
    sobel_img = sobel(gray_float)
    sobel_mean = float(np.mean(sobel_img))

    # GLCM texture
    glcm_levels = 32
    gray_quantised = np.floor(gray_float * (glcm_levels - 1)).astype(np.uint8)

    glcm = graycomatrix(
        gray_quantised,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=glcm_levels,
        symmetric=True,
        normed=True,
    )

    glcm_contrast = float(np.mean(graycoprops(glcm, "contrast")))
    glcm_homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))
    glcm_energy = float(np.mean(graycoprops(glcm, "energy")))
    glcm_correlation = float(np.mean(graycoprops(glcm, "correlation")))

    return {
        "Dataset": dataset_name,
        "FileName": image_path.name,
        "FilePath": str(image_path),
        # Color - RGB
        "R_mean": float(np.mean(img_rgb_float[:, :, 0])),
        "G_mean": float(np.mean(img_rgb_float[:, :, 1])),
        "B_mean": float(np.mean(img_rgb_float[:, :, 2])),
        "R_std": float(np.std(img_rgb_float[:, :, 0])),
        "G_std": float(np.std(img_rgb_float[:, :, 1])),
        "B_std": float(np.std(img_rgb_float[:, :, 2])),
        # Color - HSV (denormalized)
        "H_mean": float(np.mean(h_channel)),
        "S_mean": float(np.mean(s_channel)),
        "V_mean": float(np.mean(v_channel)),
        "H_std": float(np.std(h_channel)),
        "S_std": float(np.std(s_channel)),
        "V_std": float(np.std(v_channel)),
        # Luminance & Contrast (denormalized to 0-255)
        "Luminance_mean": float(np.mean(gray)),
        "Contrast_std": contrast_std,
        # Edges & Texture
        "Entropy": entropy_val,
        "Edge_density": edge_density,
        "Sobel_mean": sobel_mean,
        # GLCM
        "GLCM_contrast": glcm_contrast,
        "GLCM_homogeneity": glcm_homogeneity,
        "GLCM_energy": glcm_energy,
        "GLCM_correlation": glcm_correlation,
    }


def extract_dataset_statistics(
    image_folder: Path, prefix: str, dataset_name: str
) -> pd.DataFrame:
    """
    Extract statistics for all images in a dataset.

    Args:
        image_folder: Folder containing images
        prefix: Filename prefix to search for (e.g., "Larvik", "Rv4")
        dataset_name: Name for this dataset (e.g., "Larvik", "Rv4")

    Returns:
        DataFrame with one row per image
    """
    files = find_images_by_prefix(image_folder, prefix)

    if len(files) == 0:
        print(f"[WARNING] No images found for {dataset_name}")
        return pd.DataFrame()

    rows = []
    print(f"Processing {len(files)} {dataset_name} images...")

    for i, path in enumerate(files, start=1):
        try:
            rows.append(compute_image_statistics(path, dataset_name))
        except Exception as e:
            print(f"  [SKIP] {path.name}: {e}")

        if i % 100 == 0:
            print(f"  {dataset_name}: {i}/{len(files)}")

    if not rows:
        print(f"[WARNING] No statistics computed for {dataset_name}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    print(f"Successfully extracted {len(df)} {dataset_name} images")

    return df


# ================= MAIN =================


def main():
    # Load defaults from YAML config
    yaml_config = load_config_from_yaml()

    default_image_folder = yaml_config.get("image_folder", "data/raw")
    default_output_folder = yaml_config.get(
        "output_folder", "outputs/visual_statistics"
    )

    parser = argparse.ArgumentParser(
        description="Extract visual statistics from images"
    )
    parser.add_argument(
        "--image-folder",
        type=Path,
        default=Path(default_image_folder),
        help=f"Folder containing images (default: {default_image_folder})",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=Path(default_output_folder),
        help=f"Output folder for CSV files (default: {default_output_folder})",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Larvik", "Rv4"],
        help="Dataset prefixes to extract (default: Larvik Rv4)",
    )

    args = parser.parse_args()

    image_folder = Path(args.image_folder)
    output_folder = Path(args.output_folder)

    print(f"[INFO] Image folder: {image_folder}")
    print(f"[INFO] Output folder: {output_folder}")
    print(f"[INFO] Datasets: {args.datasets}")

    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("EXTRACTING VISUAL STATISTICS")
    print(f"{'=' * 60}")

    # Process each dataset
    for dataset_prefix in args.datasets:
        print(f"\n{dataset_prefix}:")
        print("-" * 40)

        df = extract_dataset_statistics(image_folder, dataset_prefix, dataset_prefix)

        if not df.empty:
            output_file = output_folder / f"{dataset_prefix.lower()}_statistics.csv"
            df.to_csv(output_file, index=False)
            print(f"Saved: {output_file.name}\n")
        else:
            print(f"No data extracted for {dataset_prefix}\n")

    print(f"\n{'=' * 60}")
    print("EXTRACTION COMPLETE")
    print(f"Statistics saved to: {output_folder}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
