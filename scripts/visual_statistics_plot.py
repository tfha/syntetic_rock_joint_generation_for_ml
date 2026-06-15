"""
Visual statistics plotting and summary visualization.

Reads CSV files from visual_statistics_extract and creates:
  - Comparison grid of boxplots (Larvik vs Rv4, one row per feature)
  - Excel workbooks with summary statistics

No statistical comparisons - just visualization of extracted features.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


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


def load_statistics_csv(csv_path: Path) -> pd.DataFrame:
    """Load statistics CSV file."""
    if not csv_path.exists():
        print(f"[WARNING] File not found: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path.name}")

    return df


def make_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics for a dataset."""
    if df.empty:
        return pd.DataFrame()

    dataset_name = df["Dataset"].iloc[0]

    # Get all feature columns (exclude metadata)
    metadata_cols = {"Dataset", "FileName", "FilePath"}
    feature_cols = [col for col in df.columns if col not in metadata_cols]

    rows = []
    for feature in feature_cols:
        x = df[feature].dropna()

        if len(x) == 0:
            continue

        rows.append(
            {
                "Dataset": dataset_name,
                "Feature": feature,
                "N": len(x),
                "Mean": x.mean(),
                "Std": x.std(),
                "Median": x.median(),
                "Q1": x.quantile(0.25),
                "Q3": x.quantile(0.75),
                "Min": x.min(),
                "Max": x.max(),
            }
        )

    return pd.DataFrame(rows)


def save_comparison_grid(datasets_dict: dict, output_folder: Path) -> None:
    """
    Generate comparison grid with categorized features in 3 columns.

    Layout: 3 columns (Colour, Texture, Lighting - right aligned) with overlaid
    violin and boxplots. Larvik on top, Rv4 below for each feature.

    Args:
        datasets_dict: Dict of {dataset_name: dataframe}
        output_folder: Path to save PNG/TIFF
    """
    if not datasets_dict or all(df.empty for df in datasets_dict.values()):
        return

    output_folder.mkdir(parents=True, exist_ok=True)

    print("Generating categorized comparison grid with violin and box plots...")

    # Define feature categories with feature names
    # Reordered: Colour, Texture, then Lighting on right
    categories = {
        "Colour": ["H_mean", "S_mean", "H_std", "S_std"],
        "Texture": ["Entropy", "Edge_density", "GLCM_contrast", "GLCM_homogeneity"],
        "Lighting": ["Luminance_mean", "Contrast_std"],
    }

    # Define theoretical ranges for each feature
    # These represent the physical/theoretical bounds for each metric
    theoretical_ranges = {
        # Hue: 0-360 degrees (OpenCV H: 0-179, scaled to 0-360)
        "H_mean": (0, 360),
        "H_std": (0, 180),
        # Saturation: 0-255 (OpenCV S: 0-255)
        "S_mean": (0, 255),
        "S_std": (0, 127),
        # Value: 0-255 (brightness in HSV)
        "V_mean": (0, 255),
        "V_std": (0, 127),
        # RGB values: 0-255
        "R_mean": (0, 255),
        "R_std": (0, 127),
        "G_mean": (0, 255),
        "G_std": (0, 127),
        "B_mean": (0, 255),
        "B_std": (0, 127),
        # Luminance: 0-255 (grayscale intensity)
        "Luminance_mean": (0, 255),
        # Contrast: 0-255 (standard deviation of luminance)
        "Contrast_std": (0, 255),
        # Entropy: 0-8 bits (log2(256) = 8 for 8-bit images)
        "Entropy": (0, 8),
        # Edge density: 0-1 (proportion of edge pixels)
        "Edge_density": (0, 1),
        # Sobel magnitude: 0-1 (normalized)
        "Sobel_mean": (0, 1),
        # GLCM metrics with 32 levels
        # contrast: (i-j)² weighted by probabilities, max ~(31)² = 961, but practically 0-25
        "GLCM_contrast": (0, 25),
        "GLCM_homogeneity": (0, 1),
        "GLCM_energy": (0, 1),
        "GLCM_correlation": (-1, 1),
    }

    # Create figure with 3 columns, proportional heights
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(
        4, 3, left=0.08, right=0.96, height_ratios=[1, 1, 1, 1], hspace=0.5, wspace=0.15
    )

    col_idx = 0

    def normalize_dataset_name(name: str) -> str:
        """Normalize dataset name for color mapping (Box_cardboard -> cardboard)."""
        return name.lower().replace("box_", "")

    color_map = {
        "cardboard": "lightgreen",
        "pattern": "gold",
        "larvik": "lightblue",
        "rv4": "lightcoral",
        "fracman": "#FF6B9D",  # Hot pink
        "benchmark": "#C9ADA7",  # Warm gray
    }

    # Display name mapping for legend
    display_name_map = {
        "Benchmark": "Synthetic box",
        "Box_cardboard": "Real-world cardboard box",
        "Box_pattern": "Real-world pattern box",
        "FracMan": "Synthetic DFN",
        "Larvik": "Real-world Larvik slope    ",
        "Rv4": "Real-world Rv4 slope",
    }

    # Store legend handles and labels for global legend
    legend_handles: list = []
    legend_labels: list[str] = []

    for category_name, features in categories.items():
        n_features = len(features)

        # Create subgridspec for this column with proportional height
        if category_name == "Lighting":
            sub_gs = gs[:2, col_idx]  # Align Lighting to top
        else:
            sub_gs = gs[:4, col_idx]

        # Create sub-gridspec for individual features in this column
        sub_gs_inner = sub_gs.subgridspec(n_features, 1, hspace=0.6)

        # Add category title (centered over each column)
        fig.text(
            0.20 + col_idx * 0.32,
            0.90,
            category_name,
            ha="center",
            fontsize=13,
            fontweight="bold",
        )

        for row, feature in enumerate(features):
            ax = fig.add_subplot(sub_gs_inner[row])

            # Collect data from both datasets
            all_data = {}
            for dataset_name in sorted(datasets_dict.keys()):
                df = datasets_dict[dataset_name]
                if df.empty or feature not in df.columns:
                    continue
                data = df[feature].dropna()
                if len(data) > 0:
                    all_data[dataset_name] = data

            if not all_data:
                ax.axis("off")
                continue

            # Use theoretical range for x-axis (consistent across all runs)
            if feature in theoretical_ranges:
                x_min, x_max = theoretical_ranges[feature]
            else:
                # Fallback to actual data range if feature not in theoretical ranges
                all_values = np.concatenate([data.values for data in all_data.values()])
                x_min = all_values.min()
                x_max = all_values.max()
                x_range = x_max - x_min if x_max > x_min else 1
                x_min = x_min - x_range * 0.05
                x_max = x_max + x_range * 0.05

            # Prepare data for seaborn violinplot
            plot_data = []
            dataset_list = sorted(all_data.keys())

            for dataset_name in dataset_list:
                if dataset_name not in all_data:
                    continue

                data = all_data[dataset_name]
                for value in data.values:
                    plot_data.append({"dataset": dataset_name, "value": value})

            if plot_data:
                df_plot = pd.DataFrame(plot_data)

                # Create color palette (normalize dataset names)
                palette = {
                    dataset: color_map.get(normalize_dataset_name(dataset), "gray")
                    for dataset in dataset_list
                }

                # Plot using seaborn violinplot with box inside
                sns.violinplot(
                    data=df_plot,
                    x="value",
                    y="dataset",
                    hue="dataset",
                    ax=ax,
                    palette=palette,
                    inner="box",
                    linewidth=0,
                    legend=False,
                    linecolor="k",
                    inner_kws={"box_width": 3, "whis_width": 1, "color": "black"},
                )

                # # Customize box properties (thick black outline, white median)
                # for patch in ax.artists:
                #     patch.set_alpha(0.6)
                #     patch.set_edgecolor("black")
                #     patch.set_linewidth(0.5)

                # # Style the box plot elements (median line only)
                # for line in ax.get_lines():
                #     if line.get_linestyle() == '-':  # Median line
                #         line.set_color("white")
                #         line.set_linewidth(3)

                # Create legend entries
                for dataset_name in dataset_list:
                    if not any(label == dataset_name for label in legend_labels):
                        legend_handles.append(
                            plt.Rectangle(
                                (0, 0), 1, 1, fc=palette[dataset_name], alpha=0.7
                            )
                        )
                        legend_labels.append(dataset_name)  # Store original name as key

            # Set x-axis to calculated range
            ax.set_xlim(x_min, x_max)

            # Remove y-axis labels (dataset names handled by seaborn)
            ax.set_ylabel("")

            # Set y-axis (remove labels to match original style)
            ax.set_yticks([])
            ax.set_yticklabels([])

            # Feature name as xlabel
            ax.set_xlabel(feature.replace("_", " "), fontsize=10, fontweight="bold")
            ax.grid(True, axis="x", alpha=0.3)

            # Add domain separator line at midpoint (box-domain / slope-domain boundary)
            y_min, y_max = ax.get_ylim()
            midpoint = (y_min + y_max) / 2
            ax.axhline(
                y=midpoint,
                color="gray",
                linestyle="-",
                linewidth=0.8,
                label="Domain Boundary",
            )

        col_idx += 1

    # Create separate legend boxes for Box-domain and Slope-domain
    box_domain_datasets = ["Benchmark", "Box_cardboard", "Box_pattern"]
    slope_domain_datasets = ["FracMan", "Larvik", "Rv4"]

    # Ensure all datasets are in legend (add missing ones)
    for dataset_name in sorted(datasets_dict.keys()):
        if not any(label == dataset_name for label in legend_labels):
            normalized_name = normalize_dataset_name(dataset_name)
            legend_handles.append(
                plt.Rectangle(
                    (0, 0), 1, 1, fc=color_map.get(normalized_name, "gray"), alpha=0.7
                )
            )
            legend_labels.append(dataset_name)

    # Collect handles and labels for Box-domain
    box_handles = []
    box_labels = []
    for dataset_name in box_domain_datasets:
        for i, label in enumerate(legend_labels):
            if label == dataset_name:
                box_handles.append(legend_handles[i])
                box_labels.append(display_name_map.get(dataset_name, dataset_name))
                break

    # Collect handles and labels for Slope-domain
    slope_handles = []
    slope_labels = []
    for dataset_name in slope_domain_datasets:
        for i, label in enumerate(legend_labels):
            if label == dataset_name:
                slope_handles.append(legend_handles[i])
                slope_labels.append(display_name_map.get(dataset_name, dataset_name))
                break

    legend_x_anchor = 0.8
    legend_y_anchor = 0.3

    # Add Box-domain legend
    if box_handles:
        fig.legend(
            box_handles,
            box_labels,
            loc="upper left",
            fontsize=11,
            framealpha=0.95,
            bbox_to_anchor=(legend_x_anchor, legend_y_anchor),
            frameon=True,
            title="Box-domain",
            title_fontsize=11,
        )

    # Add Slope-domain legend (positioned below Box-domain, same x-coordinate for alignment)
    if slope_handles:
        fig.legend(
            slope_handles,
            slope_labels,
            loc="upper left",
            fontsize=11,
            framealpha=0.95,
            bbox_to_anchor=(legend_x_anchor, legend_y_anchor - 0.1),
            frameon=True,
            title="Slope-domain",
            title_fontsize=11,
        )

    # Save as PNG and TIFF
    png_path = output_folder / "fig_visual_stats_violinplots.png"
    tif_path = output_folder / "fig_visual_stats_violinplots.tif"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(tif_path, dpi=300, bbox_inches="tight")
    print(f"Saved grid: {png_path.name}, {tif_path.name}")
    plt.close(fig)


# ================= MAIN =================


def main():
    # Load config from YAML first
    config = load_config_from_yaml()

    parser = argparse.ArgumentParser(description="Plot visual statistics results")
    parser.add_argument(
        "--stats-folder",
        type=Path,
        default=Path(config.get("output_folder", "outputs/visual_statistics")),
        help="Folder containing CSV statistics files (default from YAML or outputs/visual_statistics)",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=Path(config.get("output_folder", "outputs/visual_statistics")),
        help="Output folder for plots and Excel (default from YAML or outputs/visual_statistics)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Larvik", "Rv4"],
        help="Dataset names to plot (default: Larvik Rv4)",
    )

    args = parser.parse_args()

    stats_folder = Path(args.stats_folder)
    output_folder = Path(args.output_folder)

    print(f"[INFO] Statistics folder: {stats_folder}")
    print(f"[INFO] Output folder: {output_folder}")
    print(f"[INFO] Datasets: {args.datasets}")

    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("PLOTTING VISUAL STATISTICS")
    print(f"{'=' * 60}")

    all_summaries = []
    datasets_dict = {}

    # Process each dataset
    for dataset_name in args.datasets:
        print(f"\n{dataset_name}:")
        print("-" * 40)

        # Map dataset names to CSV files
        # "Box_cardboard" -> "cardboard_statistics.csv", "Box_pattern" -> "pattern_statistics.csv"
        csv_base_name = dataset_name.lower().replace("box_", "")
        csv_file = stats_folder / f"{csv_base_name}_statistics.csv"
        df = load_statistics_csv(csv_file)

        if df.empty:
            print(f"No data found for {dataset_name}\n")
            continue

        datasets_dict[dataset_name] = df

        # Create summary
        summary = make_summary_table(df)
        all_summaries.append(summary)

    # Generate comparison grid (all datasets side-by-side)
    if datasets_dict:
        save_comparison_grid(datasets_dict, output_folder)

    # Excel export (summary statistics only)
    if all_summaries:
        excel_file = output_folder / "visual_statistics_summary.xlsx"
        try:
            with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
                for _i, summary in enumerate(all_summaries):
                    dataset_name = summary["Dataset"].iloc[0]
                    summary.to_excel(writer, sheet_name=dataset_name, index=False)

            print(f"\nSaved Excel: {excel_file.name}")
        except PermissionError:
            print(
                f"\n[WARNING] Cannot write Excel file (file may be open): {excel_file.name}"
            )
            print("[INFO] Summary statistics available in CSV files instead")

    print(f"\n{'=' * 60}")
    print("PLOTTING COMPLETE")
    print(f"Results saved to: {output_folder}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
