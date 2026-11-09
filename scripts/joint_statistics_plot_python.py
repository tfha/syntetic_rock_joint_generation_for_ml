"""
Extract and plot joint-orientation, trace-length and spacing statistics.

Python equivalent of MATLAB script. Reads pre-extracted CSV files and creates:
  - Stereonet plots for joint orientations
  - Boxplots for trace length and spacing distributions

Expected CSV files:
  - joint_trace_orientations.csv
  - joint_set_mean_orientations.csv
  - joint_set_spacing.csv
  - joint_trace_lengths.csv (optional)
"""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ================= CONFIGURATION =================


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot joint statistics from extracted CSV files"
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        required=True,
        help="Directory containing extracted CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=False,
        default=None,
        help="Output directory for figures (defaults to csv_dir)",
    )
    parser.add_argument(
        "--projection",
        choices=["equalarea", "equalangle"],
        default="equalarea",
        help="Stereographic projection type",
    )
    parser.add_argument("--dpi", type=int, default=600, help="Output resolution in DPI")

    return parser.parse_args()


# ================= DATA LOADING =================


def load_csv_data(
    csv_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load all CSV files from directory.

    Returns:
        (trace_table, set_mean_table, spacing_table, trace_length_table)
    """
    csv_dir = Path(csv_dir)

    # Load required CSVs
    trace_file = csv_dir / "joint_trace_orientations.csv"
    set_file = csv_dir / "joint_set_mean_orientations.csv"
    spacing_file = csv_dir / "joint_set_spacing.csv"

    if not trace_file.exists():
        raise FileNotFoundError(f"Missing: {trace_file}")
    if not set_file.exists():
        raise FileNotFoundError(f"Missing: {set_file}")
    if not spacing_file.exists():
        raise FileNotFoundError(f"Missing: {spacing_file}")

    trace_table = pd.read_csv(trace_file)
    set_mean_table = pd.read_csv(set_file)
    spacing_table = pd.read_csv(spacing_file)

    # Load optional trace length CSV
    trace_length_file = csv_dir / "joint_trace_lengths.csv"
    trace_length_table = None
    if trace_length_file.exists():
        trace_length_table = pd.read_csv(trace_length_file)
    else:
        print(f"[WARNING] Trace length file not found: {trace_length_file}")

    return trace_table, set_mean_table, spacing_table, trace_length_table


# ================= UTILITY FUNCTIONS =================


def pole_vector(dip: np.ndarray, dipdir: np.ndarray) -> np.ndarray:
    """
    Convert dip and dip-direction to pole vectors.

    Args:
        dip: Dip angles in degrees
        dipdir: Dip directions in degrees

    Returns:
        N x 3 array of unit pole vectors [x, y, z]
    """
    dip = np.atleast_1d(np.asarray(dip, dtype=float))
    dipdir = np.atleast_1d(np.asarray(dipdir, dtype=float))

    pole_trend = np.mod(dipdir + 180, 360)
    pole_plunge = 90 - dip

    x = np.cos(np.radians(pole_plunge)) * np.sin(np.radians(pole_trend))
    y = np.cos(np.radians(pole_plunge)) * np.cos(np.radians(pole_trend))
    z = -np.sin(np.radians(pole_plunge))

    V = np.column_stack([x, y, z])
    V = V / np.linalg.norm(V, axis=1, keepdims=True)

    return V


def pole_project(
    dip: np.ndarray, dipdir: np.ndarray, projection: str = "equalarea"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project poles to lower hemisphere using stereonet projection.

    Args:
        dip: Dip angles in degrees
        dipdir: Dip directions in degrees
        projection: "equalarea" or "equalangle"

    Returns:
        (x, y) coordinates on stereonet
    """
    dip = np.atleast_1d(np.asarray(dip, dtype=float))
    dipdir = np.atleast_1d(np.asarray(dipdir, dtype=float))

    pole_trend = np.mod(dipdir + 180, 360)
    pole_plunge = 90 - dip

    if projection.lower() == "equalarea":
        r = np.sqrt(2) * np.sin(np.radians((90 - pole_plunge) / 2))
    elif projection.lower() == "equalangle":
        r = np.tan(np.radians((90 - pole_plunge) / 2))
    else:
        raise ValueError(f"Unknown projection: {projection}")

    x = r * np.sin(np.radians(pole_trend))
    y = r * np.cos(np.radians(pole_trend))

    return x, y


def stereo_radius(dip: float, projection: str = "equalarea") -> float:
    """Calculate stereonet radius for a given dip angle."""
    dip_rad = np.radians(dip)

    if projection.lower() == "equalarea":
        return np.sqrt(2) * np.sin(dip_rad / 2)
    elif projection.lower() == "equalangle":
        return np.tan(dip_rad / 2)
    else:
        raise ValueError(f"Unknown projection: {projection}")


def assign_nearest_centre(
    dip: np.ndarray,
    dipdir: np.ndarray,
    centre_dip: np.ndarray,
    centre_dipdir: np.ndarray,
) -> np.ndarray:
    """Assign each trace to nearest joint set centre."""
    dip = np.atleast_1d(np.asarray(dip, dtype=float))
    dipdir = np.atleast_1d(np.asarray(dipdir, dtype=float))
    centre_dip = np.atleast_1d(np.asarray(centre_dip, dtype=float))
    centre_dipdir = np.atleast_1d(np.asarray(centre_dipdir, dtype=float))

    n_trace = len(dip)
    set_id = np.full(n_trace, np.nan)

    V = pole_vector(dip, dipdir)
    C = pole_vector(centre_dip, centre_dipdir)

    for i in range(n_trace):
        dots = np.abs(C @ V[i, :])
        set_id[i] = np.argmax(dots) + 1  # 1-indexed

    return set_id


def build_dataset_order(dataset_short_values: pd.Series) -> list:
    """
    Build a natural sort order for dataset names.

    Order: gen1, gen2, ..., genA, genB, ..., Larvik, Rv4
    """
    unique_ds = dataset_short_values.unique()
    unique_ds = np.array([str(x) for x in unique_ds])

    gen_numeric = []
    gen_numeric_num = []
    gen_alpha = []

    for ds in unique_ds:
        ds_lower = ds.lower()
        if ds_lower.startswith("gen"):
            suffix = ds_lower[3:]
            try:
                num = int(suffix)
                gen_numeric.append(ds)
                gen_numeric_num.append(num)
            except ValueError:
                gen_alpha.append(ds)

    # Sort gen numeric
    if gen_numeric_num:
        idx_sort = np.argsort(gen_numeric_num)
        gen_numeric = [gen_numeric[i] for i in idx_sort]

    # Sort gen alpha
    gen_alpha = sorted(gen_alpha)

    # Build final order
    ordered = gen_numeric + gen_alpha

    if "Larvik" in unique_ds or "larvik" in [x.lower() for x in unique_ds]:
        ordered.append("Larvik")
    if "Rv4" in unique_ds or "rv4" in [x.lower() for x in unique_ds]:
        ordered.append("Rv4")

    return ordered


def dataset_label(dataset_short: str) -> str:
    """Convert dataset short name to label (e.g., 'gen1' -> 'DFN_Gen1')."""
    ds = str(dataset_short)
    if ds.lower().startswith("gen"):
        suffix = ds[3:]
        return f"DFN_Gen{suffix}"
    else:
        return ds


# ================= PLOTTING FUNCTIONS =================


def draw_stereo_grid(
    ax, projection: str = "equalarea", gridcolor: str | list[float] | None = None
):
    """Draw stereographic projection grid on axes."""
    if gridcolor is None:
        gridcolor = [0.45, 0.45, 0.45]

    th = np.linspace(0, 2 * np.pi, 721)

    # Primitive circle
    ax.plot(np.cos(th), np.sin(th), "k-", linewidth=1.3)

    # Dip-angle rings
    for dip in range(15, 90, 15):
        r = stereo_radius(dip, projection)
        ax.plot(r * np.cos(th), r * np.sin(th), ":", color=gridcolor, linewidth=0.8)

    # Radial bearing lines
    for az in range(0, 360, 30):
        az_rad = np.radians(az)
        ax.plot(
            [0, np.sin(az_rad)],
            [0, np.cos(az_rad)],
            ":",
            color=gridcolor,
            linewidth=0.8,
        )

    # Main N-S and E-W axes
    ax.plot([0, 0], [-1, 1], "k-.", linewidth=1.0)
    ax.plot([-1, 1], [0, 0], "k-.", linewidth=1.0)

    # Bearing labels and tick marks
    for az in range(0, 360, 30):
        az_rad = np.radians(az)
        x1 = np.sin(az_rad)
        y1 = np.cos(az_rad)

        # Tick mark
        ax.plot([0.97 * x1, x1], [0.97 * y1, y1], "k-", linewidth=1.0)

        # Bearing label
        label_radius = 1.1
        x_label = label_radius * np.sin(az_rad)
        y_label = label_radius * np.cos(az_rad)

        ax.text(
            x_label,
            y_label,
            f"{az:d}",
            ha="center",
            va="center",
            fontname="Arial",
            fontsize=10,
            color="k",
        )

    # Dip-angle labels above east axis
    for dip in range(15, 91, 15):
        r = stereo_radius(dip, projection)
        ax.text(
            r,
            0.035,
            f"{dip:d}",
            ha="center",
            va="bottom",
            fontname="Arial",
            fontsize=9,
            color="k",
            bbox={
                "boxstyle": "round,pad=0.01",
                "facecolor": "white",
                "edgecolor": "none",
            },
        )

    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.18, 1.18)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_stereo_group(
    ax,
    trace_table: pd.DataFrame,
    set_mean_table: pd.DataFrame,
    folders: list,
    title_text: str,
    projection: str = "equalarea",
    mean_circle_fill: str = "none",
    show_labels: bool = False,
):
    """Plot stereonet for a group of datasets.

    Args:
        mean_circle_fill: Color for mean orientation circles ("none" or color name)
        show_labels: Whether to show J1, J2, J3 labels next to mean circles
    """

    # Colour-blind friendly colours
    cb_colours = np.array(
        [
            [0.4940, 0.1840, 0.5560],  # purple
            [0.0000, 0.6500, 0.7500],  # cyan
            [0.9290, 0.6940, 0.1250],  # yellow
            [0.8500, 0.3250, 0.0980],  # orange
            [0.0000, 0.4470, 0.7410],  # blue
            [0.4660, 0.6740, 0.1880],  # green
            [0.6350, 0.0780, 0.1840],  # dark red
        ]
    )

    # Filter data for folders
    folders_str = [str(f) for f in folders]
    T = trace_table[trace_table["Dataset"].isin(folders_str)].copy()
    M = set_mean_table[set_mean_table["Dataset"].isin(folders_str)].copy()

    if len(T) == 0 and len(M) == 0:
        ax.text(
            0.5,
            0.5,
            f"{title_text} - no valid data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontname="Arial",
            fontweight="bold",
            fontsize=13,
        )
        ax.axis("off")
        return

    # All trace poles (grey, unassigned)
    valid_trace = T["Dip"].notna() & T["DipDirection"].notna()
    if valid_trace.any():
        x_all, y_all = pole_project(
            T.loc[valid_trace, "Dip"].values,
            T.loc[valid_trace, "DipDirection"].values,
            projection,
        )
        ax.scatter(x_all, y_all, s=12, color=[0.55, 0.55, 0.55], alpha=0.25, zorder=1)

    # Assigned traces by joint set
    T_assigned = T[
        (T["JointSet"].notna()) & (T["JointSet"] >= 1) & (T["JointSet"] % 1 == 0)
    ].copy()

    M_assigned = M[
        (M["JointSet"].notna()) & (M["JointSet"] >= 1) & (M["JointSet"] % 1 == 0)
    ].copy()

    joint_set_ids = np.unique(
        np.concatenate([T_assigned["JointSet"].values, M_assigned["JointSet"].values])
    )

    # Plot traces by joint set
    for js in joint_set_ids:
        colour = cb_colours[int((js - 1) % len(cb_colours))]

        idx = (
            (T_assigned["JointSet"] == js)
            & (T_assigned["Dip"].notna())
            & (T_assigned["DipDirection"].notna())
        )

        if idx.any():
            x, y = pole_project(
                T_assigned.loc[idx, "Dip"].values,
                T_assigned.loc[idx, "DipDirection"].values,
                projection,
            )
            ax.scatter(x, y, s=18, color=colour, alpha=0.55, zorder=2)

    # Mean joint-set orientations (red circles with labels)
    for _, row in M_assigned.iterrows():
        js = row["JointSet"]
        x, y = pole_project(
            np.array([row["MeanDip"]]), np.array([row["MeanDipDirection"]]), projection
        )

        ax.scatter(
            x,
            y,
            s=60,
            marker="o",
            edgecolor="r",
            facecolor=mean_circle_fill,
            linewidth=1.6,
            zorder=3,
        )

        # Show joint-set labels next to points (only if show_labels=True)
        if show_labels:
            # Alternate label position: odd (J1, J3) on left, even (J2, J4) on right
            if int(js) % 2 == 1:  # Odd: left side
                label_x = x[0] - 0.08
            else:  # Even: right side
                label_x = x[0] + 0.05
            label_y = y[0] + 0.025

            ax.text(
                label_x,
                label_y,
                f"J{int(js)}",
                fontname="Arial",
                fontsize=11,
                fontweight="bold",
                color="k",
            )

    # Draw grid on top
    draw_stereo_grid(ax, projection)

    # Title
    ax.set_title(
        f"{title_text}: {len(T)} planes",
        fontname="Arial",
        fontweight="bold",
        fontsize=13,
    )


def plot_figure1_stereonets(
    trace_table: pd.DataFrame,
    set_mean_table: pd.DataFrame,
    valid_gen_folders: list,
    output_dir: Path,
    projection: str = "equalarea",
    dpi: int = 600,
):
    """Create Figure 1: stereonets for different dataset groups."""

    fig = plt.figure(figsize=(15, 6.2), facecolor="white")
    gs = fig.add_gridspec(1, 3, hspace=0.05, wspace=0.05)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    plot_stereo_group(
        ax1,
        trace_table,
        set_mean_table,
        valid_gen_folders,
        "27 DFN_Gen",
        projection,
        mean_circle_fill="white",
        show_labels=False,
    )
    plot_stereo_group(
        ax2,
        trace_table,
        set_mean_table,
        ["TNA_Larvik"],
        "Larvik",
        projection,
        mean_circle_fill="none",
        show_labels=True,
    )
    plot_stereo_group(
        ax3,
        trace_table,
        set_mean_table,
        ["TNA_Rv4"],
        "Rv4",
        projection,
        mean_circle_fill="none",
        show_labels=True,
    )

    # Create legend with dummy handles
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=[0.55, 0.55, 0.55],
            markersize=2 * np.sqrt(12),
            label="Unassigned poles",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=[0.4940, 0.1840, 0.5560],
            markersize=2 * np.sqrt(18),
            label="J1",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=[0.0000, 0.6500, 0.7500],
            markersize=2 * np.sqrt(18),
            label="J2",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=[0.9290, 0.6940, 0.1250],
            markersize=2 * np.sqrt(18),
            label="J3",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=[0.8500, 0.3250, 0.0980],
            markersize=2 * np.sqrt(18),
            label="J4",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markeredgecolor="r",
            markerfacecolor="none",
            markersize=2 * np.sqrt(60),
            linewidth=1.6,
            label="Joint set mean",
        ),
    ]

    # Legend position can be adjusted here:
    # bbox_to_anchor=(0.5, y_position) where y_position controls vertical position
    # Increase y_position to move legend up (e.g., 0.05, 0.10, 0.15, etc.)
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.5, 0.1),
        handletextpad=0.4,
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    # Save
    out_tif = output_dir / "figure_joint_orientation_stereonets.tif"
    out_png = output_dir / "figure_joint_orientation_stereonets.png"

    fig.savefig(out_tif, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")

    print(f"[OK] Saved stereonet figure: {out_tif.name}")
    plt.close(fig)


def plot_figure2_boxplots(
    trace_length_table: Optional[pd.DataFrame],
    spacing_table: pd.DataFrame,
    output_dir: Path,
    dpi: int = 600,
):
    """Create Figure 2: box plots for trace length and spacing.

    Box plot features:
    - Central line: Median
    - Box boundaries: Interquartile range (Q1 to Q3)
    - Whiskers: Extend to minimum and maximum observed values
    - Cap lines: Whisker endpoints
    """
    import seaborn as sns

    if trace_length_table is None or trace_length_table.empty:
        print("[WARNING] Trace-length table is empty. Skipping boxplot figure.")
        return

    # Filter valid data
    valid_spacing = spacing_table[spacing_table["Spacing"].notna()].copy()
    valid_trace = trace_length_table[trace_length_table["TraceLength"].notna()].copy()

    if valid_trace.empty:
        print("[WARNING] No valid trace length data. Skipping boxplot figure.")
        return

    # Build dataset order
    all_datasets = pd.concat(
        [valid_trace["DatasetShort"], valid_spacing["DatasetShort"]]
    )
    dataset_order = build_dataset_order(all_datasets)

    # Prepare trace length data with labels
    trace_data_list = []
    for ds in dataset_order:
        ds_data = valid_trace[valid_trace["DatasetShort"].astype(str) == str(ds)]
        if not ds_data.empty:
            label = dataset_label(ds)
            for val in ds_data["TraceLength"].values:
                trace_data_list.append({"Dataset": label, "TraceLength": val})

    trace_df_plot = pd.DataFrame(trace_data_list)

    # Prepare spacing data with labels - grouped by dataset, then by joint set
    spacing_data_list = []
    for ds in dataset_order:
        ds_data = valid_spacing[valid_spacing["DatasetShort"].astype(str) == str(ds)]
        if not ds_data.empty:
            joint_sets = sorted(ds_data["JointSet"].unique())
            for js in joint_sets:
                label = f"{dataset_label(ds)}-J{int(js)}"
                for val in ds_data[ds_data["JointSet"] == js]["Spacing"].values:
                    spacing_data_list.append({"DatasetJointSet": label, "Spacing": val})

    # Build forward order for spacing: dataset1-J1, dataset1-J2, ..., dataset2-J1, ...

    spacing_df_plot = pd.DataFrame(spacing_data_list)

    # Build forward order for spacing
    spacing_order_forward = []
    for ds in dataset_order:
        ds_label = dataset_label(ds)
        for js in [1, 2, 3, 4]:
            label = f"{ds_label}-J{js}"
            if label in spacing_df_plot["DatasetJointSet"].values:
                spacing_order_forward.append(label)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 14), facecolor="white")

    # LEFT: Trace length boxplot
    # Forward order: DFN_Gen1 at top
    dataset_order_forward = list(trace_df_plot["Dataset"].unique())

    # Create color palette
    trace_palette = dict.fromkeys(dataset_order_forward, "lightblue")

    sns.boxplot(
        data=trace_df_plot,
        y="Dataset",
        x="TraceLength",
        ax=ax1,
        order=dataset_order_forward,
        hue="Dataset",
        palette=trace_palette,
        legend=False,
        width=0.6,
        linewidth=1.2,
        whis=1.5,
        showcaps=True,
        boxprops={"linewidth": 1.2, "alpha": 0.7},
        whiskerprops={"linewidth": 1.2},
        medianprops={"linewidth": 2, "color": "darkblue"},
        capprops={"linewidth": 1.2},
        flierprops={
            "marker": "o",
            "markersize": 5,
            "markerfacecolor": "darkblue",
            "markeredgecolor": "darkblue",
            "alpha": 0.3,
        },
    )

    ax1.set_xlabel("Trace length (m)", fontname="Arial", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Dataset", fontname="Arial", fontsize=14, fontweight="bold")
    ax1.set_title(
        "Joint trace length distribution",
        fontname="Arial",
        fontweight="bold",
        fontsize=14,
    )

    # X-axis limits
    max_trace = valid_trace["TraceLength"].max()
    x_max = int(np.ceil(max_trace / 5) * 5)
    x_max = max(x_max, 10)
    ax1.set_xlim(0, x_max)
    # Major ticks (labels) every 50m, minor grid every 10m
    ax1.set_xticks(range(0, x_max + 1, 50))
    ax1.set_xticks(range(0, x_max + 1, 10), minor=True)
    ax1.grid(
        True, alpha=0.3, axis="x", which="major", linewidth=1.0
    )  # Major grid at 50m
    ax1.grid(
        True, alpha=0.2, axis="x", which="minor", linewidth=0.5
    )  # Minor grid at 10m
    ax1.set_axisbelow(True)
    ax1.tick_params(labelsize=12)

    # RIGHT: Spacing boxplot
    # Forward order: grouped by dataset, then by joint set
    spacing_palette = dict.fromkeys(spacing_order_forward, "lightcoral")

    sns.boxplot(
        data=spacing_df_plot,
        y="DatasetJointSet",
        x="Spacing",
        ax=ax2,
        order=spacing_order_forward,
        hue="DatasetJointSet",
        palette=spacing_palette,
        legend=False,
        width=0.6,
        linewidth=1.2,
        whis=1.5,
        showcaps=True,
        boxprops={"linewidth": 1.2, "alpha": 0.7},
        whiskerprops={"linewidth": 1.2},
        medianprops={"linewidth": 2, "color": "darkred"},
        capprops={"linewidth": 1.2},
        flierprops={
            "marker": "o",
            "markersize": 5,
            "markerfacecolor": "darkred",
            "markeredgecolor": "darkred",
            "alpha": 0.3,
        },
    )

    ax2.set_xlabel(
        "Joint spacing (m)", fontname="Arial", fontsize=14, fontweight="bold"
    )
    ax2.set_ylabel(
        "Dataset-Joint Set", fontname="Arial", fontsize=14, fontweight="bold"
    )
    ax2.set_title(
        "Joint spacing distribution per joint set",
        fontname="Arial",
        fontweight="bold",
        fontsize=14,
    )

    ax2.set_xlim(0, 50)
    ax2.set_xticks(range(0, 51, 10))  # Major ticks every 10m
    ax2.set_xticks(range(0, 51, 2), minor=True)  # Minor grid every 2m
    ax2.grid(
        True, alpha=0.3, axis="x", which="major", linewidth=1.0
    )  # Major grid at 10m
    ax2.grid(
        True, alpha=0.2, axis="x", which="minor", linewidth=0.5
    )  # Minor grid at 2m
    ax2.set_axisbelow(True)
    ax2.tick_params(labelsize=12)

    plt.tight_layout()

    # Save
    out_png = output_dir / "figure_trace_length_and_spacing_boxplots.png"
    out_tif = output_dir / "figure_trace_length_and_spacing_boxplots.tif"

    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    print(f"[OK] Saved boxplot figure: {out_png.name}")

    try:
        fig.savefig(out_tif, dpi=dpi, bbox_inches="tight")
        print(f"[OK] Saved boxplot figure: {out_tif.name}")
    except Exception as e:
        print(f"[WARNING] Could not save TIF: {e}")

    plt.close(fig)


# ================= MAIN =================


def main():
    """Main entry point."""
    args = parse_args()

    csv_dir = Path(args.csv_dir).resolve()
    output_dir = Path(args.output_dir or csv_dir).resolve()

    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Loading CSV data from: {csv_dir}")
    trace_table, set_mean_table, spacing_table, trace_length_table = load_csv_data(
        csv_dir
    )

    print(f"[OK] Loaded {len(trace_table)} trace records")
    print(f"[OK] Loaded {len(set_mean_table)} joint set mean records")
    print(f"[OK] Loaded {len(spacing_table)} spacing records")
    if trace_length_table is not None:
        print(f"[OK] Loaded {len(trace_length_table)} trace length records")

    # Extract valid gen folders (for legend grouping)
    valid_gen = trace_table[trace_table["Dataset"].str.startswith("TNA_gen", na=False)][
        "Dataset"
    ].unique()
    valid_gen = sorted(valid_gen)

    # Create figures
    print("\n[INFO] Creating figures...")

    plot_figure1_stereonets(
        trace_table,
        set_mean_table,
        valid_gen,
        output_dir,
        projection=args.projection,
        dpi=args.dpi,
    )

    plot_figure2_boxplots(trace_length_table, spacing_table, output_dir, dpi=args.dpi)

    print("\n[OK] All plots generated successfully!")
    print(f"[INFO] Output directory: {output_dir}")


if __name__ == "__main__":
    main()
