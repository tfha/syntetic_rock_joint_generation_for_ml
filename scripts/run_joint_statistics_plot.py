"""
Simple runner for joint_statistics_plot_python.py.

Usage:
    python run_joint_statistics_plot.py

Configurable paths are set below.
"""

import sys
from pathlib import Path

# ================= CONFIGURATION =================

# CSV files location - set to your OneDrive path
CSV_DIR = Path(
    r"C:\Users\KYC\OneDrive - NGI\Documents\PhD\Paper - Synthetic rock joint generation for ML\Joint information\joint_statistics_outputs"
)

# Alternative: if CSV files are in the workspace
# CSV_DIR = Path(__file__).parent.parent / "data" / "joint_statistics_outputs"

# Output directory (will be created if doesn't exist)
OUTPUT_DIR = CSV_DIR  # or specify a different path

# Projection type
PROJECTION = "equalarea"  # "equalarea" or "equalangle"

# DPI for saved figures
DPI = 600

# ================= MAIN =================

if __name__ == "__main__":
    # Check CSV directory exists
    if not CSV_DIR.exists():
        print(f"[ERROR] CSV directory not found: {CSV_DIR}")
        print(
            "\nPlease update CSV_DIR in this script to point to your joint_statistics_outputs folder."
        )
        sys.exit(1)

    # Check for required CSV files
    required_files = [
        "joint_trace_orientations.csv",
        "joint_set_mean_orientations.csv",
        "joint_set_spacing.csv",
    ]

    missing = []
    for fname in required_files:
        if not (CSV_DIR / fname).exists():
            missing.append(fname)

    if missing:
        print(f"[ERROR] Missing CSV files in {CSV_DIR}:")
        for fname in missing:
            print(f"  - {fname}")
        sys.exit(1)

    print(f"[OK] Found required CSV files in: {CSV_DIR}\n")

    # Import and run
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))

    from joint_statistics_plot_python import (
        load_csv_data,
        plot_figure1_stereonets,
        plot_figure2_boxplots,
    )

    print("[INFO] Loading data...")
    trace_table, set_mean_table, spacing_table, trace_length_table = load_csv_data(
        CSV_DIR
    )

    print(f"[OK] Loaded {len(trace_table)} trace records")
    print(f"[OK] Loaded {len(set_mean_table)} joint set mean records")
    print(f"[OK] Loaded {len(spacing_table)} spacing records")
    if trace_length_table is not None:
        print(f"[OK] Loaded {len(trace_length_table)} trace length records\n")

    # Prepare output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Extract valid gen folders
    valid_gen = trace_table[trace_table["Dataset"].str.startswith("TNA_gen", na=False)][
        "Dataset"
    ].unique()
    valid_gen = sorted(valid_gen)

    print("[INFO] Creating figures...")
    plot_figure1_stereonets(
        trace_table,
        set_mean_table,
        valid_gen,
        OUTPUT_DIR,
        projection=PROJECTION,
        dpi=DPI,
    )

    plot_figure2_boxplots(trace_length_table, spacing_table, OUTPUT_DIR, dpi=DPI)

    print("\n[OK] All plots complete!")
    print(f"[INFO] Output directory: {OUTPUT_DIR}")
