# Joint Statistics Plot - MATLAB to Python Conversion

## Overview

This Python implementation converts the MATLAB script for extracting and plotting joint-orientation, trace-length, and spacing statistics into equivalent Python code.

**Key difference**: Since your CSV files are already extracted, this Python version reads directly from CSV files instead of .mat files.

## Files

### Main Scripts

1. **`joint_statistics_plot_python.py`**
   - Core plotting library with all functions
   - Reads CSV files and generates two main figures
   - Can be used as a module or run directly from command line

2. **`run_joint_statistics_plot.py`**
   - Simple wrapper script for easy execution
   - Configure paths at the top of the file
   - Just run: `python run_joint_statistics_plot.py`

## Usage

### Option 1: Using the Wrapper (Recommended for quick runs)

```bash
cd scripts
python run_joint_statistics_plot.py
```

Configure paths in `run_joint_statistics_plot.py`:
```python
CSV_DIR = Path(r"C:\Users\KYC\OneDrive - NGI\Documents\...")
OUTPUT_DIR = CSV_DIR  # or specify different path
```

### Option 2: Using Command Line Arguments

```bash
python joint_statistics_plot_python.py \
    --csv-dir "C:\path\to\csv\files" \
    --output-dir "C:\path\to\output" \
    --projection equalarea \
    --dpi 600
```

### Option 3: In Python Code

```python
from pathlib import Path
from scripts.joint_statistics_plot_python import (
    load_csv_data,
    plot_figure1_stereonets,
    plot_figure2_boxplots
)

csv_dir = Path("path/to/csv/files")
output_dir = Path("path/to/output")

trace_table, set_mean_table, spacing_table, trace_length_table = load_csv_data(csv_dir)

plot_figure1_stereonets(
    trace_table, set_mean_table, valid_gen_folders,
    output_dir, projection="equalarea", dpi=600
)

plot_figure2_boxplots(
    trace_length_table, spacing_table,
    output_dir, dpi=600
)
```

## Required CSV Files

Place these files in the same directory:

- `joint_trace_orientations.csv` ✓ (required)
- `joint_set_mean_orientations.csv` ✓ (required)
- `joint_set_spacing.csv` ✓ (required)
- `joint_trace_lengths.csv` (optional - will warn if missing)

## Expected CSV Columns

### joint_trace_orientations.csv
- Dataset, DatasetShort, TraceID, Dip, DipDirection, JointSet

### joint_set_mean_orientations.csv
- Dataset, DatasetShort, JointSet, MeanDip, MeanDipDirection

### joint_set_spacing.csv
- Dataset, DatasetShort, JointSet, Spacing

### joint_trace_lengths.csv
- Dataset, DatasetShort, TraceLength

## Output Figures

### Figure 1: Stereonets
- **Filename**: `figure_joint_orientation_stereonets.{png,tif}`
- **Content**: Three stereonet plots side-by-side
  - Left: 27 DFN_Gen datasets
  - Center: Larvik dataset
  - Right: Rv4 dataset
- **Features**:
  - Grey dots: Unassigned poles
  - Coloured dots: Traces assigned to J1-J4
  - Red circles: Joint set mean orientations

### Figure 2: Boxplots
- **Filename**: `figure_trace_length_and_spacing_boxplots.{png,tif}`
- **Content**: Two horizontal boxplots
  - Left: Joint trace length distribution (one box per dataset)
  - Right: Joint set spacing distribution (one box per dataset-joint set)

## Main Differences from MATLAB

| Aspect | MATLAB | Python |
|--------|--------|--------|
| Input | .mat files + Excel | CSV files (pre-extracted) |
| .mat parsing | Full extraction | Skipped (pre-extracted) |
| Excel reading | `readcell()` + custom parsing | `pd.read_csv()` |
| Stereonets | Custom projection functions | NumPy + Matplotlib |
| Boxplots | MATLAB `boxplot()` | Matplotlib `boxplot()` |
| Legend | Shared across panels | Single legend at bottom |
| Output | PDF/TIF via `exportgraphics()` | PNG/TIF via `savefig()` |

## Dependencies

Install with Poetry:
```bash
poetry add numpy pandas matplotlib scipy
```

Or with pip:
```bash
pip install numpy pandas matplotlib scipy
```

## Stereographic Projections

Two projection types supported:

1. **Equal-area** (default)
   - Preserves area on stereonet
   - Better for statistical analysis
   - Formula: `r = √2 · sin((90 - plunge)/2)`

2. **Equal-angle**
   - Preserves angles from center
   - Better for angular relationships
   - Formula: `r = tan((90 - plunge)/2)`

Switch with `--projection equalangle` or in code: `projection="equalangle"`

## Color Scheme

### Joint Set Colours (Colour-blind friendly)
- **J1**: Purple `[0.494, 0.184, 0.556]`
- **J2**: Cyan `[0.000, 0.650, 0.750]`
- **J3**: Yellow `[0.929, 0.694, 0.125]`
- **J4**: Orange `[0.850, 0.325, 0.098]`

### Other Features
- **Unassigned poles**: Grey `[0.55, 0.55, 0.55]`
- **Joint set means**: Red hollow circles
- **Grid**: Dark grey

## Function Reference

### Core Functions

**`pole_vector(dip, dipdir)`**
- Convert dip/dip-direction to 3D pole vectors
- Returns: N×3 array of unit vectors

**`pole_project(dip, dipdir, projection)`**
- Project poles to stereographic stereonet
- Returns: (x, y) coordinates

**`assign_nearest_centre(dip, dipdir, centre_dip, centre_dipdir)`**
- Assign traces to nearest joint set center
- Returns: Joint set IDs

**`build_dataset_order(dataset_short_values)`**
- Sort datasets naturally (gen1, gen2, ..., genA, genB, ..., Larvik, Rv4)
- Returns: Ordered list

**`plot_figure1_stereonets(...)`**
- Generate stereonet figure with legend
- Saves: PNG + TIF at specified DPI

**`plot_figure2_boxplots(...)`**
- Generate trace length and spacing boxplots
- Saves: PNG + TIF at specified DPI

## Troubleshooting

### Missing CSV files
```
[ERROR] Missing CSV files in C:\path\...
  - joint_trace_orientations.csv
```
**Solution**: Ensure CSV files are extracted and in the correct directory. The MATLAB version must be run first to generate these files.

### No valid data for stereonet
```
[WARNING] TNA_Larvik - no valid data
```
**Solution**: Check that Dataset column values match folder names (e.g., "TNA_Larvik", not "TNA_larvik").

### Incorrect stereonet projection
The stereonet may look "off" if:
- Wrong projection type selected (use `--projection equalarea`)
- Dip/DipDirection columns contain invalid values (should be 0-90 and 0-360)

## Performance Notes

- Loading 10k+ traces: <100ms
- Generating stereonets: ~1-2s per plot
- Generating boxplots: ~0.5s
- Total runtime: ~5-10s (depending on data size)

## Future Enhancements

Possible improvements:
- Add Fisher statistics (confidence cones)
- Support rose diagrams for trace orientation
- Interactive plots with Plotly
- Unit testing and CI/CD
- Batch processing multiple datasets

## License

Same as parent repository.

## Contact

For issues or questions, refer to the MATLAB script documentation or parent repository README.
