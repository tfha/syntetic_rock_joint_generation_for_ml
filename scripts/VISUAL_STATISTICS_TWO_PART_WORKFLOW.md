# Visual Statistics Analysis: Two-Part Workflow

This analysis is split into two independent scripts:

1. **`visual_statistics_extract.py`** - Feature extraction from images
2. **`visual_statistics_plot.py`** - Visualization and summary statistics

## Workflow

### Part 1: Extract Visual Statistics

Extract visual features (color, texture, edges) from all images and save to CSV.

```bash
cd scripts
poetry run python visual_statistics_extract.py
```

**Options:**
```bash
# Extract specific datasets
poetry run python visual_statistics_extract.py --datasets Larvik Rv4

# Custom image and output folders
poetry run python visual_statistics_extract.py \
  --image-folder "C:/path/to/images" \
  --output-folder "C:/path/to/output"
```

**Output:**
- `larvik_statistics.csv` - Per-image statistics for Larvik
- `rv4_statistics.csv` - Per-image statistics for Rv4

Columns include:
- **Metadata**: Dataset, FileName, FilePath
- **Color (RGB)**: R_mean, G_mean, B_mean, R_std, G_std, B_std
- **Color (HSV)**: H_mean, S_mean, V_mean, H_std, S_std, V_std
- **Luminance**: Luminance_mean, Contrast_std
- **Edges**: Entropy, Edge_density, Sobel_mean
- **Texture (GLCM)**: GLCM_contrast, GLCM_homogeneity, GLCM_energy, GLCM_correlation

### Part 2: Plot Results

Generate comparison boxplots from extracted CSV files.

```bash
poetry run python visual_statistics_plot.py
```

**Options:**
```bash
# Plot specific datasets
poetry run python visual_statistics_plot.py --datasets Larvik Rv4

# Custom folders
poetry run python visual_statistics_plot.py \
  --stats-folder "outputs/visual_statistics" \
  --output-folder "outputs/visual_statistics"
```

**Output:**
- `comparison_boxplots_grid.png` - Comparison grid with 3 categorized columns (Colour, Texture, Lighting): overlaid violin + boxplots, Larvik on top / Rv4 below per feature
- `comparison_boxplots_grid.tif` - Same grid in TIFF format (publication-ready, high quality)
- `visual_statistics_summary.xlsx` - Excel workbook with summary statistics per dataset

## Complete Workflow (One Command)

Run extraction and plotting sequentially:

```bash
cd scripts && \
poetry run python visual_statistics_extract.py && \
poetry run python visual_statistics_plot.py
```

## Configuration

Both scripts read from `scripts/config/visual_statistics.yaml`:

```yaml
# Data paths
image_folder: C:/Users/KYC/OneDrive - NGI/Documents/.../Data/rockmass
output_folder: C:/Users/KYC/OneDrive - NGI/Documents/.../Visual statistics analysis
```

Or override with command-line arguments (takes precedence over YAML).

## Key Features

### Extraction Script
- ✓ Automatic image discovery by prefix (Larvik, Rv4)
- ✓ Parallel-ready CSV output (one row per image)
- ✓ Progress reporting every 100 images
- ✓ Robust error handling (skips unreadable images)

### Plotting Script
- ✓ Reads pre-extracted CSV files
- ✓ Generates 3-column categorized comparison grid (Colour, Texture, Lighting)
- ✓ Overlaid violin plots (distribution) + boxplots (quartiles)
- ✓ Larvik on top, Rv4 below (stacked vertically per feature)
- ✓ Black median lines for clarity over colored boxes
- ✓ Absolute x-axis range (min/max across both datasets)
- ✓ Legend at bottom right
- ✓ Publication-ready PNG (300 dpi) and TIFF (300 dpi) outputs
- ✓ Excel workbook with summary statistics
- ✓ **No statistical comparisons** - visualization only

## Example Output Structure

```
outputs/visual_statistics/
├── larvik_statistics.csv            # Raw extraction output
├── rv4_statistics.csv
├── comparison_boxplots_grid.png     # Comparison grid (21 features, 2 datasets)
├── comparison_boxplots_grid.tif     # Same as above, TIFF format
└── visual_statistics_summary.xlsx   # Excel workbook with summaries
```

## Next Steps

After extraction and plotting, you can:

1. **Export comparison grid** - Use PNG/TIFF directly in presentations or manuscripts
2. **Statistical testing** - Use summary data for t-tests, effect sizes (separate script)
3. **Feature-specific analysis** - Filter CSV and re-extract for subsets
4. **Subset analysis** - Extract a subset of images and re-plot

## Troubleshooting

### No images found
- Check `image_folder` path in YAML (use forward slashes)
- Verify dataset prefixes match filenames

### Missing features in output
- Check that input images are readable (PNG/JPEG/TIFF)
- Ensure images aren't corrupted

### Plot generation errors
- Verify `output_folder` has write permissions
- Check matplotlib backend is properly configured

## Dependencies

- cv2 (OpenCV)
- numpy, pandas
- scikit-image (texture features)
- scipy (statistics)
- matplotlib (plotting)
- pyyaml (config loading)
