from itertools import islice
from pathlib import Path

from PIL import Image
from rich.progress import track


def preprocess_masks(
    input_dir: Path, output_dir: Path, threshold: int = 128, limit: int | None = None
):
    """
    Preprocess mask images to black and white (binary) using a threshold.

    Args:
        input_dir (Path): Path to the directory containing raw mask images.
        output_dir (Path): Path to the directory where processed masks will be saved.
        threshold (int): Threshold value for binarisation (default: 128). When the pixel value is greater than the threshold, it is set to 255 (white); otherwise, it is set to 0 (black).
        limit (int | None): Number of images to process. If None, processes all images.

    The function reads all images in the `input_dir`, converts them to grayscale,
    applies a threshold to produce binary (black and white) masks, and saves them
    to the `output_dir` with the same filenames.

    Only image files with extensions `.png`, `.jpg`, or `.jpeg` are processed.

    Example:
        >>> from pathlib import Path
        >>> preprocess_masks(
        ...     input_dir=Path("path/to/masks"),
        ...     output_dir=Path("path/to/processed_masks"),
        ...     threshold=128,
        ...     limit=5
        ... )

    This will process only the first 5 valid image files in `path/to/masks`.

    Requirements:
        Install the `rich` library for the progress tracker:
        $ pip install rich
    """
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files in the directory
    image_files = (
        file
        for file in input_dir.glob("*")
        if file.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    # Calculate total number of files and pass it to `track`
    image_files = list(
        file
        for file in input_dir.glob("*")
        if file.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    total_files = len(image_files)
    limited_files = islice(image_files, limit) if limit is not None else image_files

    # Use track with the total count
    for file_path in track(
        limited_files, description="Processing images...", total=total_files
    ):
        img = Image.open(file_path).convert("L")  # Convert to grayscale
        binary_img = img.point(
            lambda p: 255 if p >= threshold else 0, mode="1"
        )  # Apply threshold. White is 255, black is 0
        output_path = output_dir / file_path.name  # Preserve the original filename
        binary_img.save(output_path)


# Example usage
if __name__ == "__main__":
    input_directory = Path("path/to/masks")
    output_directory = Path("path/to/processed_masks")

    # Test with the first 5 images
    preprocess_masks(input_directory, output_directory, threshold=128, limit=5)
