import concurrent.futures
from functools import partial
from pathlib import Path

from PIL import Image
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)


def process_single_image(file_path, output_dir, threshold):
    """Process a single image file."""
    img = Image.open(file_path).convert("L")  # Convert to grayscale
    binary_img = img.point(
        lambda p: 255 if p >= threshold else 0, mode="1"
    )  # Apply threshold
    output_path = output_dir / file_path.name  # Preserve the original filename
    binary_img.save(output_path)
    return file_path.name


def preprocess_masks(
    input_dir: Path,
    output_dir: Path,
    threshold: int = 128,
    limit: int | None = None,
    max_workers: int = None,
    resize_dims: tuple = None,
    normalize: bool = False,
    augment: bool = False,
):
    """
    Preprocess mask images to black and white (binary) using a threshold.

    Args:
        input_dir (Path): Path to the directory containing raw mask images.
        output_dir (Path): Path to the directory where processed masks will be saved.
        threshold (int): Threshold value for binarisation (default: 128). When the pixel value is greater than the threshold, it is set to 255 (white); otherwise, it is set to 0 (black).
        limit (int | None): Number of images to process. If None, processes all images.
        max_workers (int | None): Maximum number of worker processes. If None, uses the default value from concurrent.futures.
        resize_dims (tuple): Optional dimensions (width, height) to resize images to.
        normalize (bool): Whether to normalize pixel values.
        augment (bool): Whether to apply data augmentation.

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

    # Get all image files in the directory (scan once)
    image_files = list(
        file
        for file in input_dir.glob("*")
        if file.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    # Apply limit if necessary
    if limit is not None:
        image_files = image_files[:limit]

    total_files = len(image_files)

    # Create a partial function with fixed output_dir and threshold parameters
    process_func = partial(
        process_single_image, output_dir=output_dir, threshold=threshold
    )

    # Improved progress tracking with Rich progress display
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[green]Processing images...", total=total_files)

        # Process files in parallel
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            # Submit all tasks
            futures = [
                executor.submit(process_func, file_path) for file_path in image_files
            ]

            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()  # Get the result to check for exceptions
                    progress.update(task, advance=1)  # Update the progress bar
                except Exception as e:
                    # Find which file caused the exception
                    for i, f in enumerate(futures):
                        if f == future:
                            file_name = image_files[i].name
                            print(f"Error processing {file_name}: {e}")
                            break
                    progress.update(task, advance=1)  # Still advance the progress bar


# Example usage
if __name__ == "__main__":
    input_directory = Path("path/to/masks")
    output_directory = Path("path/to/processed_masks")

    # Test with the first 5 images
    preprocess_masks(input_directory, output_directory, threshold=128, limit=5)
