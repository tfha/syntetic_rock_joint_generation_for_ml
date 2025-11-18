import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from omegaconf import OmegaConf
from PIL import Image
from rich.progress import track
from torch.utils.data import DataLoader, Dataset

from ml_segmentation.schema_config import ConfigSchema

if TYPE_CHECKING:
    # For type checkers only — do not import heavy optional dependency at runtime
    from torchvision import transforms  # type: ignore
else:
    try:
        from torchvision import transforms  # type: ignore
    except Exception:
        transforms = None  # type: ignore


class SegmentationDataset(Dataset):
    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        file_list: list[str],
        transform: dict[str, "transforms.Compose"] | None = None,
        return_original: bool = False,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.file_list = file_list
        self.transform = transform
        self.return_original = return_original

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(
        self, idx: int
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        image_path = self.images_dir / self.file_list[idx]
        label_path = self.labels_dir / self.file_list[idx]

        # Load image and label
        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path)

        # Store original image if requested
        if self.return_original:
            original_image = transforms.ToTensor()(image)

        # Apply transformations if any, else convert to tensors
        if self.transform is not None:
            # CRITICAL: Seed the random state to ensure geometric transforms
            # (RandomHorizontalFlip, RandomVerticalFlip, etc.) make the SAME
            # random decision for both image and label, maintaining alignment
            seed = torch.randint(0, 2**32, (1,)).item()

            # Apply image transforms with seeded random state
            torch.manual_seed(seed)
            image = self.transform["image"](image)

            # Apply label transforms with SAME seeded random state
            torch.manual_seed(seed)
            label = self.transform["label"](label)
        else:
            image = transforms.ToTensor()(image)
            # Convert label to tensor (grayscale -> 1-channel float tensor)
            label = transforms.ToTensor()(label)
        # Invert label to match the segmentation model's requirements
        label = 1 - label

        if self.return_original:
            return image, label, original_image
        return image, label


def get_datasets(
    images_dir: str | Path,
    labels_dir: str | Path,
    train_files: list[str],
    val_files: list[str],
    test_files: list[str],
    transform: dict[str, "transforms.Compose"] | None = None,
) -> tuple[SegmentationDataset, SegmentationDataset, SegmentationDataset]:
    train_dataset = SegmentationDataset(images_dir, labels_dir, train_files, transform)
    val_dataset = SegmentationDataset(images_dir, labels_dir, val_files, transform)
    test_dataset = SegmentationDataset(images_dir, labels_dir, test_files, transform)
    return train_dataset, val_dataset, test_dataset


def get_dataloaders(
    train_dataset: SegmentationDataset,
    val_dataset: SegmentationDataset,
    test_dataset: SegmentationDataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 2,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build DataLoaders with sensible defaults:
    - pin_memory True only when using CUDA (or if explicitly set)
    - persistent_workers only when num_workers > 0
    """
    if pin_memory is None:
        if device is not None:
            pin_memory = device.type == "cuda"
        else:
            pin_memory = torch.cuda.is_available()

    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader, test_loader


def get_transforms(
    optional_transforms: bool | dict[str, bool] = False,
    transforms_parameters: dict[str, Any] | None = None,
) -> dict[str, "transforms.Compose"]:
    """
    Get image and label transforms with optional augmentations.

    Using all the transforms, the effective virtual dataset size will be
    ~14.4x larger during training compared to the original 1000 images.
    While you still only have 1000 original images saved, the model will
    effectively see about 14,400 variations of your images during training,
    which significantly improves generalisation without increasing stored
    images.

    Args:
        optional_transforms: Either bool (legacy - enables ColorJitter only) or dict with individual flags:
            - horizontal_flip: Apply horizontal flipping
            - vertical_flip: Apply vertical flipping
            - color_jitter: Apply color jitter
            - rotation: Apply random rotation
            - gaussian_blur: Apply Gaussian blur
        transforms_parameters: Dict with transform parameters (e.g., crop_size)
    """
    # Read transform parameters (future-proof: add more keys as needed)
    params = transforms_parameters or {}
    crop_sz: int = int(params.get("crop_size", 768))

    # Parse transform flags (support bool, dict, or Pydantic model)
    if isinstance(optional_transforms, bool):
        # Legacy mode: bool=True enables only ColorJitter
        transform_flags = {
            "random_crop": False,
            "horizontal_flip": False,
            "vertical_flip": False,
            "color_jitter": optional_transforms,
            "rotation": False,
            "gaussian_blur": False,
        }
    elif isinstance(optional_transforms, dict):
        # Dict mode: individual control from dict
        transform_flags = optional_transforms
    else:
        # Pydantic model mode: convert to dict
        transform_flags = {
            "random_crop": optional_transforms.random_crop,
            "horizontal_flip": optional_transforms.horizontal_flip,
            "vertical_flip": optional_transforms.vertical_flip,
            "color_jitter": optional_transforms.color_jitter,
            "rotation": optional_transforms.rotation,
            "gaussian_blur": optional_transforms.gaussian_blur,
        }

    # Build transform lists for both image and label
    # CRITICAL: Geometric transforms must be applied to BOTH image and label
    # to maintain alignment
    geometric_transforms_list = []

    # Cropping: use RandomResizedCrop if enabled, otherwise CenterCrop
    # CenterCrop ensures fixed size (768x768) when images are larger
    # or have varying dimensions. If images are already 768x768, it's a no-op
    if transform_flags.get("random_crop", False):
        # RandomResizedCrop: crop random portion (65-100% area) then resize
        # scale=(0.65, 1.0): crop 65-100% of image area
        # ratio=(1.0, 1.0): keep square aspect ratio for rock images
        geometric_transforms_list.append(
            transforms.RandomResizedCrop(
                size=crop_sz,
                scale=(0.65, 1.0),  # Crop 65-100% of original image area
                ratio=(1.0, 1.0),  # Maintain square aspect ratio
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
        )
    else:
        # CenterCrop: extract center region if images are larger than crop_sz
        # This is NOT redundant if original images are e.g. 800x800 or variable
        geometric_transforms_list.append(transforms.CenterCrop(crop_sz))

    # Add geometric transforms BEFORE ToTensor
    # These must be applied to BOTH image and label
    if transform_flags.get("horizontal_flip", False):
        geometric_transforms_list.append(transforms.RandomHorizontalFlip())

    if transform_flags.get("vertical_flip", False):
        geometric_transforms_list.append(transforms.RandomVerticalFlip())

    if transform_flags.get("rotation", False):
        # Use white fill (255) to match rock color
        # Prevents black boundaries from being learned as joints
        geometric_transforms_list.append(transforms.RandomRotation(15, fill=255))

    # Build image transform: geometric + color (on PIL) + tensor + normalize
    image_transforms_list = geometric_transforms_list.copy()

    # Color augmentations (apply to PIL image BEFORE ToTensor)
    if transform_flags.get("color_jitter", False):
        # ColorJitter works on PIL images (0-255 range)
        # brightness=0.4 allows range [0.6, 1.4] of original brightness
        # This enables darker images to simulate varying lighting conditions
        image_transforms_list.append(
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.3, hue=0.15
            )
        )

    if transform_flags.get("gaussian_blur", False):
        image_transforms_list.append(
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0))
        )

    # Convert to tensor and normalize (after color augmentations)
    image_transforms_list.append(transforms.ToTensor())
    image_transforms_list.append(
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )

    image_transform = transforms.Compose(image_transforms_list)

    # Build label transform: same geometric transforms + ToTensor (NO normalization/color)
    label_transforms_list = geometric_transforms_list.copy()
    label_transforms_list.append(transforms.ToTensor())

    label_transform = transforms.Compose(label_transforms_list)

    return {"image": image_transform, "label": label_transform}


def validate_data_pre_transform(
    images_dir: Path, labels_dir: Path, file_list: list[str]
) -> None:
    """
    Validates dataset files before transformations are applied.
    This function performs basic validation checks on paired image and label
    files to ensure:
    1. Both image and label files exist at the specified paths
    2. Both files can be opened as valid images
    3. Image dimensions match between each image and its corresponding label

    Parameters
    ----------
    images_dir : Path
        Directory path containing the image files
    labels_dir : Path
        Directory path containing the label files
    file_list : list[str]
        List of file names to validate (should be the same name in both directories)

    Returns
    -------
    None
        Function only validates and raises AssertionError if validation fails

    Raises
    ------
    AssertionError
        If an image or label file is missing, cannot be opened,
        or dimensions don't match

    Notes
    -----
    Uses rich.progress.track for progress visualization during validation
    """

    for file_name in track(
        file_list, description="Validating data files pre transform..."
    ):
        image_path = images_dir / file_name
        label_path = labels_dir / file_name

        # Ensure files exist
        if not image_path.exists():
            raise FileNotFoundError(f"Image file missing: {file_name}")
        if not label_path.exists():
            raise FileNotFoundError(f"Label file missing: {file_name}")

        try:
            image = Image.open(image_path)
            label = Image.open(label_path)
        except Exception as e:
            raise AssertionError(
                f"Failed to open image or label file: {file_name}, Error: {e}"
            ) from e

        # Light validation to ensure image and label can be read properly
        if image.size != label.size:
            raise ValueError(
                f"Image and label sizes do not match for file: {file_name}"
            )


def validate_data_post_transform(
    image_tensor: torch.Tensor, label_tensor: torch.Tensor, file_name: str
) -> None:
    # Check if image dimensions are divisible by 32
    _, height, width = image_tensor.shape
    err_msg = (
        f"Transformed image dimensions (HxW): {height}x{width} "
        f"are not divisible by 32 for file: {file_name}"
    )
    if not (height % 32 == 0 and width % 32 == 0):
        raise ValueError(err_msg)

    # Check if the image has correct channels
    if image_tensor.shape[0] != 3:
        raise ValueError(f"Image does not have 3 channels for file: {file_name}")

    # Check if label tensor has a single channel
    if label_tensor.shape[0] != 1:
        raise ValueError(f"Label does not have a single channel for file: {file_name}")

    # Check if mask contains only 0 and 1 values (binary segmentation)
    unique_values = torch.unique(label_tensor)
    if not set(unique_values.tolist()).issubset({0, 1}):
        raise ValueError(
            f"Label contains values other than 0 and 1 for file: {file_name}"
        )

    # Check for black boundaries in image (rotation artifacts)
    # Sample border pixels (top, bottom, left, right edges)
    border_thickness = 5
    top_border = image_tensor[:, :border_thickness, :]
    bottom_border = image_tensor[:, -border_thickness:, :]
    left_border = image_tensor[:, :, :border_thickness]
    right_border = image_tensor[:, :, -border_thickness:]

    # Concatenate all borders and check mean value
    # After normalization, black (0) becomes around -2.1, white becomes around 2.6
    all_borders = torch.cat(
        [
            top_border.flatten(),
            bottom_border.flatten(),
            left_border.flatten(),
            right_border.flatten(),
        ]
    )
    border_mean = all_borders.mean().item()

    # If border mean is very negative (close to normalized black), warn
    if border_mean < -1.5:  # Threshold for detecting black boundaries
        raise ValueError(
            f"Image has black boundaries (mean={border_mean:.2f}) "
            f"for file: {file_name}. Check rotation fill parameter."
        )

    # Check for black boundaries in label mask
    # (should be white=1 after rotation)
    label_top = label_tensor[:, :border_thickness, :]
    label_bottom = label_tensor[:, -border_thickness:, :]
    label_left = label_tensor[:, :, :border_thickness]
    label_right = label_tensor[:, :, -border_thickness:]

    label_borders = torch.cat(
        [
            label_top.flatten(),
            label_bottom.flatten(),
            label_left.flatten(),
            label_right.flatten(),
        ]
    )
    label_border_mean = label_borders.mean().item()

    # Label borders should be mostly 1 (white/rock) not 0 (black/joint)
    if label_border_mean < 0.5:  # More black than white on borders
        raise ValueError(
            f"Label mask has black boundaries (mean={label_border_mean:.2f}) "
            f"for file: {file_name}. Check rotation fill parameter."
        )


def get_data_files(
    images_dir: Path,
    labels_dir: Path,
    train_prefixes: list[str],
    test_prefixes: list[str],
) -> tuple[list[str], list[str]]:
    """
    Get lists of training and testing image file names based on given prefixes.
    This function filters image files in the specified directory by their prefixes
    and ensures that corresponding label files exist in the labels directory.

    Args:
        images_dir (Path): Directory containing the image files.
        labels_dir (Path): Directory containing the label files.
        train_prefixes (list[str]): List of prefixes to filter training image files.
        test_prefixes (list[str]): List of prefixes to filter testing image files.

    Returns:
        tuple[list[str], list[str]]: A tuple containing two lists:
            - train_files: List of training image file names.
            - test_files: List of testing image file names.
    """
    train_files = [
        f.name
        for f in images_dir.iterdir()
        if any(f.name.startswith(prefix) for prefix in train_prefixes)
        and f.suffix == ".png"
        and (labels_dir / f.name).exists()
    ]

    test_files = [
        f.name
        for f in images_dir.iterdir()
        if any(f.name.startswith(prefix) for prefix in test_prefixes)
        and f.suffix == ".png"
        and (labels_dir / f.name).exists()
    ]

    return train_files, test_files


def split_data(
    train_files: list[str],
    test_files: list[str],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> tuple[list[str], list[str], list[str]]:
    """
    Splits file lists into training, validation, and test sets, handling both overlapping and disjoint datasets. This function takes two lists of file paths (train_files and test_files) and splits them into train, validation, and test sets according to the provided fractions. If the train_files and test_files are disjoint (no overlap), the test_files are kept as the test set, and train_files are split into train and validation sets proportionally. If the lists overlap or are from the same dataset, the union of both lists is split into train, validation, and test sets according to the specified fractions.

    The function ensures:
    - The provided fractions sum to 1.0.
    - No duplicate files within each split.
    - No overlap between train, validation, and test sets.
    - The splits are persisted as JSON files for downstream usage.
    Parameters
    ----------
        List of file paths for training data.
        List of file paths for test data.
        Fraction of data to use for training (default: 0.8).
        Fraction of data to use for validation (default: 0.1).
        Fraction of data to use for testing (default: 0.1).
    Returns
        Tuple containing three lists: (train_files, validation_files, test_files).
    Raises
    ------
        If fractions do not sum to 1.0, if there are duplicate files in the input lists,
        or if there is overlap between the resulting splits.
    """
    # Ensure fractions sum up to 1.0
    if not abs(train_frac + val_frac + test_frac - 1.0) < 1e-6:
        raise ValueError("Fractions must sum to 1.0.")

    # Validate inputs contain no duplicates
    if len(train_files) != len(set(train_files)):
        raise ValueError("Train set contains duplicate files")
    if len(test_files) != len(set(test_files)):
        raise ValueError("Test set contains duplicate files")

    # Prepare output directories and write the combined file list (with duplicates)
    raw_dir = Path("data/raw")
    model_ready_dir = Path("data/model_ready")
    raw_dir.mkdir(parents=True, exist_ok=True)
    model_ready_dir.mkdir(parents=True, exist_ok=True)

    all_input_files = train_files + test_files
    with open(raw_dir / "all_files.json", "w") as f:
        json.dump(all_input_files, f)

    # Disjoint vs similar datasets handling
    if set(train_files).isdisjoint(set(test_files)):
        # Different datasets: keep provided test_files unchanged.
        # Split train_files into train/val based on relative proportions of train+val.
        total_tv = train_frac + val_frac
        # Guard against division by zero; though validated earlier to sum to 1.0
        if total_tv <= 0:
            # No train/val requested; all train_files become train, no val
            tv_val_count = 0
        else:
            tv_val_count = int(round(len(train_files) * (val_frac / total_tv)))

        shuffled = train_files[:]
        random.shuffle(shuffled)
        val_list = shuffled[:tv_val_count]
        train_list = shuffled[tv_val_count:]
        test_list = test_files[:]
    else:
        # Similar datasets: split the union according to provided fractions
        unique_files = list(set(train_files) | set(test_files))
        random.shuffle(unique_files)
        n = len(unique_files)
        train_end = int(round(n * train_frac))
        val_end = train_end + int(round(n * val_frac))
        train_list = unique_files[:train_end]
        val_list = unique_files[train_end:val_end]
        test_list = unique_files[val_end:]

    # Validate uniqueness and separation
    def check_no_duplicates(files, label):
        if len(files) != len(set(files)):
            raise ValueError(f"{label} set contains duplicate files.")

    def check_no_overlap(set1, set2, label1, label2):
        if set1 & set2:
            raise ValueError(f"{label1} and {label2} sets have overlapping files.")

    check_no_duplicates(train_list, "Train")
    check_no_duplicates(val_list, "Validation")
    check_no_duplicates(test_list, "Test")

    check_no_overlap(set(train_list), set(val_list), "Train", "Validation")
    check_no_overlap(set(train_list), set(test_list), "Train", "Test")
    check_no_overlap(set(val_list), set(test_list), "Validation", "Test")

    # Persist the splits for downstream usage
    with open(model_ready_dir / "train_files.json", "w") as f:
        json.dump(train_list, f)
    with open(model_ready_dir / "val_files.json", "w") as f:
        json.dump(val_list, f)
    with open(model_ready_dir / "test_files.json", "w") as f:
        json.dump(test_list, f)

    return train_list, val_list, test_list


def get_datasets_prefixes(
    experiment_strategy: str,
    dataset_strategies: dict[str, dict[str, list[str]]],
    dataset_prefixes: dict[str, list[str]],
) -> dict[str, list[str]]:
    """
    Retrieve dataset prefixes for training and testing based on the chosen
    experiment strategy. This is needed to filter the datasets used in the
    experiment. All datasets are stored in the same directory, so we need
    to filter them based on the prefixes.

    Args:
        experiment_strategy (str): The selected experiment strategy.
        dataset_strategies (dict): Dictionary of dataset strategies with train
            and test datasets.
        dataset_prefixes (dict): Dictionary of prefixes for each dataset.

    Returns:
        dict: A dictionary containing lists of prefixes for training and testing
              datasets. Keys are 'train_prefixes' and 'test_prefixes'.

    Example return:
        {
            "train_prefixes": ["prefix1", "prefix2"],
            "test_prefixes": ["prefix3"]
        }

    """
    # Retrieve the datasets for the chosen experiment strategy
    datasets_used = dataset_strategies.get(experiment_strategy)
    if not datasets_used:
        msg = (
            f"Experiment strategy '{experiment_strategy}' "
            "not found in dataset strategies."
        )
        raise ValueError(msg)

    # Initialize lists for prefixes
    train_prefixes = []
    test_prefixes = []

    # Get prefixes for training datasets
    for dataset_name in datasets_used.get("train_datasets", []):
        if dataset_name in dataset_prefixes:
            train_prefixes.extend(dataset_prefixes[dataset_name])
        else:
            raise ValueError(
                f"Dataset name '{dataset_name}' not found in dataset prefixes."
            )

    # Get prefixes for testing datasets
    for dataset_name in datasets_used.get("test_datasets", []):
        if dataset_name in dataset_prefixes:
            test_prefixes.extend(dataset_prefixes[dataset_name])
        else:
            raise ValueError(
                f"Dataset name '{dataset_name}' not found in dataset prefixes."
            )

    return {"train_prefixes": train_prefixes, "test_prefixes": test_prefixes}


# Hydra is a script-level dependency (use in scripts/). Defer/guard import so
# library can be imported in minimal runtime images without hydra installed.
try:
    import hydra  # type: ignore
except Exception:
    hydra = None  # type: ignore


def _require_hydra() -> "hydra":
    """Ensure hydra is available at call-time, raise clear error otherwise."""
    if hydra is None:
        raise RuntimeError(
            "hydra (hydra-core) is not installed. Install it or run this code via"
            " the scripts/ entry points that provide config via Hydra."
        )
    return hydra


def cli_main(cfg):
    cfg_dict: dict[str, Any] = cast(dict[str, Any], OmegaConf.to_object(cfg))
    pcfg = ConfigSchema(**cfg_dict)
    print(pcfg)


if __name__ == "__main__":
    # import hydra only when running the module as a script
    from hydra import main as hydra_main

    hydra_main(
        config_path="../../scripts/config", config_name="main", version_base="1.3"
    )(cli_main)()


# LEGACY CODE
################################################################


# def validate_data(images_dir: Path, labels_dir: Path, file_list: list[str]) -> None:
#     for file_name in file_list:
#         image_path = images_dir / file_name
#         label_path = labels_dir / file_name

#         image = Image.open(image_path).convert("RGB")
#         label = Image.open(label_path).convert("L")
#         label = label.point(lambda p: 255 if p == 255 else 0)

#         # Check if image dimensions are divisible by 32
#         assert (
#             image.height % 32 == 0 and image.width % 32 == 0
#         ), (
#             f"Image dimensions (HxW): {image.height}x{image.width} "
#             "are not divisible by 32"
#         )

#         # Check if all images have the same size
#         assert (
#             image.size == label.size
#         ), f"Image and label sizes do not match for file: {file_name}"

#         # Check if mask has only 0 and 1 values (binary segmentation)
#         label_array = torch.tensor(label, dtype=torch.float)
#         unique_values = torch.unique(label_array)
#         assert set(unique_values.tolist()).issubset(
#             {0, 1}
#         ), f"Mask contains values other than 0 and 1 for file: {file_name}"

#         # Check if the image has correct axes order (convert to CHW)
#         image_tensor = transforms.ToTensor()(image)
#         assert (
#             image_tensor.shape[0] == 3
#         ), f"Image does not have 3 channels for file: {file_name}"

#         # Check if mask is converted to 1HW format
#         label_tensor = torch.tensor(label, dtype=torch.float).unsqueeze(0)
#         assert (
#             label_tensor.shape[0] == 1
#         ), f"Mask does not have a single channel for file: {file_name}"
