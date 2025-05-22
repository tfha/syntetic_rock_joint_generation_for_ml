import random
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from rich.progress import track
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ml_segmentation.schema_config import ConfigSchema


class SegmentationDataset(Dataset):
    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        file_list: list[str],
        transform: transforms.Compose | None = None,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.file_list = file_list
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.images_dir / self.file_list[idx]
        label_path = self.labels_dir / self.file_list[idx]

        # Load image and label
        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path)

        # Apply transformations if any
        if self.transform:
            image = self.transform["image"](image)
            label = self.transform["label"](label)
            # Invert label to match the segmentation model's requirements
            label = 1 - label

        return image, label


def get_datasets(
    images_dir: str | Path,
    labels_dir: str | Path,
    train_files: list[str],
    val_files: list[str],
    test_files: list[str],
    transform: dict[str, transforms.Compose] | None = None,
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
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def get_transforms(optional_transforms: bool = False) -> dict[str, transforms.Compose]:
    """
    Using all the transforms, the effective virtual dataset size will be
    ~14.4x larger during training compared to the original 1000 images.
    While you still only have 1000 original images saved, the model will
    effectively see about 14,400 variations of your images during training,
    which significantly improves generalisation without increasing stored
    images.
    """  # Crop size - must be divisible by 32
    crop_size = 768
    # resize_size = 384  # Example size that is divisible by 32

    train_transforms_list = [
        transforms.CenterCrop(crop_size),
        # transforms.Resize(
        #     (resize_size, resize_size), interpolation=Image.BILINEAR
        # ),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),  # transforms the image to a tensor in the range [0, 1]
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # normalizes the image to have a mean and standard deviation of 0.5
    ]

    if optional_transforms:
        train_transforms_list.extend(
            [
                # transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.3, saturation=0.2, hue=0.1
                ),
                # transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),
            ]
        )

    image_transform = transforms.Compose(train_transforms_list)

    # Apply the same center crop to the labels
    label_transform = transforms.Compose(
        [
            transforms.CenterCrop(crop_size),  # Centre crop to match images
            # transforms.Resize(
            #     (resize_size, resize_size), interpolation=Image.NEAREST
            # ),
            transforms.ToTensor(),
        ]
    )

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
        assert image_path.exists(), f"Image file missing: {file_name}"
        assert label_path.exists(), f"Label file missing: {file_name}"

        try:
            image = Image.open(image_path)
            label = Image.open(label_path)
        except Exception as e:
            raise AssertionError(
                f"Failed to open image or label file: {file_name}, Error: {e}"
            )

        # Light validation to ensure image and label can be read properly
        assert image.size == label.size, (
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
    assert height % 32 == 0 and width % 32 == 0, err_msg

    # Check if the image has correct channels
    assert image_tensor.shape[0] == 3, (
        f"Image does not have 3 channels for file: {file_name}"
    )

    # Check if label tensor has a single channel
    assert label_tensor.shape[0] == 1, (
        f"Label does not have a single channel for file: {file_name}"
    )

    # Check if mask contains only 0 and 1 values (binary segmentation)
    unique_values = torch.unique(label_tensor)
    assert set(unique_values.tolist()).issubset({0, 1}), (
        f"Label contains values other than 0 and 1 for file: {file_name}"
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
    Split data into training, validation, and test sets based on provided file
    lists. Splitting is only performed if train_files and test_files are similar,
    i.e the same dataset (not disjoint). If they are disjoint, an error is raised.

    Parameters:
    -----------
    train_files : list[str]
        List of file paths for training data
    test_files : list[str]
        List of file paths for test data
    train_frac : float, optional
        Fraction of data to use for training (default: 0.8)
    val_frac : float, optional
        Fraction of data to use for validation (default: 0.1)
    test_frac : float, optional
        Fraction of data to use for testing (default: 0.1)

    Returns:
    --------
    tuple[list[str], list[str], list[str]]
        Tuple containing three lists: (train_files, validation_files, test_files)

    Raises:
    -------
    ValueError
        If fractions don't add up to 1.0, or if train_files and test_files are disjoint
    """
    # Ensure fractions sum up to 1.0
    if not abs(train_frac + val_frac + test_frac - 1.0) < 1e-6:
        raise ValueError("Fractions must sum to 1.0.")

    # Check if train_files and test_files are disjoint
    if set(train_files).isdisjoint(set(test_files)):
        msg = (
            "train_files and test_files are disjoint. "
            "Splitting is only allowed when they are similar."
        )
        raise ValueError(msg)

    # Split all files into train, validation, and test sets
    all_files = list(set(train_files) | set(test_files))
    random.shuffle(all_files)
    train_end = int(len(all_files) * train_frac)
    val_end = train_end + int(len(all_files) * val_frac)

    train_list = all_files[:train_end]
    val_list = all_files[train_end:val_end]
    test_list = all_files[val_end:]

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


@hydra.main(config_path="../../scripts/config", config_name="main", version_base="1.3")
def testing_functionality(cfg: DictConfig) -> None:
    cfg_dict: dict[str, Any] = OmegaConf.to_object(cfg)
    pcfg = ConfigSchema(**cfg_dict)
    print(pcfg)


if __name__ == "__main__":
    testing_functionality()


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
