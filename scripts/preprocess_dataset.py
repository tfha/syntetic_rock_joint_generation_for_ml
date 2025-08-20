"""
This script preprocesses dataset masks for a machine learning project. It uses Hydra for configuration management
and processes raw mask labels into a specified format.
Functions:
    main(cfg: DictConfig) -> None:
        The main entry point of the script. It reads the configuration, validates it against a schema, and processes
        raw mask labels into processed mask labels.
Usage:
    Run the script with a Hydra configuration file. The configuration file should specify paths for raw and processed
    mask labels, as well as other project-specific settings.
Configuration:
    - The script expects a Hydra configuration file (`main.yaml`) located in the `config` directory.
    - The configuration is validated against the `ConfigSchema` class.
Preprocessing:
    - The `preprocess_masks` function is used to process raw mask labels.
    - Input and output directories for the masks are derived from the configuration.
    - A threshold value of 255 is used for processing the masks.
"""

from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from ml_segmentation.debug_functionality import better_traceback
from ml_segmentation.preprocessing import preprocess_masks
from ml_segmentation.schema_config import ConfigSchema
from ml_segmentation.utility import get_custom_console


@hydra.main(config_path="config", config_name="main.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg_dict: dict[str, Any] = OmegaConf.to_object(cfg)  # to dict
    pcfg = ConfigSchema(**cfg_dict)
    console = get_custom_console()
    console.print(pcfg)

    path_raw_masks = pcfg.path_project / pcfg.dataset.path_raw_mask_labels
    path_processed_masks = pcfg.path_project / pcfg.dataset.path_processed_mask_labels
    preprocess_masks(
        input_dir=path_raw_masks,
        output_dir=path_processed_masks,
        threshold=255,
        limit=None,
        max_workers=6,
    )


if __name__ == "__main__":
    better_traceback()
    main()
