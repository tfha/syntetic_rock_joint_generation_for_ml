# include neccessary imports
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from ml_segmentation.debug_functionality import better_traceback
from ml_segmentation.preprocessing import preprocess_masks
from ml_segmentation.schema_config import ConfigSchema


# from ml_segmentation.utility import get_custom_console


@hydra.main(config_path="config", config_name="main.yaml", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg_dict: dict[str, Any] = OmegaConf.to_object(cfg)  # to dict
    pcfg = ConfigSchema(**cfg_dict)
    # console = get_custom_console()
    # console.print(pcfg)

    path_raw_masks = pcfg.path_project / pcfg.dataset.path_raw_mask_labels
    path_processed_masks = pcfg.path_project / pcfg.dataset.path_processed_mask_labels
    preprocess_masks(
        input_dir=path_raw_masks,
        output_dir=path_processed_masks,
        threshold=255,
        limit=None,
    )


if __name__ == "__main__":
    better_traceback()
    main()
