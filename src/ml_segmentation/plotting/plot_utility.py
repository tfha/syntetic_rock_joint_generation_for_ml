from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
from rich import print as pprint

from ml_segmentation.utility import modify_filepath


def plot_info(func: Callable):
    """Decorator for plotting functions"""

    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
        pprint(f"\n[green]Finished plotting {func.__name__}")

    return wrapper


def save_plot(
    save_path: Path,
    save_extra_formats: bool = False,
) -> None:
    """
    Save the current plot to the specified file path.

    Parameters:
        save_path (Path): The file path where the plot will be saved.
        save_extra_formats (bool, optional): Flag indicating whether to save the plot in additional formats.
            Defaults to False.

    Returns:
        None
    """
    if save_extra_formats:
        plt.savefig(save_path.with_suffix(".svg"))
        pprint("Saved svg")
        plt.savefig(save_path)
        pprint("Saved png 100 dpi")
        new_path = modify_filepath(save_path, "_600")
        plt.savefig(new_path, dpi=600)
        pprint("Saved png 600 dpi")
        # plt.savefig(save_path.with_suffix(".eps"))
        plt.savefig(save_path.with_suffix(".pdf"))
        pprint("Saved pdf")
    else:
        plt.savefig(save_path)
        pprint("Saved png 100 dpi")
    plt.clf()
