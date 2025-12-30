import os
import time
from collections.abc import Callable

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from rich.traceback import install


def visualize_sample(image: torch.Tensor, label: torch.Tensor) -> None:
    print(label.min(), label.max())  # Should output: 0.0 1.0

    plt.subplot(1, 2, 1)
    plt.imshow(image.permute(1, 2, 0).numpy())  # Convert to H x W x C for display
    plt.title("Input Image")

    plt.subplot(1, 2, 2)
    plt.imshow(label[0].numpy(), cmap="gray")
    plt.title("Label Mask")
    plt.show()


def better_traceback() -> None:
    """
    run inspect on objects when debugging with ipdb
    e.g inspect(df, metods=True)
    """
    os.environ["HYDRA_FULL_ERROR"] = "1"
    install(show_locals=False)
    from rich import inspect  # noqa


def timeit(func: Callable) -> Callable:
    "Decorator for computing time taken to execute function."

    def timed(*args):
        t_start = time.time()
        res = func(*args)
        t_end = time.time()
        print(
            f"Function time for {func.__name__}() is: {(t_end - t_start) * 1000: 2.2f} ms"
        )
        return res

    return timed


def nn_shape(func):
    """Decorator to output debug info from a neural network.
    - input and output shape of forward method

    Note: For detailed layer-by-layer output shapes and feature maps, use torchinfo.summary()
    or PyTorch hooks (see PyTorch documentation).

    Formula for computing output shape (height*width): (w-f + 2p)/s + 1
    - w: width and height of input feature map
    - f: kernel size
    - p: padding
    - s: stride
    """

    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        print(
            f"Info for nn: {args[0].__class__.__name__} of parent: \
                {args[0].__class__.__bases__[0].__name__}"
        )
        print(f"Input shape of forward: {args[1].shape}")
        print(f"Output shape of forward: {res.shape}")
        print()
        return res

    return wrapper


def nn_info(nn: nn.Module, print_structure=False):
    """Print information about a neural network.

    Args:
        nn: The neural network module
        print_structure: Whether to print the full structure (unused, kept for compatibility)

    Prints:
        - Number of trainable parameters
        - Number of total parameters

    Note: For detailed receptive field calculations and layer-wise analysis,
    use torchinfo.summary() or specialized tools like torch-receptive-field.
    """
    num_trainable_params = sum(p.numel() for p in nn.parameters() if p.requires_grad)
    num_params = sum(p.numel() for p in nn.parameters())
    print(f"Information about the neural network: {nn.__class__.__name__}")
    print("------------------------------------")
    print(f"Number of trainable parameters: {num_trainable_params}")
    print(f"Number of total parameters: {num_params}")
    print()
    if print_structure:
        print("Network architecture: ")
        print(nn)
