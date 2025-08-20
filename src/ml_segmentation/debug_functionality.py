import os
import time
from typing import Callable

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
    """Decorator to output debug info from a nn.
    - input and output shape of forward method
    TODO: Include the output shape of feature map for each convolution and pooling step
    (cannot be seen from the input and outchannels in the initialization of a
    conv-layer). This list of output-shapes should be printed for a certain net.

    Formula for computing output shape (height*width) (w-f + 2p)/s + 1
    - w: width and height of input feature map
    - f: kernel size
    - p: padding
    - s: stride

    TODO: Set up a summary of the network as done in keras like this:
    https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/

    https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6IjE3MTllYjk1N2Y2OTU2YjU4MThjMTk2OGZmMTZkZmY3NzRlNzA4ZGUiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJuYmYiOjE2MjI2NjQyMjEsImF1ZCI6IjIxNjI5NjAzNTgzNC1rMWs2cWUwNjBzMnRwMmEyamFtNGxqZGNtczAwc3R0Zy5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsInN1YiI6IjExODM1OTEyNDY0MDI2OTk0MTUyNyIsImVtYWlsIjoidG9tLmZyb2RlLmhhbnNlbkBnbWFpbC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiYXpwIjoiMjE2Mjk2MDM1ODM0LWsxazZxZTA2MHMydHAyYTJqYW00bGpkY21zMDBzdHRnLmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29tIiwibmFtZSI6IlRGIEgiLCJwaWN0dXJlIjoiaHR0cHM6Ly9saDMuZ29vZ2xldXNlcmNvbnRlbnQuY29tL2EtL0FPaDE0R2dLRndIMnNaWWdYSllmR1NnSUFGV3luMHZ5b0pTR0dHbDNqQjk3WGc9czk2LWMiLCJnaXZlbl9uYW1lIjoiVEYiLCJmYW1pbHlfbmFtZSI6IkgiLCJpYXQiOjE2MjI2NjQ1MjEsImV4cCI6MTYyMjY2ODEyMSwianRpIjoiMTdiNDU0MWM0MmE2ZGYzODhkZWNiODM5M2U3MGM2ZTI2MGUwZWYzMyJ9.d3yg4Elw-L9JI83_dy6KSYLtkc9e06BS2q22aJBddP520W-CcN6rwwwcSCPDQnNM8lEakpYS1JveVnEIoeo8BVtOOM-GcuImkq0d1o_5Ts8H6m3TvxbP-7VU7vocTThZZkGmWZrrWrhdWjE-LpJ6AS26S1omclyfDt-As3IXOhXozU59Z9EOI_Ap3xarsBO9MKj0Y_LFF_XreTk3-WJ5UfInuK2aZKPRVI3j3YDMmwVj6q4vlky0Coo7MpBbIIRCc5nZic3v-d7g-iq7nfsJzR3t1fR_-9F9JqyC-8RccGIbWYU3fh144eDvhHLP5sx-dUgIRnZTgQwYJNFMOde_4Q

    Can probably use a hook for this:
    https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html

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
    """
    Print information about a NN.
    Use this in combination with the nn_shape decorator on the forward method

    - Number of trainable parameters
    TODO: calculate the receptive field in each layer. Se lecture 5.
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
