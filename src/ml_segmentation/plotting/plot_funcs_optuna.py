from pathlib import Path

import matplotlib.pyplot as plt
import optuna
import pandas as pd
from optuna.visualization.matplotlib import plot_parallel_coordinate

from ml_segmentation.plotting.plot_utility import plot_info, save_plot


@plot_info
def plot_optuna_pareto_3D(
    study: optuna.Study,
    savepath: Path,
) -> None:
    """
    Plots the optimization history of the given Optuna study.

    Parameters:
        study (optuna.Study): The Optuna study object.
        savepath (Path): The path to save the plot.
        figure_width (float, optional): The width of the figure. Defaults to 3.15.
        save_extra_formats (bool, optional): Whether to save the plot in additional formats. Defaults to False.
    """

    fig = optuna.visualization.plot_pareto_front(
        study,
        target_names=["Silhouette score", "Davies-Bouldain", "Calinski-Harabasz"],
    )

    # Update layout for axes labels and tick labels
    fig.update_layout(
        xaxis={
            "title": {"font": {"size": 30}},  # Adjust font size for x-axis label
            "tickfont": {"size": 20},  # Adjust font size for x-axis tick labels
        },
        yaxis={
            "title": {"font": {"size": 30}},  # Adjust font size for y-axis label
            "tickfont": {"size": 20},  # Adjust font size for y-axis tick labels
        },
    )

    # If the plot includes bar labels (annotation on bars), you can adjust them as well:
    fig.update_traces(
        textfont={"size": 20}  # Adjust font size for bar labels
    )

    fig.write_html(savepath)


@plot_info
def custom_optimization_history_plot(
    df_study: pd.DataFrame,
    savepath: Path,
    figure_width: float,
    save_extra_formats: bool = False,
) -> None:
    """Custom implementation of the plot_optimization_history() function of optuna.

    This function generates a customized plot to visualize the optimization history of an Optuna study.
    It takes a dataframe `df_study` containing the study results, a `savepath` to save the plot,
    `figure_width` to specify the width of the plot, and an optional `save_extra_formats` flag to save the plot in additional formats.

    Args:
        df_study (pd.DataFrame): A dataframe containing the study results.
        savepath (Path): The path to save the plot.
        figure_width (float): The width of the plot.
        save_extra_formats (bool, optional): Flag to save the plot in additional formats. Defaults to False.

    Returns:
        None: This function does not return anything.

    References:
        - Documentation for plot_optimization_history() function of optuna:
          https://optuna.readthedocs.io/en/stable/reference/visualization/generated/optuna.visualization.plot_optimization_history.html#optuna.visualization.plot_optimization_history
    """
    fig, ax = plt.subplots(figsize=(figure_width, figure_width))

    ax.scatter(
        df_study["number"],
        df_study["value"],
        s=30,
        alpha=0.7,
        color="grey",
        edgecolor="black",
        zorder=10,
    )
    # plot the first point in the dataframe in red color - to highlight the default parameters. The first trial is run with default parameters.
    ax.scatter(
        df_study["number"].iloc[-1],
        df_study["value"].iloc[-1],
        s=30,
        alpha=0.7,
        color="red",
        edgecolor="black",
        zorder=10,
        label="Trial with default parameters",
    )
    ax.set_ylim(
        (float(df_study["value"].min() - 0.05), float(df_study["value"].max() + 0.05))
    )
    # ax.set_ylim([0.7, 1.0])
    ax.plot(df_study["number"], df_study["value"].ffill().cummax(), color="black")
    ax.grid(alpha=0.5)
    ax.set_xlabel("trial number")
    ax.set_ylabel("Balanced accuracy")
    ax.legend()
    save_plot(savepath, save_extra_formats=save_extra_formats)


@plot_info
def plot_hyperparameter_parallel_coordinates(
    study: optuna.Study,
    savepath: Path,
    figure_width: float = 3.15,
    save_extra_formats: bool = False,
) -> None:
    """
    Plots the hyperparameter parallel coordinates for the given Optuna study.

    Parameters:
        study (optuna.Study): The Optuna study object.
        savepath (Path): The path to save the plot.
        figure_width (float, optional): The width of the figure. Defaults to 3.15.
        save_extra_formats (bool, optional): Whether to save the plot in additional formats. Defaults to False.
    """
    plot_parallel_coordinate(study)
    fig = plt.gcf()
    fig.set_size_inches(3 * figure_width, 2 * figure_width)
    fig.tight_layout()
    # fig.savefig(savepath)
    save_plot(savepath, save_extra_formats=save_extra_formats)
