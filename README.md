# Synthetic Rock Joint Generation for ML

## Table of Contents

- [Introduction](#introduction)
- [Key tools and technologies](#key-tools-and-technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Contact](#contact)

## Introduction

This project use machine learning to segment and detect rock joint traces in images. Rock joints are fractures in the rock mass that can have a significant impact on the stability of the rock mass. The detection of rock joints is important for geotechnical analysis and rock mass quality evaluation.

Specifically we evaluate the performance of models trained on synthetic data and test them on real data. The syntetic dataset is generated in a process described in the following paper:

Chiu, J. K. Y., Hansen, T. F., (2025). Synthetic Rock Joint Generation for Machine Learning Applications. Preprint on arXiv: [https://arxiv.org/abs/2501.01234](https://arxiv.org/abs/2501.01234)

Semantic segmentation is a computer vision task where the goal is to classify each pixel in an image into a category. In this project, we use semantic segmentation to classify each pixel in an image as either a rock joint or not a rock joint. We have implementd training for the following semantic segmentation networks:

- DeepLabV3Plus
- UNet++
- FracSegNet?
- CrackSegDiff?

## Datasets and training strategies

**Real-world dataset**: A dataset of real-world images of rock joints from rock cuttings in Larvik and RV-4 project.

**Synthetic dataset**: A dataset of synthetic images generated using the method described in the paper.

Variations for training and evaluation on the synthetic dataset:

- Train and evaluate only on the synthetic dataset. The goal is to evaluate the performance of the model on synthetic data and compare the performance of different models on synthetic data. Choose the best model based on the performance on the synthetic dataset.
- Use a transfer learning approach where the syntetic dataset is used to finetune a model trained on the following datasets: A brick dataset, ... (more datasets to be added). Evaluate on the synthetic dataset.
- Transfer learning finetuned with syntetic data and evaluated on real-world data. *The main goal of the study is to demonstrate that this model can be used to detect rock joints in real-world data.*
- Semi-supervised or self-supervised learning on the synthetic dataset. Use a few labelled samples to finetune the model. Evaluate on the synthetic dataset.
- Semi-supervised or self-supervised learning on the synthetic dataset. Use a few labelled samples to finetune the model. Evaluate on the real-world dataset.
- Semi-supervised or self-supervised learning on the syntehtic dataset. Use a few labelled samples from the real-world dataset to finetune the model. Evaluate on the real-world dataset.
- One shot segmentation on syntetic and real-world data samples using a foundation model (SAM2)


## Key tools and technologies

- The projects is developed as a classic software project with functionality structured as a **python package** in the `src` directory and entry points in the `scripts` directory.
- Use of the **Poetry** package manager for dependency management.
- Use of **pyenv** for managing Python versions.
- Use of the **Hydra** configuration framework for easy configuration of the model and training parameters.
- Use of **mlflow** for tracking experiments and model parameters. The path to each experiment config in hydra is stored in mlflow, so you can easily reproduce the results of each experiment.
- Use of **tensorboard** for tracking training metrics, loss-development, computational profiling and model performance while the model is training. The distinction between the two is that tensorboard is used for tracking the training process, while mlflow is used for tracking the results of the training process.
- Use **pydantic** schemes to validate the configuration parameters.
- Use of **optuna** for hyperparameter optimization.
- Use of the **black**, **isort** and **ruff** code formatters for code formatting and linting.
- Use of **segmentation-models-pytorch** for loading well implemented solutions for networks such as UNet, DeepLabV3, etc.
- Use of **torchmetrics** for metrics calculation.
- Use of **torchsummary** for model summary.
- Use of **fiftyone** for data visualization and data management.
- Use of **pytorch-lightning** for training loop.
- Use of **pytest** for testing.
- Use of **docker** for containerization, reproducability and running training in the cloud.
- Use of **dvc** for data versioning.
- Use of **captum** for model interpretability, highlighting the importance of each pixel in the image for the model's prediction.

## Metrics for evaluation

- **IoU (Intersection over Union)**: IoU is calculated as the ratio of the intersection of the predicted and true positive pixels to the union of the predicted and true positive pixels. The IoU metric is a measure of the overlap between the predicted and true positive pixels and is used to evaluate the accuracy of the model in detecting rock joints in the image.

- **Dice Coefficient**: The Dice Coefficient is calculated as the ratio of twice the intersection of the predicted and true positive pixels to the sum of the predicted and true positive pixels. The Dice Coefficient is a measure of the similarity between the predicted and true positive pixels and is used to evaluate the accuracy of the model in detecting rock joints in the image. It’s sensitive to how well the model captures fine details

- **Precision and Recall**: Precision indicates how many of the predicted joint traces are accurate, while recall shows how many of the true joint traces were detected. Balancing these metrics is key for generalising to different rock masses, as some joints may be subtle and easily missed (low recall) or overestimated (low precision)

We illustrate the prediction mask images in fiftyone, where the predicted mask is overlaid on the original image. This allows us to visually inspect the model's performance and identify any potential issues with the model's predictions.


## Installation

### Prerequisites

- Python 3.11.1+
- [pyenv](https://github.com/pyenv/pyenv)
- [Poetry](https://python-poetry.org/)

### Setting Up the Environment

1. **Install pyenv**:

    - **Install pyenv in Linux**:

        ```sh
        curl https://pyenv.run | bash
        ```

        Add the following to your `~/.bashrc` or `~/.zshrc`:

        ```sh
        export PATH="$HOME/.pyenv/bin:$PATH"
        eval "$(pyenv init --path)"
        eval "$(pyenv init -)"
        eval "$(pyenv virtualenv-init -)"
        ```

        Restart your shell:

        ```sh
        exec "$SHELL"
        ```

    - **Install pyenv in Windows**:

        ```sh
        Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"
        ```

2. **Install Python**:

    ```sh
    pyenv install 3.11.1
    pyenv local 3.11.1
    ```

3. **Install Poetry**:

    ```sh
    curl -sSL https://install.python-poetry.org | python3 -
    ```

    Add Poetry to your PATH:

    ```sh
    export PATH="$HOME/.local/bin:$PATH"
    ```

4. **Install Project Dependencies**:

    ```sh
    poetry install
    ```

5. **Activate the Virtual Environment**:

    ```sh
    poetry shell
    ```

6. **Install Pre-Commit Hooks**:

    ```sh
    poetry run pre-commit install
    ```

## Usage

### Train and Evaluate the Model

```sh
python scripts/train.py
```

Use hydra configuration options with train.py to specify different models and training parameters. For example, to train a model with DeepLabV3 architecture, use the following command:

```sh
python scripts/train.py model=deeplabv3
```

See all options with:

```sh
python scripts/train.py --help
```

### Hyperparameter Optimization

```sh
python scripts/optimize.py model=deeplabv3
```

### Inspect the experiment results in MLflow

The experiments directory is stored in a shared location, so you can inspect the results of the experiments by running the following command:

```sh
cd experiments
mlflow ui
```

Open the web interface at `http://localhost:5000` to view and inspect the results.

### Special functionality

Check if we have enough samples to get the full potential of the model based on the syntetic dataset. The script will train several machine learning models using different amounts of samples and plot the results. If the model performance converge to a certain value, it indicates that we have enough samples to get the full potential of the model.

```sh
python scripts/check_samples.py
```


## Contact

The project is developed by:

- Tom F. Hansen: [tom.frode.hansen@ngi.no](mailto:tom.frode.hansen@ngi.no)
- Jessica Ka Yi Chiu: [jessica.ka.yi.chiu@ngi.no](mailto:jessica.ka.yi.chiu@ngi.no)

For any questions or feedback, please feel free to reach out to us.