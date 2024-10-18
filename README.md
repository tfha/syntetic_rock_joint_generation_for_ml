# Synthetic Rock Joint Generation for ML

This project focuses on generating synthetic rock joints for machine learning applications.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites
- Python 3.11.1+
- [pyenv](https://github.com/pyenv/pyenv)
- [Poetry](https://python-poetry.org/)

### Setting Up the Environment

1. **Install pyenv**:
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

## Usage

To run the project, use:
```sh
python scripts/train.py
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.