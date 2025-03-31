# copilot custom instructions

## Code-generation instructions

### Structure and principles in this repo

- The repo is structured with running scripts in the `scripts` directory which use functionality implemented in a python package in `src/ml_segmentation`.
- Config values for use in running scripts are mainly defined using the hydra configuration system, seen by adding a hydra decorator to the main functions in the running scripts.
- Use hydra config functionality for CLI argument parsing and for loading configuration values from YAML files, not argparse or other libraries.
- Config values are given in yaml files in a `config`directory in the `scripts` directory
- Since I have defined the `ml_segmentation` package as include from `src` in the `pyproject.toml` file you can import functionality from the package using only `ml_segmentation`.
- Configuration management follows a strict pattern:
  - Hydra is used ONLY in the running scripts (in `scripts` directory) to load configuration values from YAML files
  - Configuration values are validated using Pydantic models defined in `schema_config.py`
  - The validated Pydantic model (`pcfg`) should be used throughout the code, NOT the raw Hydra config (`cfg`)
  - No Hydra imports or usage should appear in the package functionality (in `ml_segmentation` package)
  - This approach provides: 1) Type safety with IDE completion, 2) Validation of required fields, 3) Centralized configuration management, and 4) Clear separation between configuration and functionality

### General principles for code formatting and design

- Use the single responsibility principle and less coupled code
- Use the principle of separation of concern
- Use the DRY principle (Don't Repeat Yourself)
- Functions that are only used by other functions should have an underscore as the first character in the name
- Constants names should use capital letters
- Write code with a maximum of 88 characters on each line.

### General principles for machine learning development

- Use sound principles for MLOps, such as reproducibility, versioning, and modularity
- The main principle should be to develop code that is easy to protoype and run locally, but is then easy to run in a cloud environment, predominantly using Azure ML

### Python code - special principles

- Use syntax for Python versions higher than 3.10
- Use type annotated code with the newest syntax
- For type annotation, use for example: tuple instead of Tuple, list instead of List, str | int instead of Union[str, int], int | None instead of Optional[int, None]
- For type annotating numpy arrays, use NDarray from numpy.typing
- For Python packages with near similar functionality, choose the Python libraries: pathlib instead of os, opencv instead of PIL, httpx instead of request,  click instead of argparse, pytest instead of unittest
- Use the object-oriented way of coding in matplotlib, ie. ax.plot() instead of plt.plot()
- When inputting a path in a function, use Path from pathlib, not a string
- Use the google docstring format for documenting.
- Use a match-case structure instead of an if-elif structure.
- Use the f-string syntax for string formatting.

## Test-generation instructions

In setting up tests, you should mainly use functionality from the pytest library.

A good test suite should aim to:

- Test the function's behavior for a wide range of possible inputs
- Test edge cases that the author may not have foreseen
- Take advantage of the features of pytest to make the tests easy to write and maintain
- Be easy to read and understand, with clean code and descriptive names
- Be deterministic so that the tests always pass or fail in the same way
- Follow the arrange-act-assert pattern

Other important aspects of a good test suite include:

- Make small reusable test datasets to test edge cases and typical cases
- Utilize pytest.parametrize or pytest.fixture if that is useful, also in test classes."
- Increase and explain the functionality of the Python code by providing good tests for showcasing functionality
- Show error messages after an assertion fails, predominately showing the difference between the expected and actual output.
- Bundle several test functions for a function or class in a test class.
- Do not test functionality in the Python standard library or third party libraries, unless it is necessary for the function or class to work.

Organising and running tests:

- Organise the test in separate test files for each python file and place them in the tests directory
- After you have made the test you should run them, check the response and fix the errors in the code.

## Code review instructions

Use the principles defined in the code-generation instructions to review the code.

## Commit message generation instructions

Include a main message at the top, and bullet list of changes made in the commit.

## Pull request title and description generation instructions

Use informative titles and descriptions for the pull request. The title should be a short summary of the changes made, and the description should provide more detail about the changes, including any relevant context or background information. The description should also include any relevant links to issues or discussions related to the changes made in the pull request.
