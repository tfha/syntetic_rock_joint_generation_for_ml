# copilot custom instructions

## Code-generation instructions

### Structure and principles in this repo

- The repo is structured with running scripts in the `scripts` directory which use functionality implemented in a python package in `src/ml_segmentation`.
- Config values for use in running scripts are mainly defined using the hydra configuration system, seen by adding a hydra decorator to the main functions in the running scripts
- Config values are given in yaml files in a `config`directory in the `scripts` directory
- Since I have defined the `ml_segmentation` package as include from `src` in the `pyproject.toml` file you can import functionality from the package using only `ml_segmentation`.

### General principles for code formatting and design

- Use the single responsibility principle and less coupled code
- Use the principle of separation of concern
- Use the DRY principle (Don't Repeat Yourself)
- Functions that are only used by other functions should have an underscore as the first character in the name
- Constants names should use capital letters
- Write code with a maximum of 88 characters on each line.

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

## Commit message generation instructions

## Pull request title and description generation instructions
