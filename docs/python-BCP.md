# Python Development Best Current Practices (BCP)

This document outlines the Best Current Practices (BCP) for modern Python development. It serves as a guideline for writing clean, readable, maintainable, and efficient code.

## 1. Code Style and Formatting (Readability)

The foundation of Python readability is consistency.

### Foundation: PEP 8

PEP 8 is the official style guide for Python code. Adherence is strongly recommended.

- **Indentation:** Use 4 spaces per indentation level.
- **Line Length:** Limit lines to a maximum of 79-99 characters. Tools often default to 88 (`black`) or 99.
- **Naming Conventions:**
  - `snake_case` for functions and variables.
  - `PascalCase` for classes.
  - `UPPERCASE_SNAKE_CASE` for constants.
- **Whitespace:** Use it thoughtfully around operators and after commas to improve clarity.

### Automation: Linters and Formatters

Manual formatting is tedious and error-prone. Automation is the BCP.

- **Linters (`flake8`, `pylint`, `ruff`):** Check code for style violations and potential logical errors.
- **Formatters (`black`, `ruff format`):** Automatically reformat code to a consistent style, eliminating style debates.

## 2. Project Structure and Environment Management

A well-organized project is easier to navigate, test, and deploy.

### Virtual Environments

Isolate project dependencies to prevent conflicts. *Always* use a virtual environment.

- **Standard Tool:** `venv` (built into Python).
- **Integrated Tools:** `poetry` and `pdm` manage environments automatically.

### Dependency Management

- **`pyproject.toml` (PEP 518/621):** The modern, standardized file for defining project metadata, build requirements, and dependencies.
- **Tools:**
  - **`pip` with `requirements.txt`:** The traditional method.
  - **`poetry` / `pdm`:** Modern tools that manage dependencies, virtual environments, and packaging from `pyproject.toml`, providing lock files for reproducible builds.

### Directory Structure

The "src layout" is highly recommended to prevent common import problems.

```text
project_root/
├── pyproject.toml
├── README.md
├── src/
│   └── my_package/
│       ├── __init__.py
│       └── module.py
└── tests/
    └── test_module.py
```

## 3. Idiomatic Python ("Pythonic" Code)

Writing "Pythonic" code means using the language's features as they are intended to be used. It follows the principles in the "Zen of Python" (`import this`).

### Key Idioms

- **Comprehensions:** Use list, dict, or set comprehensions for creating collections.
- **`with` Statement:** Manage resources like files or network connections to ensure proper setup and teardown.
- **Generators (`yield`):** Use for lazy evaluation over large datasets to conserve memory.
- **Truthiness:** Check for empty sequences directly (e.g., `if my_list:`) instead of checking the length.
- **Mutable Default Arguments:** Avoid them. Use `None` as a default and initialize the mutable type inside the function body.

## 4. Documentation and Type Hinting

Code is read more often than it is written. Clear documentation is critical.

### Docstrings (PEP 257)

Every public module, function, class, and method should have a docstring.

- **Content:** Explain the purpose, arguments, return values, and any exceptions raised.
- **Formats:** Common styles include Google, reStructuredText, and NumPy.

### Type Hinting (PEP 484)

Type hints are now standard practice for new Python code.

- **What it is:** Annotating variables and function signatures with their expected types (e.g., `def greet(name: str) -> str:`).
- **Benefit:** Enables static analysis tools (`mypy`, `pyright`) to catch bugs before runtime and serves as excellent documentation.

## 5. Testing

Code without automated tests is considered legacy code.

### Frameworks

- **`pytest`:** The de facto standard for testing in Python due to its simplicity and powerful features (fixtures, plugins).
- **`unittest`:** The built-in standard library module.

### Coverage

- **`coverage.py`:** Use this tool to measure the percentage of your code that is executed by your tests. Aim for high, but meaningful, coverage.
