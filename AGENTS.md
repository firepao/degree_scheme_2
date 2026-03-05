# AGENTS.md - Development Guidelines for fertopt

This document provides guidelines for AI agents working on this fertilization optimization project.

## Project Overview

- **Project Name**: fertopt (Fertilization Optimization)
- **Type**: Python scientific computing / evolutionary algorithm project
- **Core Functionality**: Multi-objective optimization (NSGA-II) for fertilizer recommendation using surrogate models
- **Location**: `/mnt/c/Windows/system32/opencode/degree_scheme_2/scheml_2`

## 1. Build, Lint, and Test Commands

### Installation

```bash
# Install project in development mode
cd /mnt/c/Windows/system32/opencode/degree_scheme_2/scheml_2
pip install -e .

# Install all dependencies
pip install -r requirements.txt

# Optional: Install additional dependencies for full functionality
pip install torch pandas scikit-learn joblib tqdm
```

### Running Tests

```bash
# Run all tests
pytest

# Run all tests with verbose output
pytest -v

# Run a specific test file
pytest tests/test_initialization.py

# Run a specific test function
pytest tests/test_initialization.py::test_beta_initialization_respects_bounds

# Run tests matching a pattern
pytest -k "test_beta"

# Run with coverage (if coverage is installed)
pytest --cov=src/fertopt --cov-report=term-missing
```

### Running the Application

```bash
# Run baseline experiment (uses config in configs/default.yaml)
python experiments/run_baseline.py

# Run with custom config
python experiments/run_baseline.py --config configs/default.yaml --out artifacts/my_run

# Run with specific engine (random_search or deap_nsga2)
python experiments/run_baseline.py --engine deap_nsga2
```

### Linting and Formatting

```bash
# Check formatting with black (if installed)
black --check src/fertopt/

# Format code with black
black src/fertopt/

# Check linting with ruff (if installed)
ruff check src/fertopt/

# Fix linting issues
ruff check --fix src/fertopt/
```

## 2. Code Style Guidelines

### General Principles

- **Follow existing patterns**: Always match the coding style of existing files in the same module.
- **No hardcoded paths**: Use configuration files or relative paths. Never hardcode absolute paths like `D:\...` or `/home/...`.
- **Small, reversible changes**: Prefer small commits over large refactorings.
- **Test coverage**: Any new functionality should have corresponding tests.

### Imports

- Use `from __future__ import annotations` for all Python files.
- Group imports in the following order (separated by blank lines):
  1. Standard library
  2. Third-party packages
  3. Local application imports

```python
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .config import AppConfig
from .problem import FertilizationProblem
from ..models.surrogate import SurrogateManager
```

### Formatting

- Use **Black** for code formatting (line length: 88 characters default).
- Use 4 spaces for indentation (no tabs).
- Maximum line length: 120 characters (or follow Black's default).
- Use blank lines sparingly to separate logical sections within functions.

### Type Hints

- Use Python type hints for all function signatures.
- Use `np.ndarray` for numpy arrays.
- Use `list[Type]`, `dict[KeyType, ValueType]` for generic types (Python 3.9+).
- Use `| None` instead of `Optional[Type]` for simple cases.

```python
def beta_biased_initialize(
    pop_size: int,
    dimension: int,
    lower_bound: float,
    upper_bound: float,
    strength_k: float,
    rng: np.random.Generator,
) -> np.ndarray:
    ...
```

### Naming Conventions

- **Variables**: `snake_case` (e.g., `population_size`, `mutation_prob`)
- **Functions**: `snake_case` (e.g., `beta_biased_initialize`)
- **Classes**: `PascalCase` (e.g., `FertilizationProblem`, `SurrogateManager`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_POPULATION_SIZE`)
- **Modules**: `snake_case` (e.g., `initialization.py`, `surrogate_predictor.py`)
- **Private methods**: prefix with underscore (e.g., `_evolve_generation`)

### Dataclasses

- Use `@dataclass(slots=True)` for all dataclasses for memory efficiency.
- Define field types explicitly.

```python
@dataclass(slots=True)
class SurrogateParams:
    enabled: bool
    update_interval_g: int
    query_batch_size: int
    target_objectives: list[str]
    model_type: str = "lightgbm"
```

### Error Handling

- Use specific exceptions (e.g., `ValueError`, `FileNotFoundError`).
- Provide informative error messages.
- Fail fast with clear messages for invalid inputs.

```python
if pop_size <= 0:
    raise ValueError("pop_size 必须大于 0")
if upper_bound <= lower_bound:
    raise ValueError("upper_bound 必须大于 lower_bound")
```

### Logging

- Use the `logging` module for important operations.
- Use `logger = logging.getLogger(__name__)` at module level.

```python
import logging

logger = logging.getLogger(__name__)

logger.info("Starting evolution loop")
logger.warning(f"Failed to load model from {path}, using fallback")
```

### Matplotlib

- Always use `matplotlib.use("Agg")` for non-interactive backends in scripts.
- Always call `plt.close()` after saving figures to free memory.

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ... plotting code ...
plt.savefig(out_path, dpi=150)
plt.close(fig)  # Important!
```

### Testing Guidelines

- Test file naming: `test_<module_name>.py`
- Test function naming: `test_<description>() -> None:`
- Use `pytest` assertions.
- Provide meaningful test names that describe what is being tested.
- Test edge cases and boundary conditions.

```python
def test_beta_initialization_respects_bounds() -> None:
    rng = np.random.default_rng(123)
    pop = beta_biased_initialize(
        pop_size=40,
        dimension=12,
        lower_bound=0.0,
        upper_bound=300.0,
        strength_k=8.0,
        rng=rng,
    )

    assert pop.shape == (40, 12)
    assert np.min(pop) >= 0.0
    assert np.max(pop) <= 300.0
```

### Configuration

- Store configuration in YAML files under `configs/`.
- Use `AppConfig` dataclass with `load_config()` function to parse configs.
- Never hardcode configuration values in code.

### Documentation

- Write docstrings for all public functions and classes.
- Use Google-style or NumPy-style docstrings.
- Keep docstrings concise but informative.

```python
def beta_biased_initialize(...) -> np.ndarray:
    """连续型偏向随机初始化。

    对第 i 个个体采用:
      s_i = (i - 0.5) / N
      alpha = 1 + K * s_i
      beta  = 1 + K * (1 - s_i)

    Args:
        pop_size: Population size.
        dimension: Problem dimension.
        lower_bound: Lower bound for decision variables.
        upper_bound: Upper bound for decision variables.
        strength_k: Strength of bias parameter.
        rng: NumPy random generator.

    Returns:
        Initialized population array of shape (pop_size, dimension).
    """
```

## 3. Project Structure

```
scheml_2/
├── src/fertopt/          # Main source code
│   ├── core/             # Core optimization components
│   │   ├── config.py     # Configuration classes
│   │   ├── objectives.py # Objective functions
│   │   ├── problem.py   # Problem definition
│   │   └── runner.py    # Main optimization runner
│   ├── evaluation/       # Evaluation and surrogate components
│   ├── models/          # Surrogate models
│   └── operators/       # Evolutionary operators
├── experiments/         # Experiment scripts
├── tests/               # Test files
├── configs/             # Configuration files
├── artifacts/           # Output artifacts (generated)
├── docs/               # Design documents
└── .github/            # GitHub configuration
```

## 4. Key Design Patterns

### Runner Pattern

The `BaselineRunner` class orchestrates the optimization process. It handles:
- Population initialization
- Evolution loop
- Surrogate model integration
- Result saving

### Surrogate Manager

The `SurrogateManager` class handles:
- Loading pre-trained models
- Prediction (surrogate-assisted evaluation)
- Active learning updates

### Configuration-Driven

All major behaviors are controlled via `AppConfig` and YAML configuration files. Avoid adding command-line flags for things that should be in config.

## 5. Common Pitfalls to Avoid

1. **Hardcoding paths**: Use `Path(__file__).parent` or config files.
2. **Memory leaks**: Always close matplotlib figures.
3. **Mutable default arguments**: Avoid `def f(list=[])`; use `def f(list=None)`.
4. **Ignoring exceptions**: Never silently catch exceptions without logging.
5. **Missing type hints**: Always add type hints for public APIs.
