"""Test fixtures for pipeline monitoring."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_baseline() -> pd.DataFrame:
    """Create a baseline dataset."""
    np.random.seed(42)
    return pd.DataFrame({
        "feature1": np.random.normal(100, 15, 200),
        "feature2": np.random.normal(50, 5, 200),
        "feature3": np.random.choice(["A", "B", "C"], 200),
    })


@pytest.fixture
def sample_target() -> pd.Series:
    """Create a target variable."""
    np.random.seed(42)
    return pd.Series(np.random.binomial(1, 0.5, 200), name="target")


@pytest.fixture
def sample_current_no_drift() -> pd.DataFrame:
    """Create current data with no drift."""
    np.random.seed(123)
    return pd.DataFrame({
        "feature1": np.random.normal(100, 15, 150),
        "feature2": np.random.normal(50, 5, 150),
        "feature3": np.random.choice(["A", "B", "C"], 150),
    })


@pytest.fixture
def sample_current_with_drift() -> pd.DataFrame:
    """Create current data with drift."""
    np.random.seed(456)
    return pd.DataFrame({
        "feature1": np.random.normal(120, 20, 150),  # Shifted mean
        "feature2": np.random.normal(50, 5, 150),
        "feature3": np.random.choice(["A", "B", "X"], 150),  # Different category
    })


@pytest.fixture
def sample_target_with_drift() -> pd.Series:
    """Create target with drift."""
    np.random.seed(456)
    return pd.Series(np.random.binomial(1, 0.8, 150), name="target")
