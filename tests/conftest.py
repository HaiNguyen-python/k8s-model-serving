from __future__ import annotations

import pytest

from canary_serving.model import ModelBundle, train_all


@pytest.fixture(scope="session")
def bundles() -> dict[str, ModelBundle]:
    """Smaller training run than the default: same code path, faster tests."""
    return train_all(seed=42, n_train=3000, n_test=1500)
