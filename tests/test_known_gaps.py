"""Known gaps that need a statistical decision, not a silent patch."""

import pytest

from stmmlib.gaussian.transforms import nested_layout
from stmmlib.registry import get_model_spec


@pytest.mark.xfail(
    strict=True,
    reason=(
        "M6 declares 4 parameters and its equation has a separate time "
        "variance, but nested_layout has 3 and build_V_nested reuses "
        "sigma2_kab for the IID time block. Fixing it changes M6/M13 "
        "results, so it needs an explicit decision."
    ),
)
def test_m6_declares_a_separate_time_variance():
    spec = get_model_spec("m6_nested_iid_iid")
    assert len(nested_layout(spec.interaction)) == spec.n_params_normal
    assert "sigma2_eta" in nested_layout(spec.interaction)
