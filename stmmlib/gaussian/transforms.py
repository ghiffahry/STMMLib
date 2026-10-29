"""Parameter transforms, validation, and panel data preparation.

Handles theta layout, validation, raw<->theta transforms (log, atanh),
panel data sorting, index construction, and model specification.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from stmmlib._utils import (
    interaction_active,
    interaction_param_names,
    interaction_parse,
)
from stmmlib.registry import ModelSpec, get_model_spec

if TYPE_CHECKING:
    from stmmlib._backend import Backend

CORR_BOUND_DEFAULT = 0.95


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def make_model(
    area: str,
    time: str,
    interaction: str = "none",
    time_scope: str = "area_specific",
) -> dict:
    """Create model specification dict from component types.

    Args:
        area: Area type ("iid", "sar", "nested_iid", "nested_sar", "seblup_rao_yu").
        time: Time type ("iid", "ar1").
        interaction: Interaction type.
        time_scope: "area_specific" or "global".

    Returns:
        dict with area_type, time_type, interaction, time_scope.
    """
    return {
        "area_type": area,
        "time_type": time,
        "interaction": interaction_parse(interaction),
        "time_scope": time_scope,
    }


# ---------------------------------------------------------------------------
# Theta layout
# ---------------------------------------------------------------------------


def get_theta_layout(model: dict) -> list[str]:
    """Return ordered parameter names for model variance components.

    Order: [area params] + [time params] + [interaction params].
    Area: sigma2_v (always), rho (if SAR).
    Time: sigma2_u (if iid), sigma2_eta+phi (if AR1).
    Interaction: sigma2_gamma, rho_gamma/phi_gamma as needed.
    """
    area = model.get("area_type", model.get("area", "iid"))
    time = model.get("time_type", model.get("time", "iid"))
    interaction = model.get("interaction", "none")
    time_scope = model.get("time_scope", "area_specific")

    params = ["sigma2_e"]  # residual always first in layout

    if area in ("sar", "seblup_rao_yu"):
        params.append("sigma2_v")
        params.append("rho")
    elif area in ("nested_iid", "nested_sar"):
        params.append("sigma2_prov")
        params.append("sigma2_kab")
    else:
        params.append("sigma2_v")

    if time == "ar1":
        if time_scope == "global":
            params.append("sigma2_u")
        else:
            params.append("sigma2_eta")
        params.append("phi")
    elif time == "iid":
        if area not in ("nested_iid", "nested_sar"):
            params.append("sigma2_u")

    params.extend(interaction_param_names(interaction))
    return params


def get_param_bounds(model: dict, corr_bound: float = CORR_BOUND_DEFAULT) -> dict:
    """Return parameter bounds for optimizer.

    Returns:
        dict mapping param_name -> (lower, upper).
    """
    layout = get_theta_layout(model)
    bounds = {}
    for p in layout:
        if p in ("rho", "phi", "rho_gamma", "phi_gamma"):
            bounds[p] = (-corr_bound, corr_bound)
        elif p.startswith("sigma2_"):
            bounds[p] = (1e-10, None)  # positive
        else:
            bounds[p] = (-20.0, 20.0)
    return bounds


# ---------------------------------------------------------------------------
# Theta validation
# ---------------------------------------------------------------------------


def validate_theta(
    theta: dict,
    model: dict,
    context: str = "",
    corr_bound: float = CORR_BOUND_DEFAULT,
) -> None:
    """Validate theta: names match layout, positivity, correlation bounds.

    Raises:
        ValueError: On invalid theta.
    """
    layout = get_theta_layout(model)
    missing = [p for p in layout if p not in theta]
    extra = [p for p in theta if p not in layout and p not in ("n_params",)]
    if missing:
        raise ValueError(f"Theta missing params {missing} for {context}")
    if extra:
        warnings.warn(f"Theta has extra params {extra} (ignored) for {context}")

    for p in layout:
        val = theta[p]
        if p.startswith("sigma2_"):
            if val <= 0:
                raise ValueError(f"{p}={val} must be > 0 ({context})")
        elif p in ("rho", "phi", "rho_gamma", "phi_gamma"):
            if abs(val) >= corr_bound:
                raise ValueError(
                    f"{p}={val} must be in (-{corr_bound}, {corr_bound}) ({context})"
                )


# ---------------------------------------------------------------------------
# Raw <-> theta transforms
# ---------------------------------------------------------------------------

_CORR_PARAMS = ("rho", "phi", "rho_gamma", "phi_gamma")


def _layout_theta_to_raw(theta, layout, corr_bound: float) -> np.ndarray:
    """Single copy of the theta->raw loop shared by all layouts."""
    raw = []
    for p in layout:
        val = theta[p]
        if p.startswith("sigma2_"):
            raw.append(np.log(val))
        elif p in _CORR_PARAMS:
            raw.append(np.arctanh(val / corr_bound))
        else:
            raw.append(val)
    return np.array(raw, dtype=np.float64)


def _layout_raw_to_theta(raw, layout, corr_bound: float) -> dict:
    """Single copy of the raw->theta loop shared by all layouts."""
    raw = np.asarray(raw, dtype=np.float64).ravel()
    theta = {}
    for i, p in enumerate(layout):
        if p.startswith("sigma2_"):
            theta[p] = np.exp(raw[i])
        elif p in _CORR_PARAMS:
            theta[p] = corr_bound * np.tanh(raw[i])
        else:
            theta[p] = raw[i]
    return theta


def theta_to_raw(
    theta: dict,
    model: dict,
    corr_bound: float = CORR_BOUND_DEFAULT,
) -> np.ndarray:
    """Transform constrained theta to unconstrained raw vector.

    Variance params: log transform.
    Correlation params: atanh transform.
    """
    return _layout_theta_to_raw(theta, get_theta_layout(model), corr_bound)


def raw_to_theta(
    raw: np.ndarray,
    model: dict,
    corr_bound: float = CORR_BOUND_DEFAULT,
) -> dict:
    """Transform unconstrained raw vector back to constrained theta.

    Inverse of theta_to_raw.
    """
    layout = get_theta_layout(model)
    raw = np.asarray(raw, dtype=np.float64).ravel()
    if len(raw) != len(layout):
        raise ValueError(f"raw length {len(raw)} != layout length {len(layout)}")
    return _layout_raw_to_theta(raw, layout, corr_bound)


# ---------------------------------------------------------------------------
# Panel data preparation
# ---------------------------------------------------------------------------


def normalize_index(x, label: str = "index"):
    """Convert arbitrary labels to integer index (0-based).

    Args:
        x: Array-like of labels.
        label: Name for error messages.

    Returns:
        (int_array, levels_list) tuple.
    """
    import pandas as pd

    if isinstance(x, pd.Series):
        x = x.values
    elif not isinstance(x, np.ndarray):
        x = np.asarray(x)

    levels = list(dict.fromkeys(x))  # unique, preserve order
    mapping = {v: i for i, v in enumerate(levels)}
    idx = np.array([mapping[v] for v in x], dtype=np.int32)
    return idx, levels


def detect_panel_structure(area_index, time_index) -> dict:
    """Detect balanced/unbalanced panel and count dimensions.

    Args:
        area_index: Integer array (0-based).
        time_index: Integer array (0-based).

    Returns:
        dict with 'n', 'm', 'T', 'balanced', 'n_cells', 'area_counts'.
    """
    area_index = np.asarray(area_index, dtype=np.int32)
    time_index = np.asarray(time_index, dtype=np.int32)
    n = len(area_index)
    m = int(np.max(area_index)) + 1
    T = int(np.max(time_index)) + 1

    # Check balance: each area should have T observations
    area_counts = np.bincount(area_index, minlength=m)
    balanced = bool(np.all(area_counts == T))

    return {
        "n": n,
        "m": m,
        "T": T,
        "balanced": balanced,
        "n_cells": m * T,
        "area_counts": area_counts,
    }


def prepare_panel_data(
    data,
    area_col: str = "area",
    time_col: str = "time",
    prov_col: str = None,
    kab_col: str = "kab",
    sort: bool = True,
    verbose: bool = False,
) -> dict:
    """Sort data, build indices, detect panel structure.

    Args:
        data: pandas DataFrame.
        area_col: Column name for area.
        time_col: Column name for time.
        prov_col: Column name for province (nested models).
        kab_col: Column name for district (nested models).
        sort: Sort data by area, time.
        verbose: Print summary.

    Returns:
        dict with 'data', 'n', 'm', 'T', 'area_index', 'time_index',
        'area_labels', 'time_labels', 'balanced', plus nested indices if applicable.
    """
    import pandas as pd

    df = data.copy()
    if sort:
        sort_cols = [area_col, time_col]
        if prov_col and prov_col in df.columns:
            sort_cols.insert(0, prov_col)
        if kab_col and kab_col in df.columns:
            sort_cols.insert(1 if prov_col else 0, kab_col)
        df = df.sort_values(sort_cols).reset_index(drop=True)

    area_idx, area_labels = normalize_index(df[area_col].values, area_col)
    time_idx, time_labels = normalize_index(df[time_col].values, time_col)
    panel = detect_panel_structure(area_idx, time_idx)

    result = {
        "data": df,
        "area_index": area_idx,
        "time_index": time_idx,
        "area_labels": area_labels,
        "time_labels": time_labels,
        **panel,
        "W": None,
    }

    # Nested indices
    if prov_col and prov_col in df.columns:
        prov_idx, prov_labels = normalize_index(df[prov_col].values, prov_col)
        kab_idx, kab_labels = normalize_index(df[kab_col].values, kab_col)
        result["prov_index"] = prov_idx
        result["prov_labels"] = prov_labels
        result["kab_index"] = kab_idx
        result["kab_labels"] = kab_labels
        result["n_prov"] = len(prov_labels)
        result["n_kab"] = len(kab_labels)

    if verbose:
        print(
            f"Panel: n={panel['n']}, m={panel['m']}, T={panel['T']}, "
            f"balanced={panel['balanced']}"
        )

    return result


# ---------------------------------------------------------------------------
# SEBLUP5 transforms
# ---------------------------------------------------------------------------


def seblup5_layout(interaction: str, area_type: str) -> list[str]:
    """Parameter layout for SEBLUP5 model."""
    params = ["sigma2_e", "sigma2_v", "sigma2_eta", "phi"]
    if area_type in ("sar",):
        params.append("rho")
    params.extend(interaction_param_names(interaction))
    return params


def seblup5_theta_to_raw(theta, interaction, area_type, corr_bound=CORR_BOUND_DEFAULT):
    return _layout_theta_to_raw(theta, seblup5_layout(interaction, area_type), corr_bound)


def seblup5_raw_to_theta(raw, interaction, area_type, corr_bound=CORR_BOUND_DEFAULT):
    return _layout_raw_to_theta(raw, seblup5_layout(interaction, area_type), corr_bound)


# ---------------------------------------------------------------------------
# Nested transforms
# ---------------------------------------------------------------------------


def nested_layout(interaction: str) -> list[str]:
    params = ["sigma2_e", "sigma2_prov", "sigma2_kab"]
    params.extend(interaction_param_names(interaction))
    return params


def nested_theta_to_raw(theta, interaction, corr_bound=CORR_BOUND_DEFAULT):
    return _layout_theta_to_raw(theta, nested_layout(interaction), corr_bound)


def nested_raw_to_theta(raw, interaction, corr_bound=CORR_BOUND_DEFAULT):
    return _layout_raw_to_theta(raw, nested_layout(interaction), corr_bound)


# ---------------------------------------------------------------------------
# Nested AR1 transforms
# ---------------------------------------------------------------------------


def nested_ar1_layout(interaction: str) -> list[str]:
    params = ["sigma2_e", "sigma2_prov", "sigma2_kab", "sigma2_eta", "phi"]
    params.extend(interaction_param_names(interaction))
    return params


def nested_ar1_theta_to_raw(theta, interaction, corr_bound=CORR_BOUND_DEFAULT):
    return _layout_theta_to_raw(theta, nested_ar1_layout(interaction), corr_bound)


def nested_ar1_raw_to_theta(raw, interaction, corr_bound=CORR_BOUND_DEFAULT):
    return _layout_raw_to_theta(raw, nested_ar1_layout(interaction), corr_bound)


# ---------------------------------------------------------------------------
# Seed preservation
# ---------------------------------------------------------------------------


def _with_preserved_seed(seed=None, code=None):
    """Execute code with seed, preserving RNG state."""
    rng_state = np.random.get_state()
    if seed is not None:
        np.random.seed(seed)
    try:
        result = code() if callable(code) else None
        return result
    finally:
        np.random.set_state(rng_state)
