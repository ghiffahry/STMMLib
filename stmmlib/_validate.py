"""Argument contracts shared by the Gaussian and Beta entry points.

A typo used to fall through to a default: ``family='poisson'`` fitted a
Gaussian model, ``method='reml'`` fitted ML, a duplicate predictor produced
``inf``, and a Beta SAR without ``W`` returned NaN. These checks run before
dispatch so the failing argument is named instead of the optimizer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_FAMILY_ALIASES = {"gaussian": "gaussian", "normal": "gaussian", "beta": "beta"}
_METHODS = {
    "gaussian": {"ml": "ML", "reml": "REML"},
    "beta": {"ml": "ML", "laplace": "Laplace"},
}


def normalize_family(family) -> str:
    """Return 'gaussian' or 'beta'; 'normal' is a Gaussian alias."""
    key = str(family).strip().lower()
    if key not in _FAMILY_ALIASES:
        raise ValueError(
            f"family harus 'gaussian' (alias 'normal') atau 'beta', "
            f"diterima {family!r}.")
    return _FAMILY_ALIASES[key]


def normalize_method(method, family: str) -> str:
    """Return the canonical method name for one family.

    Gaussian: ML or REML. Beta: ML or Laplace (REML does not exist there).
    """
    key = str(method).strip().lower()
    table = _METHODS[family]
    if key not in table:
        valid = sorted(set(table.values()))
        hint = ("; REML tidak tersedia untuk family Beta"
                if family == "beta" else "")
        raise ValueError(
            f"method {method!r} tidak dikenal untuk family {family!r}; "
            f"pilihan: {valid}{hint}.")
    return table[key]


def normalize_inference(inference) -> str:
    """Return 'wald' or 'lrt'."""
    key = str(inference).strip().lower()
    if key not in ("wald", "lrt"):
        raise ValueError(
            f"inference harus 'wald' atau 'lrt', diterima {inference!r}.")
    return key


def check_n_starts(n_starts) -> int:
    """Return a positive int; 0 or -1 must not silently mean one start."""
    if isinstance(n_starts, bool) or not isinstance(n_starts, (int, np.integer)):
        raise ValueError(
            f"n_starts harus bilangan bulat >= 1, diterima {n_starts!r}.")
    n = int(n_starts)
    if n < 1:
        raise ValueError(
            f"n_starts harus bilangan bulat >= 1, diterima {n_starts!r}.")
    return n


def check_model(model, *, allow_none: bool = False):
    """Return the model id, or raise with the available ids."""
    if model is None and allow_none:
        return None
    from stmmlib.registry import MODELS
    if model not in MODELS:
        raise ValueError(
            f"model {model!r} tidak dikenal; pilihan: {sorted(MODELS)}")
    return model


def check_frame(data, response_col=None) -> None:
    """Reject a missing/empty frame and duplicate column names.

    A duplicated predictor name reached the design matrix as two identical
    columns; the Gaussian fit then reported ``log_lik = inf``.
    """
    if data is None:
        raise ValueError("data harus diberikan (pandas DataFrame).")
    if not isinstance(data, pd.DataFrame):
        raise ValueError(
            f"data harus pandas DataFrame, diterima {type(data).__name__}.")
    dup = data.columns[data.columns.duplicated()].tolist()
    if dup:
        raise ValueError(
            f"nama kolom duplikat: {dup}; setiap kolom harus unik.")
    if len(data) == 0:
        raise ValueError("data kosong; tidak ada baris untuk difit.")
    if response_col is not None:
        y = response_col or "y"
        if y not in data.columns:
            raise ValueError(
                f"kolom respons {response_col!r} tidak ditemukan; pass "
                "response_col=<nama>.")


def check_grouping_columns(data, *cols) -> None:
    """Area and time columns must exist; a typo used to surface as KeyError."""
    missing = [c for c in cols if c and c not in data.columns]
    if missing:
        raise ValueError(
            f"kolom pengelompokan tidak ditemukan: {missing}; pass area_col=/"
            "time_col= sesuai nama kolom di data.")


def check_W(W, n_area: "int | None" = None) -> None:
    """Require a square, finite weight matrix matching the area count."""
    if W is None:
        return
    arr = W.toarray() if hasattr(W, "toarray") else np.asarray(W, dtype=float)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"W harus matriks persegi, diterima shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("W mengandung NaN/Inf.")
    if n_area is not None and arr.shape[0] != n_area:
        raise ValueError(
            f"shape W {arr.shape} tidak cocok dengan {n_area} area.")


def check_nested_columns(data, prov_col, kab_col) -> None:
    """Nested models need both grouping columns present by name."""
    missing = [label for label, col in (("prov", prov_col), ("kab", kab_col))
               if not col or col not in data.columns]
    if missing:
        raise ValueError(
            "model nested butuh kolom provinsi dan kabupaten; pass prov_col= "
            f"dan kab_col= (default 'prov'/'kab'); tidak ditemukan: {missing}.")


_BETA_KWARGS = frozenset({
    "max_iter", "inner_max_iter", "start_theta", "seed",
    "compute_hessian", "compute_posterior",
})


def reject_unknown_beta_kwargs(kwargs) -> None:
    """The Beta backend ignores unknown kwargs; catch typos there instead."""
    extra = sorted(set(kwargs) - _BETA_KWARGS)
    if extra:
        raise ValueError(
            f"argumen tidak dikenal: {extra}; tambahan yang diterima: "
            f"{sorted(_BETA_KWARGS)}.")


def check_normal_input(data, response_col, area_col, time_col, W, spatial,
                       prov_col, kab_col) -> None:
    """Preflight for the Gaussian entry point."""
    check_frame(data, response_col)
    check_grouping_columns(data, area_col, time_col)
    n_area = int(data[area_col].nunique())
    check_W(W, n_area)
    if spatial in ("nested_iid", "nested_sar"):
        check_nested_columns(data, prov_col, kab_col)


def check_beta_input(data, response_col, W, model, area_col, time_col,
                     prov_col, kab_col) -> None:
    """Preflight for the Beta entry point: frame plus the SAR weight matrix."""
    import warnings
    check_frame(data, response_col)
    from stmmlib.registry import MODELS
    spec = MODELS[model]
    if spec.model_id != "beta_regression":
        check_grouping_columns(data, area_col, time_col)
    if spec.area in ("sar", "seblup_rao_yu") and W is None:
        raise ValueError(
            f"model {model!r} butuh matriks bobot spasial W; teruskan W=... "
            "(tanpa W hasilnya NaN, bukan error).")
    if spec.area not in ("nested_iid", "nested_sar"):
        skip = {c for c in (prov_col, kab_col) if c}
        for name in ("prov", "kab"):
            if name in data.columns and name in skip:
                warnings.warn(
                    f"kolom {name!r} tidak masuk desain regresi model ini; "
                    "rename kolom prediktor itu bila Anda bermaksud "
                    "memakainya sebagai prediktor.",
                    UserWarning, stacklevel=4)
