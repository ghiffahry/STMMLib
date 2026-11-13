"""Main user-facing functions: stmm(), stmm_normal(), stmm_beta().

Unified entry point that dispatches to Gaussian or Beta backends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import numpy as np
    from stmmlib._backend import Backend


def _reject_formula(formula) -> None:
    """Fail loudly instead of silently ignoring an unimplemented formula."""
    if formula is not None:
        raise NotImplementedError(
            "formula is not implemented. Predictors are built from every "
            "column that is not the response or a grouping column. Pass "
            "formula=None and remove unwanted predictor columns from data.")


def stmm(
    formula: str = None,
    data: pd.DataFrame = None,
    model: str = "m4_sar_ar1",
    family: str = "gaussian",
    W=None,
    method: str = "REML",
    gpu: str = "auto",
    n_starts: int = 3,
    preset: str | None = None,
    response_col: str = "y",
    area_col: str = "area",
    time_col: str = "time",
    prov_col: str | None = "prov",
    kab_col: str | None = "kab",
    config=None,
    inference: str = "wald",
    lrt_terms: "list | None" = None,
    lrt_deadline_s: "float | None" = None,
    **kwargs,
):
    """Fit a spatio-temporal mixed model (Gaussian or Beta).

    Column names are configurable so any dataset works without renaming:
    pass ``response_col``/``area_col``/``time_col`` (and optionally
    ``prov_col``/``kab_col``) matching your own column names.

    W ordering contract: row/col i of W corresponds to the i-th area in
    SORTED order (pd.Categorical sorts categories before assigning codes).

    Args:
        formula: must be None. Not implemented; X is built from every
            non-skipped data column.
        data: pandas DataFrame.
        model: Model ID (e.g. "m4_sar_ar1" for Gaussian, "m1_iid_iid" for Beta).
        family: "gaussian" or "beta".
        W: Spatial weight matrix.
        method: "ML", "REML", or "Laplace".
        gpu: "auto", "yes", or "no".
        n_starts: Number of random restarts.
        preset: accuracy preset for the Beta family ("fast", "balanced",
            "medium", "full"; default "full"). fast/low: ftol=1e-5, gtol=1e-5,
            no Hessian, max_iter cap 500 (exploratory; no SEs).
            balanced/medium-tol: ftol=1e-8, gtol=1e-6, Hessian on, max_iter
            cap 500 (cheap outer + honest SEs). medium/standard:
            ftol=1e-8, gtol=1e-6, no Hessian, no cap. full/high: ftol=1e-12,
            gtol=1e-6, Hessian on (SEs + PD certificate).
            Anchors: L-BFGS-B factr scale (Zhu-Byrd-Nocedal 1997);
            lme4 FtolAbs default 1e-5 and strict_tol ftol_abs=1e-8
            (Bates et al.).
        response_col: name of the response column (default "y").
        area_col: name of the area column (default "area").
        time_col: name of the time column (default "time").
        prov_col: optional province column (default "prov").
        kab_col: optional district column (default "kab").
        config: BetaFitConfig (Beta only). Opt-in weak-prior MAP penalty:
            phi_prior_mean/sd on logphi, dep_prior_sd on raw_dep (all None
            = off, default). MAP with weak Gaussian prior; precedent Chung
            et al. 2013 blme and brglm2 MPL Jeffreys. Always report
            penalized alongside the unpenalized solution. Auto MAP refit
            on boundary: when config is None (no explicit config sent) and
            the unpenalized Beta fit is invalid with log_phi or raw_dep at
            the box bounds, one automatic weak-prior refit runs (blme
            style, Chung et al. 2013); the unpenalized result is kept in
            optim["unpenalized"]. Disable by passing an explicit config
            (even BetaFitConfig()).
        **kwargs: Additional arguments passed to backend.

    Returns:
        StMMResult (Gaussian) or BetastmmResult (Beta).
    """
    if family.lower() == "beta":
        from stmmlib.registry import MODELS
        spec = MODELS.get(model)
        if spec is not None and spec.n_params_beta == 0:
            raise ValueError(
                f"model {model!r} is not available for family='beta' "
                f"(it has no Beta structure).")
        return stmm_beta(formula, data, model, W, method="Laplace", gpu=gpu,
                         n_starts=n_starts, preset=preset,
                         response_col=response_col,
                         area_col=area_col, time_col=time_col,
                         prov_col=prov_col, kab_col=kab_col, config=config,
                         inference=inference, lrt_terms=lrt_terms,
                         lrt_deadline_s=lrt_deadline_s, **kwargs)
    else:
        from stmmlib.registry import MODELS
        spec = MODELS.get(model)
        if spec is not None and spec.n_params_normal == 0:
            raise ValueError(
                f"model {model!r} is not available for family='gaussian' "
                f"(it has no Gaussian structure); use family='beta'.")
        return stmm_normal(formula, data, model, W, method=method, gpu=gpu,
                           n_starts=n_starts, response_col=response_col,
                           area_col=area_col, time_col=time_col,
                           prov_col=prov_col, kab_col=kab_col,
                           inference=inference, lrt_terms=lrt_terms,
                           lrt_deadline_s=lrt_deadline_s, **kwargs)


def stmm_normal(
    formula: str = None,
    data: pd.DataFrame = None,
    model: str = "m4_sar_ar1",
    W=None,
    method: str = "REML",
    gpu: str = "auto",
    n_starts: int = 3,
    response_col: str = "y",
    area_col: str = "area",
    time_col: str = "time",
    prov_col: str | None = "prov",
    kab_col: str | None = "kab",
    spatial: str = None,
    temporal: str = None,
    interaction: str = None,
    inference: str = "wald",
    lrt_terms: "list | None" = None,
    lrt_deadline_s: "float | None" = None,
    **kwargs,
):
    """Fit a Gaussian spatio-temporal mixed model.

    ``model`` may be a registry model id (m1-m14); it is translated into
    the matching structural args and routed to the right backend:
    seblup models -> seblup_rao_yu_it(), nested models -> nested wrappers,
    everything else -> stmm_normal_core(). Explicit ``spatial``/``temporal``/
    ``interaction`` overrides still win when provided.

    Pass explicit ``area_col``/``time_col`` (and optionally ``spatial``/
    ``temporal``/``prov_col``) when your data uses non-default names.
    """
    _reject_formula(formula)
    from stmmlib.registry import MODELS
    from stmmlib._utils import interaction_parse
    from stmmlib._backend import Backend

    spec = MODELS.get(model)
    if spec is not None:
        _spatial_from_spec = spatial is None
        _temporal_from_spec = temporal is None
        if spatial is None:
            spatial = spec.area
        if temporal is None:
            temporal = spec.time
        if interaction is None:
            interaction = spec.interaction
        # Honor the registry's time_scope when the model was resolved from
        # the registry without overrides; an explicit override wins and the
        # scope falls back to the sar+ar1 derivation.
        time_scope = spec.time_scope if (_spatial_from_spec
                                         and _temporal_from_spec) else None
        known = ("iid", "sar", "nested_iid", "nested_sar", "seblup_rao_yu")
        if spec.area not in known:
            raise ValueError(
                f"model {model!r}: area type {spec.area!r} tidak didukung "
                f"untuk family gaussian; tipe yang dikenal: {list(known)}")
        if spec.time not in ("iid", "ar1"):
            raise ValueError(
                f"model {model!r}: time type {spec.time!r} tidak didukung "
                f"untuk family gaussian; tipe yang dikenal: ['iid', 'ar1']")
    else:
        time_scope = None

    _gpu = {"yes": True, "no": False}.get(str(gpu).lower(), "auto")
    backend = Backend(gpu=_gpu)

    if spatial == "seblup_rao_yu":
        from stmmlib.gaussian.models import seblup_rao_yu_it
        result = seblup_rao_yu_it(formula, data, area_col, time_col, W=W,
                                  interaction=interaction_parse(interaction),
                                  method=method, n_starts=n_starts,
                                  response_col=response_col,
                                  backend=backend, time_scope=time_scope,
                                  **kwargs)
    elif spatial in ("nested_iid", "nested_sar"):
        from stmmlib.gaussian.models import nested_iid_iid, nested_iid_ar1
        wrapper = nested_iid_ar1 if temporal == "ar1" else nested_iid_iid
        result = wrapper(formula, data, area_col, time_col, prov_col, kab_col,
                         interaction=interaction_parse(interaction), W=W,
                         method=method, n_starts=n_starts,
                         response_col=response_col,
                         backend=backend, time_scope=time_scope,
                         **kwargs)
    else:
        from stmmlib.gaussian.models import stmm_normal_core
        result = stmm_normal_core(formula, data, area_col, time_col,
                                   spatial=spatial, temporal=temporal,
                                   interaction=interaction_parse(interaction), W=W,
                                   method=method, n_starts=n_starts,
                                   response_col=response_col,
                                   backend=backend, time_scope=time_scope,
                                   **kwargs)

    if str(inference).lower() == "lrt":
        # The LRT re-fits the model with ML (REML likelihoods are not
        # comparable across designs) and leaves the fit above untouched.
        from stmmlib.gaussian.lrt_fixed import (attach_lrt_inference,
                                                lrt_fixed_gaussian)
        lrt = lrt_fixed_gaussian(result, lrt_terms, backend=backend,
                                 deadline_s=lrt_deadline_s)
        attach_lrt_inference(result, lrt)
    return result


def stmm_beta(
    formula: str = None,
    data: pd.DataFrame = None,
    model: str = "m1_iid_iid",
    W=None,
    method: str = "Laplace",
    gpu: str = "auto",
    n_starts: int = 3,
    preset: str | None = None,
    response_col: str = "y",
    area_col: str = "area",
    time_col: str = "time",
    prov_col: str | None = "prov",
    kab_col: str | None = "kab",
    config=None,
    inference: str = "wald",
    lrt_terms: "list | None" = None,
    lrt_deadline_s: "float | None" = None,
    **kwargs,
):
    """Fit a Beta spatio-temporal mixed model.

    preset="fast": skip the FD Hessian + cap optimizer effort (results
    identical per benchmark; no SEs and hessian_pd=None in the certificate).
    preset="balanced": medium tolerances + Hessian + cap 500 (cheap outer
    with honest SEs). Default (None/"full") keeps SEs + full certificate.

    Auto MAP refit on boundary: with config=None (no explicit config sent),
    an unpenalized fit that ends invalid with log_phi or raw_dep at the box
    bounds is refit once with a weak Gaussian prior (blme-style, Chung et
    al. 2013); the unpenalized result is stored in optim["unpenalized"].
    Disable by passing an explicit config (even BetaFitConfig()).

    Dispatches to the Beta backend modules in beta/.

    Pass ``response_col``/``area_col``/``time_col`` to match your own
    column names instead of the conventional ``y``/``area``/``time``.
    """
    _reject_formula(formula)
    from stmmlib.beta.fit import stmm_beta as _stmm_beta
    return _stmm_beta(formula, data, model, W, method=method, gpu=gpu,
                      n_starts=n_starts, preset=preset,
                      response_col=response_col,
                      area_col=area_col, time_col=time_col,
                      prov_col=prov_col, kab_col=kab_col, config=config,
                      inference=inference, lrt_terms=lrt_terms,
                      lrt_deadline_s=lrt_deadline_s, **kwargs)
