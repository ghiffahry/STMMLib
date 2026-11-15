"""Universal smoke matrix: 14 models x 2 families, arbitrary column names.

Run manually from the repository root, with the root importable so the local
tree (not an installed copy) is exercised:

    PYTHONPATH=. python tests/universal_matrix.py [--family gaussian|beta]
                     [--models m1_iid_iid,m4_sar_ar1] [--no-shuffle]

PowerShell: ``$env:PYTHONPATH = (Get-Location).Path;`` first.

Not collected by pytest (name does not start with test_): this is a
development harness, not a CI gate. Every model is fitted on three panels of
different sizes with different column names, including string area labels and
non-contiguous years, so a hardcoded column or index assumption fails here.
"""

from __future__ import annotations

import argparse
import traceback

import numpy as np
import pandas as pd

from stmmlib.registry import stmm_available_models
from stmmlib.stmm import stmm

SAR = {"m3_sar_iid", "m4_sar_ar1", "m5_seblup_rao_yu_it",
       "m9_sar_iid_int", "m11_sar_ar1_int", "m12_seblup_rao_yu_it_int"}
NESTED = {"m6_nested_iid_iid", "m7_nested_ar1",
          "m13_nested_iid_iid_int", "m14_nested_ar1_int"}

# Three panels: small/short, large/long, medium/short-T. Names change every
# config so no model can depend on "area"/"time"/"y"/"x1".
CONFIGS = [
    dict(m=6, T=4, seed=11, year0=2015,
         area="kec", time="thn", resp="respon",
         xs=("belanja", "ipm"), prov="provinsi", kab="kabupaten"),
    dict(m=12, T=7, seed=29, year0=2001,
         area="region_id", time="year_id", resp="target",
         xs=("z1", "z2"), prov="prov_id", kab="kab_id"),
    dict(m=9, T=3, seed=5, year0=1998,
         area="a", time="t", resp="r",
         xs=("x_a",), prov="p", kab="k"),
]


def _softmax_weights(m, rng):
    raw = rng.random((m, m))
    np.fill_diagonal(raw, 0.0)
    return raw / raw.sum(1, keepdims=True)


def panel(cfg, family):
    """Build one toy panel with the config's column names.

    Area labels are zero-padded strings so sort order is the label order and
    W rows line up with the sorted categories. Years are offset and rows are
    shuffled: the fit must sort internally.
    """
    rng = np.random.default_rng(cfg["seed"])
    m, T = cfg["m"], cfg["T"]
    n = m * T
    area_codes = np.repeat(np.arange(m), T)
    time_codes = np.tile(np.arange(T), m)

    prov_codes = area_codes // max(1, m // 3)
    kab_codes = area_codes

    n_prov = int(prov_codes.max()) + 1
    pef = rng.normal(0.0, 0.25, size=n_prov)[prov_codes]
    kef = rng.normal(0.0, 0.35, size=m)[area_codes]
    tef = rng.normal(0.0, 0.2, size=T)[time_codes]

    x = rng.normal(size=(n, len(cfg["xs"])))
    eta = -0.6 + x @ np.linspace(0.4, -0.3, x.shape[1]) + pef + kef + tef

    if family == "gaussian":
        y = eta + rng.normal(0.0, 0.6, size=n)
    else:
        mu = np.clip(1.0 / (1.0 + np.exp(-eta)), 0.05, 0.95)
        phi = 18.0
        y = rng.beta(mu * phi, (1.0 - mu) * phi)

    df = pd.DataFrame({
        cfg["resp"]: y,
        cfg["area"]: [f"A{i:02d}" for i in area_codes],
        cfg["time"]: cfg["year0"] + time_codes,
    })
    for j, name in enumerate(cfg["xs"]):
        df[name] = x[:, j]
    df[cfg["prov"]] = [f"P{i:02d}" for i in prov_codes]
    df[cfg["kab"]] = [f"K{i:03d}" for i in kab_codes]

    # Area labels sort as A00..A{m-1}; W row i maps to that sorted order.
    W = _softmax_weights(m, rng)
    return df.sample(frac=1.0, random_state=cfg["seed"]).reset_index(drop=True), W


def fit(model, family, cfg, df, W, *, shuffle_check):
    kwargs = dict(model=model, family=family, gpu="no", n_starts=1,
                  response_col=cfg["resp"], area_col=cfg["area"],
                  time_col=cfg["time"],
                  # Declare the grouping columns even for flat models: a frame
                  # may carry them, and undeclared text columns become
                  # predictors (numeric conversion error downstream).
                  prov_col=cfg["prov"], kab_col=cfg["kab"])
    if family == "beta":
        kwargs.update(max_iter=400, inner_max_iter=30)
    if model in SAR:
        kwargs["W"] = W

    r = stmm(data=df, **kwargs)
    if family == "gaussian":
        ll = float(r["log_lik"])
        beta = np.asarray(r["beta_hat"], float)
        se = np.asarray(r["se_beta"], float)
    else:
        ll = float(r._opt["log_lik"])
        beta = np.asarray(r.beta, float)
        se = np.asarray(r.inference["se"], float)
    out = {"ll": ll, "beta": beta, "se": se}

    if shuffle_check:
        df2 = df.sample(frac=1.0, random_state=cfg["seed"] + 1).reset_index(drop=True)
        r2 = stmm(data=df2, **kwargs)
        ll2 = float(r2["log_lik"] if family == "gaussian" else r2._opt["log_lik"])
        out["ll_shuffled"] = ll2
    return out


def check(family, res):
    """Return a list of failure strings (empty when the fit is sane)."""
    bad = []
    if not np.isfinite(res["ll"]):
        bad.append(f"log_lik not finite: {res['ll']}")
    if not np.all(np.isfinite(res["beta"])):
        bad.append("beta_hat not finite")
    if not np.all(np.isfinite(res["se"])):
        bad.append("se not finite")
    elif np.any(res["se"] < 0):
        bad.append("negative SE")
    if "ll_shuffled" in res:
        gap = abs(res["ll"] - res["ll_shuffled"])
        # Beta: the standardisation means are summed in frame order, so a
        # permutation moves x_scale at 1e-16; the auto-prior refit is a
        # threshold branch on the likelihood and can flip near its boundary,
        # which lands ~1e-4 from the same optimum. Gaussian is exactly
        # invariant because it now sorts the frame before computing anything.
        tol = (1e-6 * max(1.0, abs(res["ll"])) if family == "gaussian"
               else 2e-3)
        if gap > tol:
            bad.append(f"row-order dependent: dloglik={gap:.3e} > {tol:.1e}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["gaussian", "beta", "both"], default="both")
    ap.add_argument("--models", default="")
    ap.add_argument("--no-shuffle", action="store_true")
    args = ap.parse_args()

    families = ["gaussian", "beta"] if args.family == "both" else [args.family]
    models = [m.strip() for m in args.models.split(",") if m.strip()] \
        or stmm_available_models()
    failures = []
    for family in families:
        for cfg_i, cfg in enumerate(CONFIGS):
            df, W = panel(cfg, family)
            label = f"{cfg['m']}x{cfg['T']} seed={cfg['seed']}"
            for model in models:
                try:
                    res = fit(model, family, cfg, df, W,
                              shuffle_check=not args.no_shuffle)
                    bad = check(family, res)
                    state = "ok" if not bad else "FAIL " + "; ".join(bad)
                    print(f"{family:8s} {model:28s} {label:20s} "
                          f"ll={res['ll']:10.4f} {state}", flush=True)
                    if bad:
                        failures.append((family, model, label, "; ".join(bad)))
                except Exception as exc:  # noqa: BLE001 - harness reports everything
                    print(f"{family:8s} {model:28s} {label:20s} EXC "
                          f"{type(exc).__name__}: {exc}", flush=True)
                    failures.append((family, model, label,
                                     f"{type(exc).__name__}: {exc}"))
                    traceback.print_exc(limit=2)

    print()
    if failures:
        print(f"{len(failures)} failure(s):")
        for fam, model, label, msg in failures:
            print(f"  [{fam}] {model} {label}: {msg}")
        return 1
    print("all fits finite; row order invariant where checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
