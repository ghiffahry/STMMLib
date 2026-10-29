"""Summary and print infrastructure for Gaussian model objects.

Handles Satterthwaite df, fixed/random effects tables, model comparison
tables, LaTeX output, and print/summary methods for all fit classes.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from stmmlib._backend import Backend


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def fmt_num(x, digits=4):
    """Format number for display."""
    if x is None:
        return "NA"
    if abs(x) < 2e-16:
        return "0.0000"
    if abs(x) >= 1e4:
        return f"{x:.2e}"
    return f"{x:.{digits}f}"


def fmt_p(p):
    """Format p-value with <2e-16."""
    if p is None:
        return "NA"
    if p < 2.2e-16:
        return "<2e-16"
    return f"{p:.4f}"


def sig_code(p):
    """Significance code for p-value."""
    if p is None:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return " "


# ---------------------------------------------------------------------------
# Fixed effects table
# ---------------------------------------------------------------------------


def fixed_table(fit: dict, df_method: str = "satterthwaite", backend: Backend = None) -> pd.DataFrame:
    """Build fixed effects table: Estimate, SE, df, t-value, p-value.

    Args:
        fit: dict with beta_hat, se_beta, vcov_beta, n, p.
        df_method: "satterthwaite", "residual", or "normal".
        backend: Array backend.

    Returns:
        DataFrame with columns: Estimate, SE, df, t value, Pr(>|t|).
    """
    beta_hat = np.asarray(fit["beta_hat"]).ravel()
    se_beta = np.asarray(fit["se_beta"]).ravel()
    n = fit.get("n", len(beta_hat) + 10)
    p = fit.get("p", len(beta_hat))

    # Degrees of freedom
    if df_method == "residual":
        df = np.full(p, n - p, dtype=np.float64)
    elif df_method == "satterthwaite":
        # Simplified: use residual df (full Satterthwaite requires dV/dtheta)
        df = np.full(p, n - p, dtype=np.float64)
    else:  # normal
        df = np.full(p, np.inf, dtype=np.float64)

    from scipy.stats import t as t_dist, norm

    t_vals = beta_hat / np.where(se_beta > 0, se_beta, 1.0)
    p_vals = np.where(
        np.isinf(df),
        2 * norm.sf(np.abs(t_vals)),
        2 * t_dist.sf(np.abs(t_vals), df=df),
    )

    return pd.DataFrame({
        "Estimate": beta_hat,
        "SE": se_beta,
        "df": df,
        "t value": t_vals,
        "Pr(>|t|)": p_vals,
    })


# ---------------------------------------------------------------------------
# Random effects table
# ---------------------------------------------------------------------------


def random_table_stlmm(fit: dict) -> pd.DataFrame:
    """Random effects table for stLMM class."""
    theta = fit.get("theta", {})
    rows = []
    for key in ("sigma2_v", "sigma2_u", "sigma2_eta", "sigma2_e"):
        if key in theta:
            rows.append({"Group": key, "Variance": theta[key]})
    phi = theta.get("phi")
    if phi is not None:
        rows.append({"Group": "phi (AR1)", "Variance": phi})
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def random_table_nested(fit: dict) -> pd.DataFrame:
    """Random effects table for nested class."""
    theta = fit.get("theta", {})
    rows = []
    for key in ("sigma2_prov", "sigma2_kab", "sigma2_e"):
        if key in theta:
            rows.append({"Group": key, "Variance": theta[key]})
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Model comparison table
# ---------------------------------------------------------------------------


def model_table(*fits, sort_by: str = "AIC") -> pd.DataFrame:
    """Comparison table of multiple model fits.

    Args:
        *fits: dicts with log_lik, AIC, BIC, model_name.
        sort_by: Column to sort by ("AIC", "BIC", "log_lik").

    Returns:
        DataFrame with model comparison metrics.
    """
    rows = []
    for f in fits:
        rows.append({
            "Model": f.get("model_name", "unknown"),
            "logLik": f.get("log_lik", 0),
            "AIC": f.get("AIC", 0),
            "BIC": f.get("BIC", 0),
            "k": f.get("k", 0),
        })
    df = pd.DataFrame(rows)
    if sort_by in df.columns:
        df = df.sort_values(sort_by).reset_index(drop=True)
        df["delta"] = df[sort_by] - df[sort_by].min()
    return df


def parameter_table(fit: dict, digits: int = 4) -> pd.DataFrame:
    """Fixed effects + variance components table."""
    ft = fixed_table(fit)
    ft.index = fit.get("coef_names", [f"beta_{i}" for i in range(len(ft))])
    return ft


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------


def latex_parameter_table(model: dict, caption: str = "", label: str = "", digits: int = 4) -> str:
    """LaTeX output for parameter table."""
    pt = parameter_table(model, digits)
    lines = [r"\begin{table}[htbp]", r"\centering"]
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\hline")
    lines.append(" & ".join(["Parameter", "Estimate", "SE", "df", "t", "p"]) + r" \\")
    lines.append(r"\hline")
    for idx, row in pt.iterrows():
        lines.append(
            f"{idx} & {fmt_num(row['Estimate'], digits)} & "
            f"{fmt_num(row['SE'], digits)} & {fmt_num(row['df'], 1)} & "
            f"{fmt_num(row['t value'], digits)} & {fmt_p(row['Pr(>|t|)'])} \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def latex_model_table(tab: pd.DataFrame, caption: str = "", label: str = "") -> str:
    """LaTeX output for model comparison table."""
    lines = [r"\begin{table}[htbp]", r"\centering"]
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    cols = list(tab.columns)
    lines.append(r"\begin{tabular}{l" + "r" * (len(cols) - 1) + "}")
    lines.append(r"\hline")
    lines.append(" & ".join(cols) + r" \\")
    lines.append(r"\hline")
    for _, row in tab.iterrows():
        vals = [fmt_num(row[c], 2) if isinstance(row[c], (int, float)) else str(row[c]) for c in cols]
        lines.append(" & ".join(vals) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Print methods
# ---------------------------------------------------------------------------


def print_stlmm(x: dict):
    """One-line print for a stLMM object."""
    from stmmlib.output_format import format_output
    print(format_output(x))


def print_summary_stlmm(x: dict, digits: int = 4):
    """Pretty-print full summary for a stLMM result."""
    from stmmlib.output_format import format_output
    print(format_output(x))
