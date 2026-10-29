"""Model registry: 14 structural models (M1-M14) + beta_regression.

Each model maps to area-level, time-level, time_scope, interaction type,
and parameter count for Normal and Beta families.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Model specification dataclass
# ---------------------------------------------------------------------------

InteractionType = Literal["none", "iid", "spatial", "temporal", "st"]
TimeScope = Literal["area_specific", "global"]
AreaType = Literal["iid", "sar", "nested_iid", "nested_sar", "seblup_rao_yu"]
TimeType = Literal["iid", "ar1"]


@dataclass(frozen=True)
class ModelSpec:
    """Immutable specification for one model structure."""

    model_id: str
    area: AreaType
    time: TimeType
    time_scope: TimeScope
    interaction: InteractionType
    n_params_normal: int  # variance params (excl. fixed effects)
    n_params_beta: int  # variance/precision params for Beta
    label: str


# ---------------------------------------------------------------------------
# Registry of 14+1 models
# ---------------------------------------------------------------------------

MODELS: dict[str, ModelSpec] = {
    "m1_iid_iid": ModelSpec(
        model_id="m1_iid_iid",
        area="iid",
        time="iid",
        time_scope="area_specific",
        interaction="none",
        n_params_normal=3,
        n_params_beta=3,
        label="IID + IID",
    ),
    "m2_iid_ar1": ModelSpec(
        model_id="m2_iid_ar1",
        area="iid",
        time="ar1",
        time_scope="area_specific",
        interaction="none",
        n_params_normal=4,
        n_params_beta=4,
        label="IID + AR(1)",
    ),
    "m3_sar_iid": ModelSpec(
        model_id="m3_sar_iid",
        area="sar",
        time="iid",
        time_scope="area_specific",
        interaction="none",
        n_params_normal=4,
        n_params_beta=4,
        label="SAR + IID",
    ),
    "m4_sar_ar1": ModelSpec(
        model_id="m4_sar_ar1",
        area="sar",
        time="ar1",
        time_scope="global",
        interaction="none",
        n_params_normal=5,
        n_params_beta=5,
        label="SAR + AR(1) [global]",
    ),
    "m5_seblup_rao_yu_it": ModelSpec(
        model_id="m5_seblup_rao_yu_it",
        area="seblup_rao_yu",
        time="ar1",
        time_scope="area_specific",
        interaction="none",
        n_params_normal=5,
        n_params_beta=5,
        label="SEBLUP Rao-Yu IT",
    ),
    "m6_nested_iid_iid": ModelSpec(
        model_id="m6_nested_iid_iid",
        area="nested_iid",
        time="iid",
        time_scope="area_specific",
        interaction="none",
        n_params_normal=4,
        n_params_beta=4,
        label="Nested IID + IID",
    ),
    "m7_nested_ar1": ModelSpec(
        model_id="m7_nested_ar1",
        area="nested_sar",
        time="ar1",
        time_scope="area_specific",
        interaction="none",
        n_params_normal=5,
        n_params_beta=5,
        label="Nested IID + AR(1)",
    ),
    "m8_iid_iid_int": ModelSpec(
        model_id="m8_iid_iid_int",
        area="iid",
        time="iid",
        time_scope="area_specific",
        interaction="iid",
        n_params_normal=4,
        n_params_beta=4,
        label="IID + IID + Type I",
    ),
    "m9_sar_iid_int": ModelSpec(
        model_id="m9_sar_iid_int",
        area="sar",
        time="iid",
        time_scope="area_specific",
        interaction="spatial",
        n_params_normal=6,
        n_params_beta=6,
        label="SAR + IID + Type II",
    ),
    "m10_iid_ar1_int": ModelSpec(
        model_id="m10_iid_ar1_int",
        area="iid",
        time="ar1",
        time_scope="area_specific",
        interaction="temporal",
        n_params_normal=6,
        n_params_beta=6,
        label="IID + AR(1) + Type III",
    ),
    "m11_sar_ar1_int": ModelSpec(
        model_id="m11_sar_ar1_int",
        area="sar",
        time="ar1",
        time_scope="global",
        interaction="st",
        n_params_normal=8,
        n_params_beta=8,
        label="SAR + AR(1) + Type IV [global]",
    ),
    "m12_seblup_rao_yu_it_int": ModelSpec(
        model_id="m12_seblup_rao_yu_it_int",
        area="seblup_rao_yu",
        time="ar1",
        time_scope="area_specific",
        interaction="st",
        n_params_normal=8,
        n_params_beta=8,
        label="SEBLUP Rao-Yu IT + Type IV",
    ),
    "m13_nested_iid_iid_int": ModelSpec(
        model_id="m13_nested_iid_iid_int",
        area="nested_iid",
        time="iid",
        time_scope="area_specific",
        interaction="temporal",
        n_params_normal=6,
        n_params_beta=6,
        label="Nested IID + IID + Type III",
    ),
    "m14_nested_ar1_int": ModelSpec(
        model_id="m14_nested_ar1_int",
        area="nested_sar",
        time="ar1",
        time_scope="area_specific",
        interaction="iid",
        n_params_normal=6,
        n_params_beta=6,
        label="Nested IID + AR(1) + Type I",
    ),
    "beta_regression": ModelSpec(
        model_id="beta_regression",
        area="iid",
        time="iid",
        time_scope="area_specific",
        interaction="none",
        n_params_normal=0,
        n_params_beta=2,
        label="Beta Regression (fixed only)",
    ),
}

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def stmm_available_models(family: str = "normal") -> list[str]:
    """Return list of available model IDs for a given family.

    Args:
        family: "normal" or "beta".

    Returns:
        List of model ID strings.
    """
    if family == "normal":
        return [mid for mid, s in MODELS.items() if s.n_params_normal > 0]
    if family == "beta":
        return list(MODELS.keys())
    raise ValueError(f"Unknown family: {family!r}. Use 'normal' or 'beta'.")


def stmm_model_label(model_id: str) -> str:
    """Human-readable label for a model ID."""
    spec = MODELS.get(model_id)
    if spec is None:
        raise KeyError(f"Unknown model: {model_id!r}")
    return spec.label


def stmm_parse_model(model_id: str) -> dict:
    """Parse model ID into its structural components.

    Returns:
        dict with keys: area, time, time_scope, interaction, n_params_normal,
        n_params_beta, label.
    """
    spec = MODELS.get(model_id)
    if spec is None:
        raise KeyError(f"Unknown model: {model_id!r}")
    return {
        "area": spec.area,
        "time": spec.time,
        "time_scope": spec.time_scope,
        "interaction": spec.interaction,
        "n_params_normal": spec.n_params_normal,
        "n_params_beta": spec.n_params_beta,
        "label": spec.label,
    }


def get_model_spec(model_id: str) -> ModelSpec:
    """Get the full ModelSpec for a model ID."""
    spec = MODELS.get(model_id)
    if spec is None:
        raise KeyError(f"Unknown model: {model_id!r}")
    return spec


# ---------------------------------------------------------------------------
# Model equations (M1-M14, both families)
# ---------------------------------------------------------------------------

def stmm_model_equation(model_id: str, family: str = "beta") -> str:
    """Equation string for one model (thesis-ready, link-scale form).

    Beta family: logit(mu_it) = x_it' beta + <random effects>,
        y_it ~ Beta(mu_it * phi, (1 - mu_it) * phi).
    Gaussian family: y_it = x_it' beta + <random effects> + e_it,
        e_it ~ N(0, sigma_e2).

    Random-effect blocks (Knuttson-Held Type I-IV for interactions):
        area IID:        u_i ~ N(0, sigma_v2)
        area SAR:        u ~ N(0, sigma_v2 (I - rho W)^-1 (I - rho W)^-T)
        prov/kab nested: v_p ~ N(0, sigma_p2), v_pk ~ N(0, sigma_k2)
        time IID:        eta_t (global) or eta_it (area-specific) ~ N(0, sigma2)
        time AR(1):      AR(1) with coef phi_T, innovation sigma_eta2
        Type I (iid):    delta_it ~ N(0, sigma_g2)
        Type II (sp):    delta_t ~ N(0, sigma_g2 (I - rho_g W)^-1 (...)^-T)
        Type III (tmp):  delta_i ~ AR(1) with coef phi_g
        Type IV (st):    Type II x Type III (Kronecker)

    Args:
        model_id: e.g. "m5_seblup_rao_yu_it".
        family: "beta" or "gaussian"/"normal".

    Returns:
        Multiline equation string.
    """
    spec = MODELS.get(model_id)
    if spec is None:
        raise KeyError(f"Unknown model: {model_id!r}")
    fam = family.lower()
    if fam not in ("beta", "gaussian", "normal"):
        raise ValueError(f"Unknown family: {family!r}. Use 'beta' or 'gaussian'.")

    if fam == "beta":
        head = ("y_it ~ Beta(mu_it * kappa, (1 - mu_it) * kappa)\n"
                "logit(mu_it) = x_it' beta + RE_it")
    else:
        head = ("y_it = x_it' beta + RE_it + e_it,  e_it ~ N(0, sigma_e2)")

    area_txt = {
        "iid": "u_i ~ N(0, sigma_v2)",
        "sar": "u ~ N(0, sigma_v2 (I - rho_S W)^-1 (I - rho_S W)^-T)",
        "nested_iid": "v_p ~ N(0, sigma_p2), v_pk ~ N(0, sigma_k2)",
        "nested_sar": "v_p ~ N(0, sigma_p2), v_pk ~ N(0, sigma_k2)",
        "seblup_rao_yu": ("u ~ N(0, sigma_v2 (I - rho_S W)^-1 (I - rho_S W)^-T) "
                           "[area, Rao-Yu]"),
    }[spec.area]
    if spec.time_scope == "global":
        time_txt = {"iid": "eta_t ~ N(0, sigma2)",
                    "ar1": "eta_t ~ AR(1; phi_T, sigma_eta2)"}[spec.time]
    else:
        time_txt = {"iid": "eta_it (area-specific) ~ N(0, sigma2)",
                    "ar1": "eta_it (area-specific) ~ AR(1; phi_T, sigma_eta2)"}[spec.time]
    int_txt = {
        "none": None,
        "iid": "delta_it ~ N(0, sigma_g2)  [Type I]",
        "spatial": ("delta_t ~ N(0, sigma_g2 (I - rho_g W)^-1 (...)^-T)"
                    "  [Type II]"),
        "temporal": "delta_i ~ AR(1; phi_g, sigma_g2)  [Type III]",
        "st": ("delta ~ Type II (x) Type III Kronecker  "
               "[Type IV: rho_g, phi_g]"),
    }[spec.interaction]

    lines = [f"{model_id} ({spec.label}) [{fam}]", head,
             f"RE: {area_txt} + {time_txt}"]
    if int_txt is not None:
        lines.append(f"  + {int_txt}")
    return "\n".join(lines)
