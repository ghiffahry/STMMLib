"""The report must not present a boundary fit as inference-grade."""
import numpy as np
import pytest

from conftest import fit_beta
from stmmlib.output_format_beta import format_beta


def _break_certificate(fit):
    cert = fit._opt.setdefault("certificate", {})
    cert["valid"] = False
    cert["optimum_valid"] = True
    cert["at_bound"] = True
    cert["bound_params"] = ["log_phi"]
    cert["grad_norm"] = 28.5
    cert["projected_grad"] = 0.026
    return fit


def test_report_flags_an_invalid_certificate(df_flat):
    fit = _break_certificate(fit_beta(df_flat, "m1_iid_iid"))
    txt = format_beta(fit)
    assert "SERTIFIKAT" in txt
    assert "log_phi" in txt
    assert "valid" in txt.lower()


def test_report_names_the_gradient_problem(df_flat):
    fit = _break_certificate(fit_beta(df_flat, "m1_iid_iid"))
    txt = format_beta(fit)
    assert "grad" in txt.lower()


def test_strict_report_hides_the_p_values(df_flat):
    fit = fit_beta(df_flat, "m1_iid_iid")
    p_wald = float(fit.inference["pvalue"][1])
    assert f"{p_wald:.4f}" in format_beta(fit)
    _break_certificate(fit)
    txt = format_beta(fit, strict=True)
    assert f"{p_wald:.4f}" not in txt
    assert "SERTIFIKAT" in txt


def test_valid_fit_has_no_certificate_warning(df_flat):
    fit = fit_beta(df_flat, "m1_iid_iid")
    fit._opt.setdefault("certificate", {})["valid"] = True
    txt = format_beta(fit)
    assert "SERTIFIKAT" not in txt


def test_report_states_the_standardised_scale(df_flat):
    txt = format_beta(fit_beta(df_flat, "m1_iid_iid"))
    assert "baku" in txt.lower() or "standar" in txt.lower()


def test_report_states_the_actual_number_of_periods(df_flat):
    """The period count in the footer must come from the data.

    It was a literal "11 level" from the thesis panel, so every other
    dataset got a wrong number printed in its own summary.
    """
    txt = format_beta(fit_beta(df_flat, "m1_iid_iid"))
    assert "11 level" not in txt
    assert "efek-waktu 4 level" in txt


def test_ljung_box_says_why_it_is_missing(df_flat):
    """A short panel gets no Ljung-Box; the report must say why."""
    from stmmlib.beta.report import stmm_beta_diagnostics
    fit = fit_beta(df_flat, "m1_iid_iid")
    _diag, txt = stmm_beta_diagnostics(fit, as_str=True)
    assert "Ljung-Box" in txt
    assert "11 periode" in txt
