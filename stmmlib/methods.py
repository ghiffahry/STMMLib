"""StMM class: unified model result wrapper with S3-like method dispatch.

Wraps both Gaussian (StMMResult) and Beta (BetastmmResult) results
with a common interface for print, summary, plot, predict, etc.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class StMM:
    """Unified result wrapper for Gaussian and Beta STMMs.

    Provides a common interface regardless of family.
    """

    def __init__(self, result: Any, family: str = "gaussian"):
        self._result = result
        self._family = family

    @property
    def family(self) -> str:
        return self._family

    @property
    def model(self) -> str:
        return getattr(self._result, "model", "")

    @property
    def coefficients(self) -> dict:
        return self._result.coefficients

    @property
    def logLik(self) -> float:
        return self._result.logLik

    @property
    def AIC(self) -> float:
        return self._result.AIC

    @property
    def BIC(self) -> float:
        return self._result.BIC

    @property
    def convergence(self) -> dict:
        return self._result.convergence

    @property
    def phi(self) -> float:
        return getattr(self._result, "phi", np.nan)

    @property
    def sigma_area(self) -> float:
        return getattr(self._result, "sigma_area", np.nan)

    @property
    def sigma_time(self) -> float:
        return getattr(self._result, "sigma_time", np.nan)

    def summary(self) -> dict:
        """Return inference summary."""
        return self._result.summary

    def predict(self, newdata=None) -> np.ndarray:
        """Compute predictions (eta = X @ beta)."""
        return self._result.get_selected(newdata)

    def print(self) -> None:
        """Print model summary."""
        print(self)

    def __str__(self) -> str:
        from stmmlib.output_format import format_output
        return format_output(self._result)

    def __repr__(self) -> str:
        try:
            return self.__str__()
        except Exception:
            return (
                f"StMM(family={self._family!r}, model={self.model!r}, "
                f"logLik={self.logLik:.2f}, AIC={self.AIC:.2f}, BIC={self.BIC:.2f})"
            )
