"""Analytical curvature, score, and gradient pieces of the Beta NLL."""

from __future__ import annotations

import numpy as np
from scipy.special import digamma, polygamma


def beta_nll_curvature(eta, y, phi):
    """Analytical curvature (working weight) for Beta NLL.

    w_i = -d^2 l_i / d eta_i^2

    With mu = logit^{-1}(eta), m1=mu, m2=1-mu:
        h_i = -phi^2 [psi_1(mu*phi) + psi_1((1-mu)*phi)] (m1*m2)^2
              + phi [psi_0((1-mu)*phi) - psi_0(mu*phi) + logit(y)] (m1*m2)(1-2*mu)

    w_i = max(-h_i, floor)

    Args:
        eta: Linear predictor (n,).
        y: Response (n,), in (0,1).
        phi: Precision parameter.

    Returns:
        w: Working weights (n,), floored for PD guarantee.
    """
    mu = 1.0 / (1.0 + np.exp(-eta))
    mu = np.clip(mu, 1e-15, 1.0 - 1e-15)
    m1 = mu
    m2 = 1.0 - mu

    a = mu * phi
    b = (1.0 - mu) * phi

    psi1_a = polygamma(1, a)
    psi1_b = polygamma(1, b)
    psi0_a = digamma(a)
    psi0_b = digamma(b)

    logit_y = np.log(np.clip(y, 1e-15, 1.0 - 1e-15) / np.clip(1.0 - y, 1e-15, 1.0))

    h = (
        -phi**2 * (psi1_a + psi1_b) * (m1 * m2) ** 2
        + phi * (psi0_b - psi0_a + logit_y) * (m1 * m2) * (1.0 - 2.0 * mu)
    )

    w = np.maximum(-h, 1e-6)
    return w


def _beta_nll_score(eta, y, phi):
    """Score (gradient) of Beta NLL per observation.

    s_i = -phi * mu * (1-mu) [psi_0((1-mu)*phi) - psi_0(mu*phi) + log(y) - log(1-y)]
    """
    mu = 1.0 / (1.0 + np.exp(-eta))
    mu = np.clip(mu, 1e-15, 1.0 - 1e-15)

    a = mu * phi
    b = (1.0 - mu) * phi

    psi0_a = digamma(a)
    psi0_b = digamma(b)

    log_y = np.log(np.clip(y, 1e-15, 1.0))
    log_1my = np.log(np.clip(1.0 - y, 1e-15, 1.0))

    s = -phi * mu * (1.0 - mu) * (psi0_b - psi0_a + log_y - log_1my)
    return s


def _beta_grad_pieces(eta, y, phi):
    """Analytic derivative pieces of the Beta log-lik for the Laplace gradient.

    Returns dict with (all elementwise over obs):
      l_phi : d l_i / d phi                        (for the -nll term)
      w_dot : d w_i / d eta   (= -d^2 s / d eta^2) (beta log-det term)
      w_dphi: d w_i / d phi at fixed eta           (log(phi) log-det term)
    Both w-derivatives are zeroed where the curvature floor (1e-6) binds,
    matching beta_nll_curvature's max(-h, floor).

    Derivation (envelope theorem: at the inner mode b_hat the total
    derivative of the Laplace marginal equals the partial derivative with
    b frozen -- TMB paper, Kristensen et al. 2016, JSS 70):
      l_i   = -betaln(a,b) + (a-1)ln y + (b-1)ln(1-y),  a=mu*phi, b=(1-mu)*phi
      h     = d^2 l/d eta^2 = phi*t*(1-2mu)*g - phi^2*t^2*A
      w     = -h,  t = mu(1-mu),  g = logit(y) - psi(a) + psi(b)
      A     = psi1(a) + psi1(b),  D2 = psi2(a) - psi2(b)
      w_dot = -dh/d eta = phi*t*g*[(1-2mu)^2 - 2t] ... (see code)
    """
    mu = 1.0 / (1.0 + np.exp(-eta))
    mu = np.clip(mu, 1e-15, 1.0 - 1e-15)
    t = mu * (1.0 - mu)
    a = mu * phi
    b = (1.0 - mu) * phi

    psi0_a, psi0_b = digamma(a), digamma(b)
    psi1_a, psi1_b = polygamma(1, a), polygamma(1, b)
    psi2_a, psi2_b = polygamma(2, a), polygamma(2, b)

    log_y = np.log(np.clip(y, 1e-15, 1.0))
    log_1my = np.log(np.clip(1.0 - y, 1e-15, 1.0))
    logit_y = log_y - log_1my

    g = logit_y - psi0_a + psi0_b           # d l / d mu = phi * t * g
    A = psi1_a + psi1_b
    D2 = psi2_a - psi2_b
    om = 1.0 - 2.0 * mu                      # (1 - 2 mu)

    # l_phi = d l / d phi  (= -[psi(a) mu + psi(b)(1-mu) - psi(a+b)] + mu ln y
    #        + (1-mu) ln(1-y);  a+b = phi so lnGamma(a+b) contributes +psi(phi))
    l_phi = (-(psi0_a * mu + psi0_b * (1.0 - mu)) + digamma(phi)
             + mu * log_y + (1.0 - mu) * log_1my)

    # h = phi*om*t*g - phi^2*t^2*A ; w = -h
    h = phi * om * t * g - phi**2 * t**2 * A
    floor_mask = h >= -1e-6                  # w floored at 1e-6 here

    # dh/d eta = phi*t*g*(om^2 - 2t) - 3 phi^2 om t^2 A - phi^3 t^3 D2
    dh_deta = (phi * t * g * (om**2 - 2.0 * t)
               - 3.0 * phi**2 * om * t**2 * A
               - phi**3 * t**3 * D2)
    w_dot = -dh_deta

    # dh/d phi at fixed eta:
    #   d g/d phi = -mu*psi1(a) + (1-mu)*psi1(b)   (mixed signs!)
    #   d A/d phi =  mu*psi2(a) + (1-mu)*psi2(b)   = F
    Gphi = -mu * psi1_a + (1.0 - mu) * psi1_b
    F = mu * psi2_a + (1.0 - mu) * psi2_b
    dh_dphi = (om * t * g + phi * om * t * Gphi
               - 2.0 * phi * t**2 * A - phi**2 * t**2 * F)
    w_dphi = -dh_dphi

    w_dot = np.where(floor_mask, 0.0, w_dot)
    w_dphi = np.where(floor_mask, 0.0, w_dphi)
    return {"l_phi": l_phi, "w_dot": w_dot, "w_dphi": w_dphi,
            "w_raw": -h, "s_dphi": t * g + phi * t * Gphi}
