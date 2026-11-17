"""Proper scoring rules for sample-based posteriors: CRPS and the interval score.

Both are computed per case and per parameter in **unit-cube coordinates** (the prior box
mapped to [0,1], log on the log-uniform dims), not physical units. That makes the 24
parameters commensurate, so they can be pooled or shown on one axis, and it fixes the
reference distribution: the prior is exactly uniform on [0,1] in this space, so a prior
forecast scores the same on every dimension.

CRPS uses the fair (unbiased) estimator

    CRPS = mean_i |x_i - y| - 1/(m(m-1)) * sum_i (2i - m - 1) * x_(i)

where x_(i) is the ascending-sorted ensemble. The second term is the ensemble's own mean
absolute pairwise difference and is what stops a point forecast from winning by being
degenerate; the 1/(m(m-1)) normalisation (rather than 1/m^2) removes the O(1/m) bias that
would otherwise reward small ensembles. Lower is better; a point mass scores |x - y|.

The interval score at central level 1-alpha is Gneiting & Raftery's

    IS = (u - l) + (2/alpha)(l - y) 1{y < l} + (2/alpha)(y - u) 1{y > u}

reported alongside its two pieces -- the width u-l (sharpness) and the penalty (calibration
failure) -- because the sum alone cannot distinguish a wide honest posterior from a narrow
overconfident one, which is the whole question for a misspecified likelihood.
"""
import numpy as np

ALPHA = 0.05            # central 95% interval


def crps_fair(samples, y):
    """samples (m, d), y (d,) -> CRPS (d,). Fair/unbiased ensemble estimator."""
    m = samples.shape[0]
    if m < 2:
        return np.abs(samples[0] - y) if m == 1 else np.full(y.shape, np.nan)
    term1 = np.abs(samples - y[None, :]).mean(0)
    xs = np.sort(samples, axis=0)
    i = np.arange(1, m + 1, dtype=np.float64)[:, None]
    term2 = ((2.0 * i - m - 1.0) * xs).sum(0) / (m * (m - 1.0))
    return term1 - term2


def interval_score(samples, y, alpha=ALPHA):
    """samples (m, d), y (d,) -> (is, width, covered) each (d,)."""
    if samples.shape[0] < 2:
        nan = np.full(y.shape, np.nan)
        return nan, nan, nan
    lo = np.percentile(samples, 100 * alpha / 2, axis=0)
    hi = np.percentile(samples, 100 * (1 - alpha / 2), axis=0)
    width = hi - lo
    pen = (2.0 / alpha) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))
    return width + pen, width, ((y >= lo) & (y <= hi)).astype(np.float64)


def prior_reference(y_unit, alpha=ALPHA, **_):
    """CRPS and IS of the prior itself, per case and parameter, in unit coordinates.

    The prior is exactly U[0,1] on every dimension in this space, so both scores are
    closed-form rather than estimated: CRPS(U[0,1], y) = y^2 - y + 1/3 for y in [0,1],
    and the central 1-alpha interval is [alpha/2, 1-alpha/2]. Gives the denominator for a
    skill score (1 - score/score_prior: 1 perfect, 0 no better than the prior, negative
    worse). Verified against the Monte-Carlo estimator it replaces.
    """
    y = np.asarray(y_unit, dtype=np.float64)
    crps = y ** 2 - y + 1.0 / 3.0
    lo, hi = alpha / 2, 1 - alpha / 2
    pen = (2.0 / alpha) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))
    return crps, (hi - lo) + pen


def score_set(samples, true, in_box, alpha=ALPHA):
    """Score a whole test set.

    samples (N, m, d) and true (N, d) in unit coordinates; in_box (N, m) bool marks the
    samples a rejection sampler would keep (the joint-in-prior mask, as every method's
    inference path applies it). Every score is computed twice: `*_box` on the deployed
    posterior and `*_raw` on the full ensemble, so a method cannot look good or bad purely
    through its rejection rate.

    When fewer than 2 draws survive the box, the `*_box` score falls back to the full
    ensemble -- which is what the inference paths themselves do (`eval_common`'s
    "or of the raw samples if none land in-prior"). Returning NaN instead would quietly
    drop those cases from the mean, and since the fraction reaches ~18% for some
    method/level pairs that would be a survivor bias in favour of exactly the methods
    whose rejection rate is worst. `n_degenerate` reports how often the fallback fired.
    """
    N, m, d = samples.shape
    out = {k: np.full((N, d), np.nan) for k in
           ("crps_box", "crps_raw", "is_box", "is_raw", "width_box", "cov_box")}
    n_degen = 0
    for i in range(N):
        s_all, y = samples[i].astype(np.float64), true[i].astype(np.float64)
        out["crps_raw"][i] = crps_fair(s_all, y)
        out["is_raw"][i] = interval_score(s_all, y, alpha)[0]
        s_ok = s_all[in_box[i]]
        if s_ok.shape[0] < 2:
            n_degen += 1
            s_ok = s_all
        out["crps_box"][i] = crps_fair(s_ok, y)
        isc, w, cov = interval_score(s_ok, y, alpha)
        out["is_box"][i], out["width_box"][i], out["cov_box"][i] = isc, w, cov
    out["n_degenerate"] = np.array(n_degen)
    cp, ip = prior_reference(true, alpha=alpha)
    out["crps_prior"], out["is_prior"] = cp, ip
    return out
