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


def prior_reference(y_unit, n=20000, seed=0, alpha=ALPHA):
    """CRPS and IS of the prior itself, per case and parameter, in unit coordinates.

    The prior is U[0,1] on every dimension here, so one ensemble is reused for every case
    and parameter -- what varies is only the truth it is scored against. Gives the
    denominator for a skill score (1 - score/score_prior: 1 perfect, 0 no better than the
    prior, negative worse).
    """
    u = np.random.default_rng(seed).random((n, 1))
    y = np.asarray(y_unit, dtype=np.float64).ravel()
    term1 = np.abs(u - y[None, :]).mean(0)
    xs = np.sort(u, axis=0)
    i = np.arange(1, n + 1, dtype=np.float64)[:, None]
    term2 = float(((2.0 * i - n - 1.0) * xs).sum() / (n * (n - 1.0)))
    crps = term1 - term2
    lo, hi = np.percentile(u, 100 * alpha / 2), np.percentile(u, 100 * (1 - alpha / 2))
    width = hi - lo
    pen = (2.0 / alpha) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))
    return crps.reshape(np.shape(y_unit)), (width + pen).reshape(np.shape(y_unit))


def score_set(samples, true, in_box, alpha=ALPHA):
    """Score a whole test set.

    samples (N, m, d) and true (N, d) in unit coordinates; in_box (N, m) bool marks the
    samples a rejection sampler would keep (the joint-in-prior mask, as every method's
    inference path applies it). Every score is computed twice: `*_box` on the kept samples
    -- the posterior that is actually deployed -- and `*_raw` on the full ensemble, so a
    method cannot look good or bad purely through its rejection rate.

    Cases with fewer than 2 kept samples get NaN in the `_box` scores (counted in
    `n_degenerate`) rather than a silent fallback to the raw ensemble, which would mix two
    different estimands in one column.
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
            continue
        out["crps_box"][i] = crps_fair(s_ok, y)
        isc, w, cov = interval_score(s_ok, y, alpha)
        out["is_box"][i], out["width_box"][i], out["cov_box"][i] = isc, w, cov
    out["n_degenerate"] = np.array(n_degen)
    cp, ip = prior_reference(true, alpha=alpha)
    out["crps_prior"], out["is_prior"] = cp, ip
    return out
