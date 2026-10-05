"""Fit low-fi and high-fi to representative real beats: low-fi -> high-fi -> real.

Stages (outputs in OUT_DIR):
  select          pick N representative real beats (one per patient-cluster
                  medoid; the patient's beat closest to its mean beat)
  fit --beat i    for beat i: low-fi theta fit (bank + CMA-ES), high-fi phi-only
                  fit (theta fixed), joint theta+phi fit, per-element phi-only
                  ablation fits; writes beat_<i>.json
  plot            figure + summary table from the beat_<i>.json files

Loss: mean over Prv, Pra, Pap of squared NRMSE (rms error / real peak-to-peak),
each channel allowed a circular shift of up to MAX_SHIFT grid points. Pvp
(wedge) is excluded from the fit; respiration and the wedge observation are off.
HR is fixed to the real beat's.

High-fi is parameterised per element by a gain g in [0, 1] (g = 0 is that
element off) times prior-space draws, so g = 0 everywhere is exactly low-fi.
The low-fi point is always evaluated, so high-fi fits are never worse.

Usage:
  python scripts/fit_real_beats.py select OUT_DIR [--n-beats 6]
  python scripts/fit_real_beats.py fit OUT_DIR --beat i [--workers 40]
  python scripts/fit_real_beats.py plot OUT_DIR FIG.png
"""

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cma
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nested_sim import simulate  # noqa: E402
from nested_sim.params import THETA_BOUNDS, full_theta  # noqa: E402
from nested_sim.phi import PHI_GROUPS, PHI_OFF, _physical, load_priors  # noqa: E402
from real_vs_sim import load_real  # noqa: E402

FIT_CH = ["Prv", "Pra", "Pap"]
ALL_CH = ["Prv", "Pra", "Pap", "Pvp"]
MAX_SHIFT = 8
FAIL_LOSS = 10.0
FIT_GROUPS = ["wk_s", "wk_p", "lv_av", "lv_pv", "res", "av_l", "av_r"]
THETA_FIT = [k for k in THETA_BOUNDS if k != "HR"]
LOG_THETA = {"Emax_RV", "Eap", "Rap", "Eedref_lv", "Eedref_rv", "Eedref_la", "Eedref_ra"}
PRIORS = load_priors()


# ---------------------------------------------------------------- real beats
def real_beats():
    df, W = load_real()
    return df, {c: W[:, j] for j, c in enumerate(ALL_CH)}


def select(out, n_beats):
    df, W = real_beats()
    rows = []
    for pid, idx in df.groupby("id").groups.items():
        idx = list(idx)
        mean = {c: W[c][idx].mean(axis=0) for c in FIT_CH}
        d = [sum(np.mean((W[c][i] - mean[c]) ** 2) / np.ptp(mean[c]) ** 2 for c in FIT_CH) for i in idx]
        b = idx[int(np.argmin(d))]
        rows.append({"id": pid, "beat": int(b), "cohort": df.cohort[b], "hr": 60000.0 / df["T"][b],
                     **{k: float(df[k][b]) for k in ("mpap", "spa", "dpa", "rv_s", "rv_d", "cvp", "pcw")}})
    pat = pd.DataFrame(rows)
    feats = ["mpap", "spa", "dpa", "rv_s", "rv_d", "cvp", "hr"]
    Z = ((pat[feats] - pat[feats].mean()) / pat[feats].std()).to_numpy()
    cent, lab = kmeans2(Z, n_beats, seed=0, minit="++")
    chosen = []
    for k in range(n_beats):
        members = np.flatnonzero(lab == k)
        i = members[np.argmin(((Z[members] - cent[k]) ** 2).sum(1))]
        chosen.append({**pat.iloc[i].to_dict(), "cluster": k, "cluster_size": int(members.size)})
    chosen.sort(key=lambda r: r["mpap"])
    json.dump(chosen, open(out / "selected.json", "w"), indent=1, default=float)
    print(pd.DataFrame(chosen)[["id", "beat", "cohort", "cluster_size", "hr", "mpap", "spa", "dpa",
                                "rv_s", "cvp", "pcw"]].round(1).to_string(index=False))


# ---------------------------------------------------------------- loss
def shifted_nrmse(sim, real):
    """min over circular shifts of rms(sim - real) / ptp(real); sim is periodic."""
    period = sim[:-1]
    best = np.inf
    for s in range(-MAX_SHIFT, MAX_SHIFT + 1):
        x = np.roll(period, s)
        x = np.append(x, x[0])
        best = min(best, float(np.sqrt(np.mean((x - real) ** 2))))
    return best / max(np.ptp(real), 1e-6)


def errors(r, real):
    return {c: shifted_nrmse(r.waves201[c], real[c]) for c in ALL_CH}


def loss_of(err):
    return float(np.mean([err[c] ** 2 for c in FIT_CH]))


# ---------------------------------------------------------------- parameter maps
def theta_from_u(u, hr):
    th = {"HR": hr}
    for x, k in zip(u, THETA_FIT):
        lo, hi = THETA_BOUNDS[k]
        x = min(max(x, 0.0), 1.0)
        th[k] = math.exp(math.log(lo) + x * (math.log(hi) - math.log(lo))) if k in LOG_THETA else lo + x * (hi - lo)
    return th


def u_from_theta(th):
    u = []
    for k in THETA_FIT:
        lo, hi = THETA_BOUNDS[k]
        v = th[k]
        u.append((math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)) if k in LOG_THETA else (v - lo) / (hi - lo))
    return np.clip(u, 0, 1)


def phi_layout(groups):
    """List of (kind, name) for the phi vector: ('gain', group) or ('draw', phi name)."""
    lay = []
    for g in groups:
        lay.append(("gain", g))
        lay += [("draw", k) for k in PHI_GROUPS[g] if k in PRIORS]
    return lay


def phi_from_u(u, theta, groups):
    lay = phi_layout(groups)
    gains, draws = {}, {}
    for x, (kind, name) in zip(u, lay):
        x = min(max(x, 0.0), 1.0)
        if kind == "gain":
            gains[name] = x
        else:
            spec = PRIORS[name]
            lo, hi = spec["low"], spec["high"]
            draws[name] = (math.exp(math.log(lo) + x * (math.log(hi) - math.log(lo)))
                           if spec["dist"] == "loguniform" else lo + x * (hi - lo))
    th = full_theta(theta)
    full = dict(PHI_OFF)
    order = [k for g in ("res", "wk_s", "wk_p", "lv_av", "lv_pv", "av_l", "av_r") for k in PHI_GROUPS[g]]
    for k in order:
        if k in draws:
            full[k] = _physical(k, draws[k], PRIORS[k], th, full)
    phi = {}
    for g in groups:
        for k in PHI_GROUPS[g]:
            if k in draws:
                phi[k] = PHI_OFF[k] + gains[g] * (full[k] - PHI_OFF[k])
    return phi, gains


# ---------------------------------------------------------------- evaluation
def evaluate(args):
    theta, phi, real = args
    try:
        r = simulate(theta, phi)
    except Exception:  # noqa: BLE001 - any solver blow-up counts as a failed sim
        return FAIL_LOSS, None
    if not r.success:
        return FAIL_LOSS, None
    err = errors(r, real)
    return loss_of(err), err


def cma_fit(f_args, x0, sigma, pool, maxiter, popsize, seed, extra_x=()):
    """Bounded CMA-ES in [0,1]^d; f_args(x) -> (theta, phi). Returns best (loss, x, err)."""
    es = cma.CMAEvolutionStrategy(x0, sigma, {"bounds": [0, 1], "popsize": popsize, "seed": seed,
                                              "maxiter": maxiter, "verbose": -9, "tolfun": 1e-6})
    best = (np.inf, None, None)
    for x in extra_x:  # always evaluate given points (e.g. the low-fi point)
        lval, err = pool_eval(pool, [f_args(x)])[0]
        if lval < best[0]:
            best = (lval, np.asarray(x), err)
    while not es.stop():
        X = es.ask()
        res = pool_eval(pool, [f_args(x) for x in X])
        es.tell(X, [r[0] for r in res])
        for x, (lval, err) in zip(X, res):
            if lval < best[0]:
                best = (lval, np.asarray(x), err)
    return best


REAL = None  # set per process


def pool_eval(pool, args):
    return list(pool.map(evaluate, [(th, ph, REAL) for th, ph in args]))


def fit_beat(out, i, workers):
    """Run all fit stages for beat i; each stage is checkpointed and skipped on rerun."""
    global REAL
    sel = json.load(open(out / "selected.json"))[i]
    df, W = real_beats()
    b = sel["beat"]
    REAL = {c: W[c][b] for c in ALL_CH}
    hr = 60000.0 / df["T"][b]
    ckpt = out / f"beat_{i}.partial.json"
    res = json.load(open(ckpt)) if ckpt.exists() else {"selected": sel, "hr": hr}

    def save():
        json.dump(res, open(ckpt, "w"), indent=1, default=float)

    t0 = time.time()
    n_th = len(THETA_FIT)
    with ProcessPoolExecutor(workers) as pool:
        if "lofi" not in res:
            # 1. low-fi bank, 2. CMA-ES from the 3 best bank points
            rng = np.random.default_rng(100 + i)
            U = rng.uniform(size=(3000, n_th))
            bank = pool_eval(pool, [(theta_from_u(u, hr), None) for u in U])
            order = np.argsort([x[0] for x in bank])
            res["bank_best_loss"] = float(bank[order[0]][0])
            lo_best = (np.inf, None, None)
            for k in range(3):
                cand = cma_fit(lambda x: (theta_from_u(x, hr), None), U[order[k]], 0.08, pool,
                               maxiter=80, popsize=workers, seed=k + 1)
                if cand[0] < lo_best[0]:
                    lo_best = cand
            res["lofi"] = {"loss": lo_best[0], "err": lo_best[2], "x": list(lo_best[1]),
                           "theta": theta_from_u(lo_best[1], hr)}
            save()
        print(f"[beat {i}] low-fi loss {res['lofi']['loss']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        x_lo = np.asarray(res["lofi"]["x"])
        th_lo = theta_from_u(x_lo, hr)

        lay = phi_layout(FIT_GROUPS)
        x_off = np.array([0.0 if kind == "gain" else 0.5 for kind, _ in lay])
        if "hifi_phi" not in res:
            # 3. high-fi phi-only (theta fixed), all elements
            x0 = np.array([0.3 if kind == "gain" else 0.5 for kind, _ in lay])
            f = lambda x: (th_lo, phi_from_u(x, th_lo, FIT_GROUPS)[0])  # noqa: E731
            hb = cma_fit(f, x0, 0.25, pool, maxiter=60, popsize=workers, seed=7, extra_x=[x_off])
            phi_hi, gains = phi_from_u(hb[1], th_lo, FIT_GROUPS)
            res["hifi_phi"] = {"loss": hb[0], "err": hb[2], "x": list(hb[1]), "phi": phi_hi, "gains": gains}
            save()
        print(f"[beat {i}] high-fi phi-only loss {res['hifi_phi']['loss']:.4f} ({time.time() - t0:.0f}s)", flush=True)

        if "hifi_joint" not in res:
            # 4. joint theta + phi from (theta_lo, phi_hi)
            def fj(x):
                th = theta_from_u(x[:n_th], hr)
                return th, phi_from_u(x[n_th:], th, FIT_GROUPS)[0]
            xj0 = np.concatenate([x_lo, np.asarray(res["hifi_phi"]["x"])])
            jb = cma_fit(fj, xj0, 0.05, pool, maxiter=60, popsize=workers, seed=11, extra_x=[xj0])
            th_j = theta_from_u(jb[1][:n_th], hr)
            phi_j, gains_j = phi_from_u(jb[1][n_th:], th_j, FIT_GROUPS)
            res["hifi_joint"] = {"loss": jb[0], "err": jb[2], "theta": th_j, "phi": phi_j, "gains": gains_j}
            save()
        print(f"[beat {i}] high-fi joint loss {res['hifi_joint']['loss']:.4f} ({time.time() - t0:.0f}s)", flush=True)

        # 5. ablation: each element alone, phi-only
        res.setdefault("ablation", {})
        for g in FIT_GROUPS:
            if g in res["ablation"]:
                continue
            lay_g = phi_layout([g])
            xo = np.array([0.0 if kind == "gain" else 0.5 for kind, _ in lay_g])
            xs = np.array([0.3 if kind == "gain" else 0.5 for kind, _ in lay_g])
            fg = lambda x, g=g: (th_lo, phi_from_u(x, th_lo, [g])[0])  # noqa: E731
            ab = cma_fit(fg, xs, 0.25, pool, maxiter=25, popsize=min(workers, 16), seed=3, extra_x=[xo])
            res["ablation"][g] = {"loss": ab[0], "err": ab[2], "phi": phi_from_u(ab[1], th_lo, [g])[0]}
            save()
        print(f"[beat {i}] ablation done ({time.time() - t0:.0f}s)", flush=True)

    json.dump(res, open(out / f"beat_{i}.json", "w"), indent=1, default=float)


# ---------------------------------------------------------------- plot
def plot(out, fig_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted((p for p in out.glob("beat_*.json") if ".partial" not in p.name),
                   key=lambda p: int(p.stem.split("_")[1]))
    fits = [json.load(open(p)) for p in files]
    df, W = real_beats()
    colors = {"real": "#1f1f1e", "lofi": "#2a78d6", "hifi_phi": "#1baf7a", "hifi_joint": "#eb6834"}
    labels = {"real": "real", "lofi": "low-fi fit", "hifi_phi": "high-fi, phi fit (theta fixed)",
              "hifi_joint": "high-fi, joint fit"}
    fig, axes = plt.subplots(len(fits), 4, figsize=(16, 3.0 * len(fits)), squeeze=False)
    rows = []
    for i, fr in enumerate(fits):
        sel = fr["selected"]
        b = sel["beat"]
        real = {c: W[c][b] for c in ALL_CH}
        t = np.linspace(0, df["T"][b], 201)
        sims = {"lofi": simulate(fr["lofi"]["theta"]),
                "hifi_phi": simulate(fr["lofi"]["theta"], fr["hifi_phi"]["phi"]),
                "hifi_joint": simulate(fr["hifi_joint"]["theta"], fr["hifi_joint"]["phi"])}
        for j, c in enumerate(ALL_CH):
            ax = axes[i, j]
            ax.plot(t, real[c], color=colors["real"], lw=2.4, label=labels["real"])
            for k, r in sims.items():
                if r.success:
                    ax.plot(t, best_shift(r.waves201[c], real[c]), color=colors[k], lw=1.6,
                            ls="-" if k != "hifi_phi" else "--", label=labels[k])
            e = " ".join(f"{k[0].upper() if k == 'lofi' else ('P' if k == 'hifi_phi' else 'J')}"
                         f"={fr[k]['err'][c]:.2f}" for k in ("lofi", "hifi_phi", "hifi_joint"))
            title = f"{c}{' (not fitted)' if c not in FIT_CH else ''}   NRMSE {e}"
            if j == 0:
                title = (f"beat {i}: patient {sel['id']} ({sel['cohort']}), HR {sel['hr']:.0f}, "
                         f"mPAP {sel['mpap']:.0f}\n" + title)
            ax.set_title(title, fontsize=8.5, loc="left", color="#1f1f1e")
            ax.grid(True, color="#e6e5df", lw=0.8)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(colors="#6b6a63", labelsize=8)
        rows.append({"beat": i, "patient": sel["id"], "cohort": sel["cohort"],
                     **{f"{k}_loss": fr[k]["loss"] for k in ("lofi", "hifi_phi", "hifi_joint")},
                     **{f"abl_{g}": fr["ablation"][g]["loss"] for g in FIT_GROUPS},
                     **{f"gain_{g}": fr["hifi_joint"]["gains"][g] for g in FIT_GROUPS}})
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
    for ax in axes[-1]:
        ax.set_xlabel("ms", fontsize=8, color="#6b6a63")
    fig.suptitle("Real beats vs best-fit low-fi and high-fi (fit on Prv, Pra, Pap; L/P/J = low-fi / "
                 "phi-only / joint NRMSE)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=120)
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(tab.round(4).to_string(index=False))
    print("wrote", fig_path)


def best_shift(sim, real):
    period = sim[:-1]
    best, out = np.inf, sim
    for s in range(-MAX_SHIFT, MAX_SHIFT + 1):
        x = np.roll(period, s)
        x = np.append(x, x[0])
        e = np.mean((x - real) ** 2)
        if e < best:
            best, out = e, x
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "fit", "plot"])
    ap.add_argument("out")
    ap.add_argument("fig", nargs="?")
    ap.add_argument("--n-beats", type=int, default=6)
    ap.add_argument("--beat", type=int)
    ap.add_argument("--workers", type=int, default=40)
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    if a.stage == "select":
        select(out, a.n_beats)
    elif a.stage == "fit":
        fit_beat(out, a.beat, a.workers)
    else:
        plot(out, a.fig)


if __name__ == "__main__":
    main()
