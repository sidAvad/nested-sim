"""Inference adapter for the Sun et al. Route A baseline, for this project's eval harness.

Route A's inference network is the conditional MDN p_beta(theta|x) trained on the pairs the
annealed Langevin synthesised (`cv-dann-sbi-gibbs/baselines/sun_routeA`). Two things make it
unlike the flow-based methods in `eval_pretrained_lofihifi.py`:

  - theta lives in unit-cube coordinates, so draws must be pushed back through BoxToUnit's
    inverse to be scored against physical ground truth;
  - the prior box is enforced at inference by rejection (`sample_with_rejection`), not by a
    bounded transform, so a large fraction of raw draws land outside it.

`x` needs no adaptation: Route A's base set is built by `routeA_data.build_base` through the
same `ReducedCVDataset` and the same `lofi_train/norm_stats.json` that `build_patients` uses,
giving an identical 809-dim layout (4 z-scored waves, then Pas mean/max/min, SV/vlv_std, HR_z).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HOME = Path.home()
GIBBS_REPO = HOME / "projects/cv-dann-sbi-gibbs"
ROUTEA = GIBBS_REPO / "baselines/sun_routeA"
OUTPUTS_ROOT = HOME / "outputs/cv-dann-sbi"


def _paths():
    for p in (GIBBS_REPO, ROUTEA, ROUTEA / "vendor"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def resolve_mdn_ckpt(run, ckpt_ts=None):
    cks = sorted((OUTPUTS_ROOT / run / "checkpoints").glob("*/"))
    if not cks:
        raise FileNotFoundError(f"no checkpoints under {OUTPUTS_ROOT / run / 'checkpoints'}")
    if ckpt_ts is None:
        return cks[-1]
    return OUTPUTS_ROOT / run / "checkpoints" / ckpt_ts


def load_routeA(run, device, ckpt_ts=None, nested_sim_root=HOME / "projects/nested-sim"):
    """Returns (model, norm_theta, norm_x, box_to_unit, meta)."""
    _paths()
    from RouteB_utils import ConditionalMDN                              # noqa: E402
    from routeA_data import box_to_unit                                  # noqa: E402

    ckpt_dir = resolve_mdn_ckpt(run, ckpt_ts)
    d = torch.load(ckpt_dir / "mdn.pt", map_location=device, weights_only=False)
    arch = d["arch"]
    model = ConditionalMDN(theta_dim=d["theta_dim"], x_dim=d["x_dim"],
                           hidden=arch["hidden"], n_components=arch["n_components"],
                           n_hidden_layers=arch["n_hidden_layers"]).to(device)
    model.load_state_dict(d["model"])
    model.eval()

    tf, _ = box_to_unit(str(nested_sim_root))
    tf = tf.to(device)
    cfg_path = ckpt_dir / "config.json"
    meta = {"ckpt_dir": str(ckpt_dir), "best_val": d.get("best_val"),
            "train_betas": json.load(open(cfg_path)).get("betas") if cfg_path.exists() else None}
    return model, d["norm_theta"], d["norm_x"], tf, meta


@torch.no_grad()
def run_posterior_inference_routeA(model, norm_theta, norm_x, tf, patients, beta,
                                   n_samples, device):
    """Same return convention as eval_common.run_posterior_inference.

    Draws are taken once per case without rejection and all are returned in physical units,
    with `accept_rates` recording the fraction inside the prior box. Downstream scoring
    applies that mask itself, so the ensemble a rejection sampler would have produced is
    recoverable without this function having to oversample to a fixed count (which is what
    upstream's `sample_with_rejection` does, and which silently pads when it cannot fill).
    """
    _paths()
    from RouteB_utils import mdn_sample                                  # noqa: E402

    theta_means, all_samples, accept_rates = [], [], []
    n_clipped = 0
    for pat in patients:
        x = pat["x_avg"].reshape(1, -1).to(device)
        u = mdn_sample(model, x, float(beta), norm_theta, norm_x,
                       num_samples=n_samples, device=device)
        in_box = ((u >= 0.0) & (u <= 1.0)).all(dim=1)
        accept_rates.append(float(in_box.float().mean()))
        # inverse() is applied to out-of-box draws too, so the caller can map the whole
        # ensemble back to unit coordinates and score it there. Only the log dims can
        # overflow (they exponentiate), so only those are clipped, far enough out
        # (u in [-20, 21]) that the clip is inert on any draw near the cube.
        u_safe = torch.where(tf.log_mask, u.clamp(-20.0, 21.0), u)
        n_clipped += int((u_safe != u).any(dim=1).sum())
        phys = tf.inverse(u_safe)[0].double().cpu()
        ok = phys[in_box.cpu()] if bool(in_box.any()) else phys
        theta_means.append(ok.mean(0).numpy())
        all_samples.append(phys.numpy())
    if n_clipped:
        print(f"  routeA: {n_clipped} draws clipped at the log-dim overflow guard "
              f"(of {len(patients) * n_samples})", flush=True)
    return np.stack(theta_means), np.stack(all_samples), np.array(accept_rates)
