# nested-sim

Nested low-fi / high-fi 8-compartment cardiovascular simulator for a sim-to-sim
domain-adaptation benchmark. Low-fi is a Python port of the `Cv8SimApp` binary
(cv8Eed model); high-fi adds a 4-element Windkessel, semilunar valve inertance
and freed resistances (phi), and reduces exactly to low-fi when phi is off.

```python
from nested_sim import simulate, PHI_OFF, sample_phi
lo = simulate(theta)                     # low-fi
hi = simulate(theta, PHI_OFF)            # must equal lo
phi = sample_phi(rng, alpha=0.5, theta=theta)
hi = simulate(theta, phi)                # structural phi
obs = measure(hi, phi, rng)              # identity unless alpha_meas > 0
```
theta uses ASCII names (`tau`, `tau_a`); see `nested_sim/params.py` for the name
map to the binary and the PDF write-up.

## Low-fi model

Ported from the Julia source in `reference/julia/` (`cv8Eed.jl`,
`dynamic_elastance.jl`, `cv8Eed_constants_031326.jl`). Notes:

- Activation is `epsilon_smooth` (logistic blend of sine rise and exponential
  decay, a = 0.03, b = 3, minus epsilon(0)). t = 0 is atrial onset; ventricles
  are delayed by AVD (the PDF's `t - AVD` formula has this reversed for atria).
- Rcs/Rcp = aortic/pulmonic valve resistances (PDF Rav/Rpv); Rra/Rla = sys/pulm
  vein -> atrium (PDF Rvs/Rvp). Hard diodes. Flow outputs are labelled by the
  compartment they enter.
- Not in the provided source (`cv8.jl`, `odesolve.jl` missing), so inferred from
  the binary: the `PV_OPEN` beat starts at the first grid point `j*T/200` at/after
  pulmonic valve opening; summaries are plain statistics of the 201-point beat
  (mean includes both endpoints), pcw = mean Pvp, cvp = mean Pra; lvedp/rvedp
  only approximated (not used downstream).
- The binary reports `success` even when its periodicity check never passed;
  treat `steadystate_error > tol` as failure.

Validation (`scripts/compare_binary.py`, 60 uniform-prior theta, binary at
reltol 1e-9): 47 converge in both; sv and all max/min pressures agree to
<= 3e-4 (median ~1e-5); waveforms median ~1e-5. Two high-HR beats with
incomplete relaxation start one grid step (T/200) later in the binary, which
also moves their mean pressures by up to 0.02 mmHg.

## Layout

- `nested_sim/` - package (`lowfi.py`, `hifi.py`, `solver.py`, `phi.py`,
  `activation.py`, `params.py`)
- `configs/phi_priors.toml` - phi priors (all PLACEHOLDERS, see spec Open items)
- `ref/` - binary reference runs (`ref_params.json`, `ref_nt200.h5`,
  `ref_nt4000.h5`, `ref_default.h5`); regenerate on palladium from
  `~/outputs/nested-sim/ref/`
- `reference/` - model write-up PDF and the Julia source of the binary
- `scripts/compare_binary.py` - low-fi vs binary comparison

## High-fi model notes

- Only active extra states are integrated (Q_L when that Windkessel is on, Qv
  when Lv > 0). A state with identically zero derivative makes scipy's
  finite-difference Jacobian blow its perturbation up to inf (NaNs, stalls).
- Valve inertance: under a reverse gradient the decelerating pressure is gated
  by clip(Qv / 1e-3 mL/ms, 0, 1); without it the RHS jumps at Qv = 0 and solver
  noise there collapses the step size. Lv -> 0 converges linearly to the Lv = 0
  branch (error ~1e-3 mmHg at Lv = 0.1).
- `scripts/check_hifi.py` (200 random theta, phi at alpha=1, placeholder priors):
  success 98.5% for low-fi, hifi(off) and hifi alike; nesting exact (0.0);
  runtime median 1.2 s low-fi, 4.3 s high-fi (p95 13 s), LSODA.
- Figures (palladium `~/results/nested-sim/figures/`, synced locally to
  `~/sa4604/results_palladium/nested-sim/figures/`):
  `hifi_vs_lowfi_alpha1.png` (T7 visual), `measurement_check.png`.

## Measurement layer

`nested_sim/measurement.py` acts only on the 4 cath lab waveforms (Prv, Pra,
Pap, Pvp): one catheter, so shared 2nd-order low-pass (fn_cath, zeta_cath),
shared zero offset, plus white noise. Everything else (Pas, volumes, flows) is
left exactly as simulated. Scalars are not modelled separately: as in the
binary, they are `summaries()` of the measured 201-pt waveforms (so sbp/dbp/map
and sv are exact; spa/dpa/mpap/pcw/cvp/rv_* carry the catheter effects).
Sequential (pullback) recording is not modelled; a per-channel time shift
could be added later as one more phi (off = 0).
`scripts/check_measurement.py`: exact identity with phi off; prewarped bilinear
filter matches the analog H(jw) to <= 7e-4 relative (exact at fn), unit DC gain.

## phi priors (`configs/phi_priors.toml`)

Structural priors are conditional on theta; phi always stores physical values
and alpha interpolates them from off (`sample_phi(rng, alpha, theta)`).
Each high-fi element is independently switchable (`PHI_GROUPS`: wk_s, wk_p,
lv_av, lv_pv, res, av_l, av_r, resp, wedge, meas); `sample_phi(..., groups=[...])`
keeps only the listed elements on, and any phi left out of a dict is off.

| phi | prior (log-uniform) |
|---|---|
| Rmv, Rav, Rtv, Rpv, Rvs, Rvp | low-fi value x [0.5, 2] |
| Zc_s | Ras x [0.02, 0.05] |
| L_s | Zc_s x tau_L, tau_L in [100, 1000] ms |
| Zc_p | Rap x [0.05, 0.15], capped at 100 |
| L_p | Zc_p x tau_L, tau_L in [200, 2000] ms |
| Lv_av, Lv_pv | R_v^2 / (4 zeta_v^2 Emax), zeta_v in [0.7, 2] |
| k_av_l, k_av_r | U[0.1, 0.5] (AV-plane coupling) |
| resp_amp | U[1, 4] mmHg; resp_rate U[10, 20]/min and resp_phase U[0, 1) are nuisances (not alpha-scaled) |
| w_wedge, tau_wedge | no prior (implemented, parked) |
| fn_cath, zeta_cath | U[10, 25] Hz, U[0.15, 0.6] (fn interpolated in 1/fn) |
| offset_cath, sigma_cath | U[-3, 3], U[0, 0.3] mmHg |

Why (sweeps over 300 prior theta, `scripts/prior_sweep.py`,
`scripts/prior_proposal.py`; CSVs in palladium `~/outputs/nested-sim/prior_sweep/`):
- Absolute Zc_p is meaningless across Rap in [10, 1200]; Zc as a fraction of R.
- tau_L < ~100 ms produces a large post-systolic dip/rebound (up to 30 mmHg);
  above that the rebound stays <= ~2 mmHg (p95).
- Absolute Lv rings with a stiff ventricle (valve + chamber LC oscillator,
  damping R_v / (2 sqrt(Emax Lv))): Emax_RV 3.6, Lv_pv 50 gives 4 flow peaks per
  ejection. Parameterising by zeta_v >= 0.7 avoids it.
- Joint check at alpha = 1 (`proposal_joint_v1.csv`): 100% success; systolic PP
  x1.24 median (p95 x2.5), pulmonary x1.07 (p95 x2.1); systolic peak 36 ms earlier
  (median); rebound <= 1.5 mmHg (p95); MAP -6..+4 mmHg (p5-p95); SV +-3 mL; flow
  peak counts as in low-fi.
- Limitation: pulmonary hangout stays short (median ~2 ms vs 30-80 ms
  physiological); valve inertance cannot produce more without ringing.
- Pulmonary PP tail (up to ~x5 at p99) comes from a low-fi theta corner (very
  high Rap with very low Eap: tiny low-fi PA pulse), not from high-fi.

## Real data comparison (palladium `~/data/real_data/onebeat_300patients`)

802 beats, 144 patients (4-8 beats each), channels Prv, Pra, Pap, Pvp (wedge).
Cath summaries are waveform summaries; map/sbp/dbp are integer readings.
Scripts: `real_vs_sim.py`, `real_checks.py`, `calibrate_lowpass.py`,
`wedge_experiment.py`, `wedge_matched.py`, `resp_experiment.py`,
`attribute_phi.py`; outputs under palladium `~/outputs/nested-sim/` and
`~/results/nested-sim/real_vs_sim/`.

- Real beats carry no white noise and no catheter ringing (100x smoother than
  noiseless sims); a catheter low-pass does not reproduce their spectra.
  -> measurement model is not supported by the data; keep it out of the main gap.
- Real "Pvp" is a wedge (a + v waves, 2nd harmonic dominant in 72% of beats,
  2.8 mmHg). The model family rarely produces this (4-6% of prior sims, no theta
  region explains it): open structural gap, parked. AV-plane coupling closes
  most of the RA shape gap (double-hump share 0.17 -> 0.36, real 0.46).
- Beat-to-beat variability (first-order respiration removed, second order kept):
  respiration with a 3-4 mmHg swing matches Prv/Pap level and shape variation;
  Pra prefers ~1 mmHg. Real Pra-Pvp level shifts correlate +0.96, which second-
  order respiration cannot produce (sim: -0.1): a small shared additive residual
  in that simultaneously recorded pair. Real Prv-Pap correlate only +0.45
  (sequential recording); sims +0.89. Mimicking sequential recording (channels
  from different beats of `simulate_breath`) is possible but not implemented.

## High-fi with all phi (alpha = 1, 200 theta, `check_hifi.py`, `attribute_phi.py`)

Success 97.5% (low-fi 98.5%; the 2 extra failures are near-unstable theta pushed
over by the freed resistances); nesting exact. Runtime median 9-11 s (p95 ~22 s),
dominated by valve inertance. Effects are super-additive: SBP +10 mmHg median with
all groups (Windkessel alone +5, respiration +1.7, AV coupling +1.3).
For datasets use ss_tol ~1e-6 (1e-7 is occasionally missed by a hair).

The SBP shift is high-fi minus low-fi on the same theta (simulator gap), not a
realism measure. Matched to real beats on cath summaries + HR + MAP
(`matched_systemic.py`): pulse pressure real 49, low-fi 24, high-fi 36 mmHg
(per-beat error -21 -> -13); DBP 79 / 78 / 78. High-fi moves toward real.

## Fit to real beats (v1, `scripts/fit_real_beats.py`)

6 representative beats (patient-cluster medoids: 2 controls, 4 PAH), fit on
Prv/Pra/Pap (Pvp, respiration, wedge excluded); outputs palladium
`~/outputs/nested-sim/fit_real/v1/`, figure `figures/fit_real_v1.png`.
- After fitting theta, low-fi already matches Prv/Pra/Pap to 3-9% NRMSE
  (worst beat 8-15%). Best high-fi fit lowers the loss 0-28% (median ~13%);
  visually small. Of single elements, only the freed resistances help in every
  beat; the rest add ~nothing alone.
- Systemic elements (wk_s, lv_av, av_l) barely touch the fitted right-heart
  channels, so their fitted gains are arbitrary; left-heart theta is likewise
  unconstrained (unfitted Pvp mean off by up to 20 mmHg).
- The joint theta+phi CMA-ES mostly did not leave its start point (45 dims,
  sigma 0.05, 60 generations): theta-shift results are inconclusive.
- Next (v2): add real scalars (pcw = mean wedge, map/sbp/dbp, sv) to the loss so
  left-heart/systemic theta and the systemic elements are constrained (the
  matched pulse-pressure analysis suggests that is where high-fi helps most);
  stronger joint search or drop it.

## Theta-bias tests (`scripts/theta_bias.py`, `compare_bias.py`)

Low-fi best fit (CMA-ES, common perturbed start) to high-fi data (A) vs to
low-fi data (B, control floor); encoder observation (4 cath waves + MAP/SBP/
DBP/SV, encoder normalisation), block-balanced loss, 200 targets. Outputs
palladium `~/outputs/nested-sim/theta_bias/v{1,2,3}_block/`; comparison figure
`figures/theta_bias_v1_v2_v3.png`.

| priors | loss at theta* | after low-fi refit | Cas signed shift (median) | other params |
|---|---|---|---|---|
| v1 original | 0.074 | 0.017 | -0.04 | within floor (~0.03-0.05 of prior range) |
| v2 lever 1 (bigger systemic Windkessel) | 0.134 | 0.028 | -0.12 (IQR to -0.27) | within floor |
| v3 lever 1 + interdependence | 0.195 | 0.028 | -0.11 | small extra RV/LV passive shifts, within floor |

- Lever 1 triples the Cas bias, clearly above the floor, and makes high-fi match
  real pulse pressure (48 vs 49) and RA shape (0.46 vs 0.46).
- Interdependence adds misfit at theta* but low-fi absorbs it almost entirely
  through small shifts spread over RV/LV passive parameters.

## Benchmark levels (decided 2026-10-05)

| level | simulator | priors |
|---|---|---|
| none | low-fi (cv8Eed port) | - |
| realistic | high-fi | `configs/phi_priors.toml` = `phi_priors_v2.toml` |
| exaggerated | high-fi | `configs/phi_priors_v2strong.toml` (same elements, wider ranges) |

alpha in [0, 1] interpolates within a level. Ventricular interdependence
(`c_spt`, `phi_priors_v3.toml`) is implemented but off in both levels: it kept
pulse pressure realistic (45 vs real 49) but lowered the matched RA
double-hump share (0.37 vs v2 0.46 = real). Realism of v2 (matched to real
beats): pulse pressure 48 vs 49, RA double-hump 0.46 vs 0.46.

## Datasets (generated 2026-10-05/06, commit 0907e25, `scripts/make_datasets.py`)

palladium `~/outputs/nested-sim/data/<set>/` (66 GB), Cv8SimApp-export layout
(batch_XXXX.h5 + manifest.json; waves 24 continuous + 4 valve + t, float32;
summaries; parameters with binary names incl. fixed; phi for high-fi).

| set | simulator | sims | failed draws (replaced) |
|---|---|---|---|
| lofi_train | low-fi | 990,000 | 17,451 (1.7%) |
| lofi_test / hifi_v2_test / hifi_v2strong_test | paired, same theta | 10,000 each | 397 theta dropped (any of the 3 failed) |
| hifi_v2_target | high-fi v2 (realistic) | 90,000 | 2,023 (2.2%) |
| hifi_v2strong_target | high-fi v2-strong (exaggerated) | 90,000 | 3,157 (3.4%) |

Target pools are meant to be used unlabeled in training; their labels are
stored for evaluation. Median pulse pressure: low-fi 21, v2 45, v2-strong 65.
Normalisation stats in the cv-* repos come from the binary 10M set; recompute
from lofi_train if a method needs them (e.g. cv-sbi-sim scripts/compute_stats.py).

## Tests

`tests/` (pytest; on palladium: `.venv/bin/python -m pytest -n 128 --runslow tests`,
183 tests in ~40 s):

| File | Covers |
|---|---|
| test_nesting.py | T1 hifi(PHI_OFF) == lowfi on 100 random theta (env NESTED_SIM_T1_N), identical state vector; T2 each switch/partner alone off (Zc/L, k_av, resp_amp, w_wedge, Lv, freed R at low-fi values) |
| test_physics.py | T3 R_art is the mean-flow resistance (both outlets, with/without valve inertance); T4 volume conservation over a breath with all phi on; T5 Lv -> 0 linear convergence; respiration beat count, transmural output, resp_amp -> 0 continuity, resp_phase selects the beat; AV coupling continuity; activation eps(0) = 0, peak at Tmax |
| test_phi.py | alpha = 0 is off; measurement off unless alpha_meas > 0; same seed -> same draws along alpha (tau_L fixed, nuisances unscaled); conditional priors within range (Zc/R, cap, damping ratio, resistance multipliers) |
| test_measurement.py | T6 identity when off; only cath channels change; filter vs analog H(jw) |
| test_binary.py | low-fi vs Cv8SimApp reference (scalars 1e-3 mmHg, waves 0.5% of pulse; grid points exactly on an activation reset excluded) |
| test_figures.py | T7 figure script runs (`--runslow`); visual check is manual |

## Status (2026-10-01)

Done: low-fi port (from Julia source) + validation; high-fi model + robustness
check; measurement layer (cath channels only) + checks; theta-conditional phi
priors + joint check; real-data comparison; AV-plane coupling; respiration
(`simulate_breath` returns every beat of a breath).

Measurement model is a separate axis: `sample_phi(..., alpha_meas=0)` by default
(off). Parked: wedge shape (LA mechanism).

TODO, in order:
1. (done) Tests T1-T7 + new-mechanism tests under `tests/`.
2. Dataset generation script (theta, phi, alpha, seed, dense raw + measured
   waves, 201-pt inputs + scalars), parallel on palladium.
3. (done) Git repo github.com/sidAvad/nested-sim; palladium ~/projects/nested-sim is a read-only deploy-key clone.
4. Optional: cv8.jl / odesolve.jl to close the 2 window-offset beats and EDP.
5. Optional: mimic sequential (pullback) recording - build each sample's
   channels from different beats of one `simulate_breath` breath (Pra + Pvp
   from one beat, Prv and Pap from others) to reproduce the real cross-channel
   beat-to-beat correlations (real Prv-Pap +0.45, sim +0.89).
6. Optional: wedge (LA) mechanism - proximal pulmonary-vein segment, atrial
   activation shape, or mitral inertance; judge with scripts/wedge_matched.py.
