"""Same format as cv-sbi-spin/scripts/plot_zscore_shrinkage_across_runs_merged_with_gibbs.py,
for the nested-sim-trained (lofi 300k sims + hifi_v2strong real-stand-in) models,
evaluated on the real 802-patient cohort (eval_zscore_real_lofihifi.py's output).

Usage:
    python plot_zscore_violin_lofihifi_real.py
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path.home() / 'results/nested-sim/pretrained_eval_lofihifi_realpatients'

METHODS = [
    ('Path A ("regular SPIN")', ROOT / 'spin-pathA-lofihifi' / 'zscore_shrinkage_data.npz'),
    ('Vanilla NPE',             ROOT / 'npe-baseline-lofihifi' / 'zscore_shrinkage_data.npz'),
    ('NPE + 11dB noise',        ROOT / 'npe-noise-lofihifi' / 'zscore_shrinkage_data.npz'),
    ('Gibbs NPE (w=5.92)',      ROOT / 'npe-gibbs-lofihifi' / 'zscore_shrinkage_data.npz'),
    ('Path D (SPIN)',           ROOT / 'spin-pathD-lofihifi' / 'zscore_shrinkage_data.npz'),
]

PARAMS = ['Rap', 'Ras']
LABELS = {'Rap': 'Rap (PVR)', 'Ras': 'Ras (SVR)'}
COLORS = {'Rap': 'mediumseagreen', 'Ras': 'orchid'}
OFFSET = 0.2
WIDTH = 0.35


def main():
    method_names = [name for name, _ in METHODS]
    data = {param: [] for param in PARAMS}
    for name, path in METHODS:
        d = np.load(path)
        for param in PARAMS:
            data[param].append(d[f'{param}_z'])

    fig, ax = plt.subplots(figsize=(4 * len(METHODS), 7))

    for param, sign in zip(PARAMS, (-1, 1)):
        positions = [i + sign * OFFSET for i in range(len(METHODS))]
        parts = ax.violinplot(data[param], positions=positions, widths=WIDTH,
                               showmedians=True, showextrema=True)
        for body in parts['bodies']:
            body.set_facecolor(COLORS[param])
            body.set_alpha(0.6)
        for key in ('cmedians', 'cbars', 'cmins', 'cmaxes'):
            parts[key].set_color(COLORS[param])

        for i, z in enumerate(data[param]):
            med = np.median(z)
            frac2 = (np.abs(z) > 2).mean()
            ax.annotate(f'{med:.1f}\n{frac2:.0%}', (i + sign * OFFSET, ax.get_ylim()[1]),
                        ha='center', va='bottom', fontsize=6.5, color=COLORS[param])

    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.axhline(2, color='gray', linestyle=':', linewidth=0.8)
    ax.axhline(-2, color='gray', linestyle=':', linewidth=0.8)
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels(method_names, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('z-score  ((post_mean - true) / post_std)')

    legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[p], alpha=0.6, label=LABELS[p])
                       for p in PARAMS]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=9)

    fig.suptitle('Posterior z-score distribution by method (lofi/hifi-trained, 300k sims),\n'
                  'evaluated on the real 802-patient cohort\n'
                  'dotted lines: |z|=2; annotations: median / frac |z|>2; '
                  'Gibbs NPE on its held-out test patients, others on all 802',
                  fontsize=11)
    plt.tight_layout()

    ROOT.mkdir(parents=True, exist_ok=True)
    out = ROOT / 'zscore_violin_lofihifi_real_merged_with_gibbs.png'
    fig.savefig(out, dpi=150)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
