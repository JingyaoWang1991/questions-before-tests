"""figS5_layers.py — Figure S5: three transport settings on identical metrics.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from style_supp import (SRC, GREY, DARK, ALERT, OK, LIGHT, W2, MM, fig, tidy,
                           save, panel_letter)

lay = pd.read_csv('results/transport_layers.csv')
sr = pd.read_csv('results/standardised_risk.csv')
LAB = {'CHARLS': 'China', 'ELSA': 'England', 'HRS': 'USA', 'SHARE': 'Europe'}

LAYERS = ['L1  other wave', 'L2  other country, same instrument',
          'L3  other country, other survey']
# the left-hand axis carries the setting names; panel b needs no repeat of them
LLAB = {'L1  other wave': 'same survey,\nother wave',
        'L2  other country, same instrument': 'other country,\nsame instrument',
        'L3  other country, other survey': 'other country,\nother survey'}
LCOL = {'L1  other wave': '#6b5b95',
        'L2  other country, same instrument': SRC['HRS'],
        'L3  other country, other survey': SRC['CHARLS']}

fig = plt.figure(figsize=(W2, 72 * MM))
gs = fig.add_gridspec(1, 3, width_ratios=[1.30, 1.00, 0.86], wspace=0.34)

# ---------------------------------------------------------------- panel a
ax = fig.add_subplot(gs[0, 0])
ax.axvspan(0.8, 1.2, color=LIGHT, zorder=0)
ax.axvline(1.0, color=GREY, lw=0.8, ls=(0, (3, 2)), zorder=1)
rng = np.random.default_rng(11)
for i, L in enumerate(LAYERS):
    v = lay[lay['layer_short'] == L]['cal_slope'].dropna().values
    y = 2 - i
    ax.scatter(v, y + rng.normal(0, 0.055, len(v)), s=16, color=LCOL[L],
               edgecolor='white', linewidth=0.4, zorder=3)
    ax.plot([v.min(), v.max()], [y, y], color=LCOL[L], lw=1.0, alpha=0.35, zorder=2)
    ax.plot(v.mean(), y, '|', ms=11, mew=1.6, color=DARK, zorder=4)
ax.set_yticks([2, 1, 0])
ax.set_yticklabels([LLAB[L] for L in LAYERS], fontsize=6.3)
ax.set_ylim(-0.6, 2.6)
ax.set_xlim(0.48, 1.35)
ax.set_xlabel('calibration slope after transport', fontsize=6.6)
ax.text(1.0, 2.46, 'calibrated band', ha='center', va='top', fontsize=6.2,
        color='#6b7f8f')
tidy(ax)
ax.margins(x=0.02)
means = ' / '.join(f"{np.abs(lay[lay['layer_short'] == L]['cal_slope'].dropna().values - 1).mean():.2f}"
                   for L in LAYERS)
ax.text(0.5, -0.20, f'mean |slope\u22121| = {means}, top to bottom',
        transform=ax.transAxes, ha='center', va='top', fontsize=6.2, color=GREY)
panel_letter(ax, 'a', dx=-0.42, dy=1.03)

# ---------------------------------------------------------------- panel b
ax = fig.add_subplot(gs[0, 1])
for i, L in enumerate(LAYERS):
    v = lay[lay['layer_short'] == L]['auc_loss'].dropna().values
    y = 2 - i
    ax.scatter(v, y + rng.normal(0, 0.05, len(v)), s=16, color=LCOL[L],
               edgecolor='white', linewidth=0.4, zorder=3)
    ax.plot(v.mean(), y, 'D', ms=4.2, color=DARK, zorder=4,
            markeredgecolor='white', markeredgewidth=0.5)
    ax.text(v.mean(), y + 0.24, f'{v.mean():.3f}', fontsize=6.2, ha='center',
            va='bottom', color=DARK)
ax.set_yticks([2, 1, 0])
ax.set_yticklabels(['', '', ''])          # the names are on panel a
ax.set_ylim(-0.6, 2.6)
ax.set_xlim(-0.004, 0.030)
ax.set_xlabel('AUC lost on transport', fontsize=6.6)
tidy(ax)
ax.margins(x=0.04)
panel_letter(ax, 'b', dx=-0.16, dy=1.03)

# ---------------------------------------------------------------- panel c
ax = fig.add_subplot(gs[0, 2])
order = ['CHARLS', 'ELSA', 'HRS', 'SHARE']
ypos = np.arange(len(order))[::-1]
for y, s in zip(ypos, order):
    r = sr[sr['src'] == s].iloc[0]
    c0, c1 = r['crude_dis'] * 100, r['std_dis'] * 100
    ax.plot([c0, c1], [y, y], color=SRC[s], lw=1.6, zorder=2,
            solid_capstyle='round')
    ax.plot(c0, y, 'o', ms=4.0, color='white', markeredgecolor=SRC[s],
            markeredgewidth=1.0, zorder=3)
    ax.plot(c1, y, 'o', ms=4.4, color=SRC[s], markeredgecolor='white',
            markeredgewidth=0.5, zorder=4)
    ax.text(max(c0, c1) + 0.5, y, f'{c1:.1f}', fontsize=6.2, va='center',
            ha='left', color=SRC[s])
ax.set_yticks(ypos)
ax.set_yticklabels([LAB[s] for s in order], fontsize=6.3)
ax.set_xlim(6, 22)
ax.set_xlabel('two-year ADL-difficulty risk (%)', fontsize=6.6)
ax.plot([], [], 'o', ms=4, color='white', markeredgecolor=GREY, markeredgewidth=1.0,
        label='crude')
ax.plot([], [], 'o', ms=4.4, color=GREY, label='age-sex standardised')
ax.legend(frameon=False, fontsize=6.2, loc='lower left', handlelength=1.0,
          borderpad=0.1, ncol=2, columnspacing=1.2, handletextpad=0.3)
ax.set_ylim(-1.25, 3.4)
tidy(ax)
ax.margins(x=0.06, y=0.10)
panel_letter(ax, 'c', dx=-0.40, dy=1.03)

save(fig, 'figS5_layers')
print('fig5 (layers) written')
print(lay.groupby('layer_short').agg(
    abs_slope_dev=('cal_slope', lambda s: (s - 1).abs().mean()),
    auc_loss=('auc_loss', 'mean')).round(3).to_string())
