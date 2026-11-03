"""fig2_increment.py — Figure 2: incremental discrimination by tier and domain.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from style_main import (INK, INK2, MUTED, GRID, SLATE, PALE, COH, MK, CLAB, MM, shade, tint,
                       PT_TITLE, PT_ROW, PT_AXIS, PT_TICK, PT_LEG, PT_VAL,
                       PT_VAL_SM, canvas, clean, panel_letter, zero_line,
                       ref_line, export)

d = pd.read_csv('results/paired_delta_b1000.csv')
it = pd.read_csv('results/item_level_exploratory.csv')
it = it[it['block'] == 'POOLED'].set_index('model')

fig = canvas(170, 92)
gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.34,
                      left=0.180, right=0.972, top=0.905, bottom=0.185)

rows = []

# ============================================================== panel a
ax = fig.add_subplot(gs[0, 0])
# Comparisons are separate paired samples, not a cumulative vertical ladder.
TIERS = [('R->Q', '+ questionnaire'),
         ('Q->E1', '+ grip'),
         ('Q->E2', '+ memory'),
         ('Q->E3', '+ grip + memory'),
         ('E3->L', '+ prior wave')]
BLOCKS = ['CHARLS', 'ELSA', 'HRS', 'POOLED']
OFF = {'CHARLS': -0.24, 'ELSA': -0.08, 'HRS': 0.08, 'POOLED': 0.24}
PITCH = 1.0

ys = {t: i for i, (t, _) in enumerate(TIERS)}
allv = []
for blk in BLOCKS:
    sub = d[d['block'] == blk].set_index('comparison')
    for tier, _ in TIERS:
        if tier not in sub.index:
            continue
        r = sub.loc[tier]
        if not np.isfinite(r['dauc']):
            continue
        y = ys[tier] + OFF[blk]
        col = COH[blk]
        ax.plot([r['dauc_lo'], r['dauc_hi']], [y, y], color=col, lw=1.0,
                solid_capstyle='butt', zorder=4)
        for xv in (r['dauc_lo'], r['dauc_hi']):
            ax.plot([xv, xv], [y - 0.055, y + 0.055], color=col, lw=0.9, zorder=4)
        ax.plot(r['dauc'], y, MK[blk], ms=5.0, color=col,
                markeredgecolor='white', markeredgewidth=0.5, zorder=5)
        rows.append((blk, tier, round(float(r['dauc']), 4),
                     round(float(r['dauc_lo']), 4), round(float(r['dauc_hi']), 4),
                     int(r['n'])))
        allv += [r['dauc_lo'], r['dauc_hi']]

zero_line(ax, 'v')
ax.set_axisbelow(True)
ax.xaxis.grid(True, color=GRID, alpha=0.55, linewidth=0.5, zorder=0)
ax.set_yticks([ys[t] for t, _ in TIERS])
ax.set_yticklabels([lab for _, lab in TIERS], fontsize=PT_ROW)
ax.set_ylim(len(TIERS) - 0.35, -0.65)
xmin = min(allv) - 0.005
xmax = max(allv) + 0.005
ax.set_xlim(xmin, xmax)
ax.set_xticks([0, 0.02, 0.04, 0.06, 0.08])
ax.set_xlabel('\u0394AUC (95% CI)', fontsize=PT_AXIS)
clean(ax, left=False, bottom=True)
ax.tick_params(axis='y', length=0)

handles = [Line2D([], [], marker=MK[b], linestyle='none', ms=5.0, color=COH[b],
                  markeredgecolor='white', markeredgewidth=0.5, label=CLAB[b])
           for b in BLOCKS]
ax.legend(handles=handles, frameon=False, fontsize=PT_LEG, loc='upper center',
          bbox_to_anchor=(0.5, -0.135), ncol=4, handletextpad=0.35,
          columnspacing=1.1, borderpad=0, labelspacing=0.3)
panel_letter(ax, 'a', dx=-0.30, dy=1.02)

# ============================================================== panel b
ax = fig.add_subplot(gs[0, 1])
# Hue answers the clinical question this panel exists to answer: does the model
# need the 8-10 item depressive-symptom scale? Cool bars do not, warm bars do.
# Depth separates a single domain from a combined model, so the two dark bars are
# the ones to compare: two single questions (0.055) against all three (0.061).
CHEAP = SLATE                    # no mood scale
CHEAP_LIGHT = tint(SLATE, 0.55)
COSTLY = '#B06844'               # includes the mood scale
COSTLY_LIGHT = tint(COSTLY, 0.52)
BARS = [('R+SRH', 'self-rated health', CHEAP_LIGHT, False),
        ('R+climb', 'stair climbing', CHEAP_LIGHT, False),
        ('R+mood', 'depressive symptoms', COSTLY_LIGHT, False),
        ('R+SRH+climb', 'two single questions', CHEAP, True),
        ('Q', 'all three domains', COSTLY, True)]
yy = np.arange(len(BARS))[::-1]
H = 0.55
for y, (k, lab_, col, aggregate) in zip(yy, BARS):
    r = it.loc[k]
    ax.barh(y, r['dauc_vs_R'], H, color=col, edgecolor='none', zorder=2)
    ax.plot([r['dauc_lo'], r['dauc_hi']], [y, y], color=INK, lw=0.9, zorder=4)
    for xv in (r['dauc_lo'], r['dauc_hi']):
        ax.plot([xv, xv], [y - 0.055, y + 0.055], color=INK, lw=0.9, zorder=4)
    ax.text(r['dauc_hi'] + 0.0022, y, f"{r['dauc_vs_R']:.3f}", fontsize=PT_VAL,
            color=INK, ha='left', va='center',
            fontweight='bold' if aggregate else 'normal')
    rows.append(('pooled item model', k, round(float(r['dauc_vs_R']), 4),
                 round(float(r['dauc_lo']), 4), round(float(r['dauc_hi']), 4),
                 int(r['n'])))

sum3 = float(it.loc[['R+SRH', 'R+climb', 'R+mood'], 'dauc_vs_R'].sum())
ref_line(ax, sum3, 'v', label=f'sum of single domains ({sum3:.3f})')
rows.append(('reference', 'sum of three single domains', round(sum3, 4), '', '', ''))

ax.set_yticks(yy)
ax.set_yticklabels([lab_ for _, lab_, _, _ in BARS], fontsize=PT_ROW)
ax.set_ylim(-0.62, len(BARS) - 0.42)
ax.set_xlim(0, 0.105)
ax.set_xticks([0, 0.05, 0.10])
ax.set_xlabel('\u0394AUC vs background (R)', fontsize=PT_AXIS)
fig.text(0.63, 0.99, 'Pooled, post hoc', fontsize=PT_LEG, ha='left', va='top')
fig.text(0.18, 0.99, 'Higher ΔAUC = better discrimination', fontsize=PT_LEG, ha='left', va='top')
clean(ax, left=False, bottom=True)
ax.tick_params(axis='y', length=0)
key = [Rectangle((0, 0), 1, 1, facecolor=CHEAP_LIGHT, edgecolor='none'),
       Rectangle((0, 0), 1, 1, facecolor=CHEAP, edgecolor='none'),
       Rectangle((0, 0), 1, 1, facecolor=COSTLY_LIGHT, edgecolor='none'),
       Rectangle((0, 0), 1, 1, facecolor=COSTLY, edgecolor='none')]
ax.legend(key, ['one question', 'both single questions',
                'mood scale alone', 'all three domains'],
          frameon=False, fontsize=PT_VAL_SM, loc='upper center',
          bbox_to_anchor=(0.40, -0.155), ncol=2, handlelength=0.9,
          handleheight=0.8, columnspacing=0.9, borderpad=0, labelspacing=0.3)
panel_letter(ax, 'b', dx=-0.30, dy=1.02)

header = ['series', 'category', 'estimate', 'lower_95', 'upper_95', 'n']
export(fig, 'fig2_increment', rows, header)
print('fig2 done')
