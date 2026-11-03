"""figS7_bounds.py — Figure S7: bounds under alternative assignments of unknown states.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from style_supp import SRC, GREY, DARK, ALERT, OK, W1, MM, fig, tidy, save

b = pd.read_csv('results/unknown_bounds.csv')
s = pd.read_csv('results/unknown_bounds_summary.csv')
LAB = {'CHARLS': 'China', 'ELSA': 'England', 'HRS': 'USA'}
CMP = [('R->Q', 'R \u2192 Q  (+ 3 questions)'), ('Q->E3', 'Q \u2192 Q+grip+memory')]

fig, axes = plt.subplots(1, 2, figsize=(W1 * 1.55, 56 * MM),
                         gridspec_kw={'wspace': 0.34})

for ax, (cmp_, title) in zip(axes, CMP):
    sub = s[s['comparison'] == cmp_]
    ys = np.arange(len(sub))[::-1]
    for y, (_, r) in zip(ys, sub.iterrows()):
        col = SRC[r['src']]
        ax.plot([r['dauc_min'], r['dauc_max']], [y, y], color=col, lw=2.6,
                alpha=0.30, solid_capstyle='round', zorder=2)
        ax.plot(r['dauc_cc'], y, 'o', ms=4.4, color=col, zorder=3,
                markeredgecolor='white', markeredgewidth=0.5)
        ax.text(r['dauc_cc'], y + 0.26, f"{r['dauc_cc']:+.3f}", fontsize=6,
                ha='center', va='bottom', color=col)
    ax.axvline(0, color=GREY, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([LAB[x] for x in sub['src']], fontsize=6.0)
    ax.set_xlabel('\u0394AUC')
    ax.set_title(title, loc='left', fontsize=6.8)
    ax.set_ylim(-0.6, len(sub) - 0.25)
    tidy(ax)
    ax.margins(x=0.10, y=0.06)

axes[0].plot([], [], 'o', ms=4.4, color=DARK, label='complete case')
axes[0].plot([], [], '-', lw=2.6, color=DARK, alpha=0.30,
             label='range over unknown-outcome assignments')
axes[0].legend(frameon=False, fontsize=6, loc='lower right', handlelength=1.4,
               borderpad=0.2, labelspacing=0.3)

fig.suptitle('Assigning every unknown outcome to the least favourable state\n'
             'does not reverse the questionnaire gain',
             x=0.02, ha='left', fontsize=7.2, y=1.10)
save(fig, 'figS7_bounds')
print('figS7 written')
