"""fig1_flow.py — Figure 1: participant flow and target-interview states.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from style_main import (INK, INK2, MUTED, HAIRLINE, SLATE, PALE, GAIN, MM,
                       COH, PT_TITLE, PT_ROW, PT_AXIS, PT_TICK, PT_LEG, PT_VAL,
                       PT_VAL_SM, canvas, clean, panel_letter, export,
                       shade, tint, label_colour)

f = pd.read_csv('results/flow_prisma.csv').set_index('src')
ORDER = ['CHARLS', 'ELSA', 'HRS', 'SHARE']
THREE = ['CHARLS', 'ELSA', 'HRS']
LAB = {'CHARLS': 'China', 'ELSA': 'England', 'HRS': 'USA', 'SHARE': 'Europe'}

records = int(f['records_in_file'].sum())
interviewed = int(f['interviewed_baseline'].sum())
adl_free = int(f['adl_independent'].sum())
aged60 = int(f['aged_60plus'].sum())
ascertained = int(f['analysed'].sum())
ex = [int(f['excluded_not_interviewed'].sum()), int(f['excluded_adl'].sum()),
      int(f['excluded_under60'].sum()), int(f['excluded_no_state'].sum())]
ex_lab = ['not interviewed', 'reported ADL difficulty', 'aged under 60',
          'no 24-month state']

assert records - ex[0] == interviewed
assert interviewed - ex[1] == adl_free
assert adl_free - ex[2] == aged60
assert aged60 - ex[3] == ascertained
n_three = int(f.loc[THREE, 'analysed'].sum())
n_alive = int(f.loc['SHARE', 'analysed'])
assert n_three + n_alive == ascertained

fig = canvas(170, 112)
gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.10,
                      left=0.012, right=0.985, top=0.955, bottom=0.205)

# ============================================================== panel a
ax = fig.add_subplot(gs[0, 0])
ax.set_axis_off()
ax.set_xlim(0, 103)
ax.set_ylim(3, 103)

NODE_W, NODE_H = 50.0, 9.0
NX = 2.0                       # left edge of the spine, same for every node
NY = [100.0, 85.0, 70.0, 55.0, 40.0]     # top edge of each node
STEPS = [(records, 'records in the four surveys'),
         (interviewed, 'interviewed at baseline'),
         (adl_free, 'no difficulty in any of five ADLs'),
         (aged60, 'aged 60 or over'),
         (ascertained, '24-month state ascertainable')]
ARROW_X = NX + NODE_W + 2.0     # every exclusion arrow starts here
EXCL_X = ARROW_X + 6.5          # every exclusion label is left-aligned here

rows = []
for i, (val, lab_) in enumerate(STEPS):
    y = NY[i]
    ax.add_patch(Rectangle((NX, y - NODE_H), NODE_W, NODE_H, facecolor='white',
                           edgecolor=HAIRLINE, linewidth=0.8, zorder=2))
    ax.text(NX + 2.2, y - NODE_H * 0.34, f'{val:,}', fontsize=PT_TITLE,
            fontweight='bold', color=INK, ha='left', va='center', zorder=3)
    ax.text(NX + 2.2, y - NODE_H * 0.74, lab_, fontsize=PT_ROW, color=INK2,
            ha='left', va='center', zorder=3)
    rows.append((lab_, val))
    if i < len(STEPS) - 1:
        y_top, y_bot = y - NODE_H, NY[i + 1]
        ax.add_patch(FancyArrowPatch((NX + NODE_W / 2, y_top),
                                     (NX + NODE_W / 2, y_bot),
                                     arrowstyle='-|>', mutation_scale=7,
                                     color='#8C8C8C', linewidth=0.8,
                                     shrinkA=0, shrinkB=0, zorder=1))
        ymid = (y_top + y_bot) / 2
        ax.add_patch(FancyArrowPatch((ARROW_X, ymid), (EXCL_X - 1.0, ymid),
                                     arrowstyle='-|>', mutation_scale=6,
                                     color='#8C8C8C', linewidth=0.8,
                                     shrinkA=0, shrinkB=0, zorder=1))
        ax.text(EXCL_X, ymid, f'{ex[i]:,}  {ex_lab[i]}', fontsize=PT_LEG,
                color=MUTED, ha='left', va='center')
        rows.append((f'excluded: {ex_lab[i]}', ex[i]))

# ---- fork to the two result boxes
y_last = NY[-1] - NODE_H
FORK_Y = y_last - 5.0
XL, XR = 2.0, 52.0
BOX_W, BOX_H = 46.0, 12.0
BOX_Y = FORK_Y - 3.0
ax.plot([NX + NODE_W / 2, NX + NODE_W / 2], [y_last, FORK_Y], color='#8C8C8C',
        lw=0.8, zorder=1)
ax.plot([XL + BOX_W / 2, XR + BOX_W / 2], [FORK_Y, FORK_Y], color='#8C8C8C',
        lw=0.8, zorder=1)
for xb in (XL + BOX_W / 2, XR + BOX_W / 2):
    ax.add_patch(FancyArrowPatch((xb, FORK_Y), (xb, BOX_Y), arrowstyle='-|>',
                                 mutation_scale=7, color='#8C8C8C',
                                 linewidth=0.8, shrinkA=0, shrinkB=0, zorder=1))
for x, n, name, col in [(XL, n_three, 'three-state analysis', INK),
                        (XR, n_alive, 'alive-only analysis', GAIN)]:
    ax.add_patch(Rectangle((x, BOX_Y - BOX_H), BOX_W, BOX_H, facecolor=col,
                           edgecolor=col, linewidth=0.8, zorder=2))
    ax.text(x + BOX_W / 2, BOX_Y - BOX_H * 0.36, f'{n:,}', fontsize=PT_TITLE,
            fontweight='bold', color='white', ha='center', va='center', zorder=3)
    ax.text(x + BOX_W / 2, BOX_Y - BOX_H * 0.74, name, fontsize=PT_ROW,
            color='white', ha='center', va='center', zorder=3)
rows.append(('three-state analysis', n_three))
rows.append(('alive-only analysis', n_alive))
paired = pd.read_csv('results/paired_delta_b1000.csv')
q_n = int(paired.loc[(paired['block'] == 'POOLED') &
                    (paired['comparison'] == 'R->Q'), 'n'].iloc[0])
ax.text(XL + BOX_W / 2, BOX_Y - BOX_H - 3.0,
        f'Questionnaire comparison: n = {q_n:,}',
        fontsize=PT_VAL_SM, color=INK2, ha='center', va='center')
rows.append(('questionnaire complete-case comparison', q_n))
panel_letter(ax, 'a', dx=-0.02, dy=0.985)

# ============================================================== panel b
ax = fig.add_subplot(gs[0, 1])
ypos = np.arange(len(ORDER))[::-1]
dis = np.array([100 * f.loc[s, 'disabled'] / f.loc[s, 'analysed'] for s in ORDER])
ded = np.array([100 * f.loc[s, 'dead'] / f.loc[s, 'analysed'] for s in ORDER])
H = 0.55

for y, d1, d2, s in zip(ypos, dis, ded, ORDER):
    # Hue encodes the cohort; depth encodes the outcome: the mid fill is new ADL
    # difficulty, the darker shade of the same hue is death. Europe keeps a
    # neutral hatch because death was not recorded there.
    base = '#9E9E9E' if s == 'SHARE' else COH[s]
    dark = shade(base, 0.45)
    if s == 'SHARE':
        ax.barh(y, d1, H, color=base, edgecolor='white', linewidth=0.8,
                hatch='///', zorder=2)
    else:
        ax.barh(y, d1, H, color=base, edgecolor='white', linewidth=0.25,
                zorder=2)
        if d2 > 0:
            ax.barh(y, d2, H, left=d1, color=dark, edgecolor='white',
                    linewidth=0.25, zorder=2)
    if d1 >= 4:
        ax.text(d1 / 2, y, f'{d1:.1f}', fontsize=PT_VAL, fontweight='bold',
                color=label_colour(base), ha='center', va='center', zorder=4)
    if d2 > 0:
        ax.text(d1 + d2 / 2, y, f'{d2:.1f}', fontsize=PT_VAL, fontweight='bold',
                color=label_colour(dark), ha='center', va='center', zorder=4)
    if s == 'SHARE':
        ax.text(d1 + 0.45, y, 'death not recorded', fontsize=PT_VAL_SM,
                color=MUTED, ha='left', va='center')
    rows.append((f'{LAB[s]} new ADL difficulty %', round(float(d1), 1)))
    rows.append((f'{LAB[s]} died %', None if s == 'SHARE' else round(float(d2), 1)))
    rows.append((f'{LAB[s]} ADL-difficulty fill', base))
    rows.append((f'{LAB[s]} died fill', None if s == 'SHARE' else dark))

ax.set_yticks(ypos)
ax.set_yticklabels([LAB[s] for s in ORDER], fontsize=PT_TICK)
ax.set_ylim(-0.62, len(ORDER) - 0.38)
ax.set_xlim(0, 25)
ax.set_xticks([0, 5, 10, 15, 20])
ax.set_xlabel('share of the analysed cohort (%)', fontsize=PT_AXIS)
clean(ax, left=False, bottom=True)
ax.tick_params(axis='y', length=0, pad=7)

# The legend explains the depth axis only; hue is read off the cohort row labels.
DEMO = '#7A7A7A'
handles = [Rectangle((0, 0), 1, 1, facecolor=DEMO, edgecolor='white'),
           Rectangle((0, 0), 1, 1, facecolor=shade(DEMO, 0.45), edgecolor='white')]
ax.legend(handles, ['new ADL difficulty (lighter)', 'died (darker)'],
          frameon=False, fontsize=PT_LEG, loc='upper center',
          bbox_to_anchor=(0.5, -0.135), ncol=2, handlelength=0.7,
          handleheight=0.7, columnspacing=1.6, borderpad=0, labelspacing=0)
panel_letter(ax, 'b', dx=-0.16, dy=0.985)

header = ['item', 'value']
export(fig, 'fig1_flow', rows, header)
print('fig1 done')
