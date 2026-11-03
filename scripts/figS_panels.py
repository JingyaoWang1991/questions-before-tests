"""figS_panels.py — supplementary figures S1 to S4 and S6 panels.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from style_supp import (SRC, GREY, DARK, ALERT, OK, LIGHT, W2, W1, MM, fig,
                           tidy, save, panel_letter)

ORDER = ['CHARLS', 'ELSA', 'HRS', 'SHARE']
LAB = {'CHARLS': 'China', 'ELSA': 'England', 'HRS': 'USA', 'SHARE': 'Europe',
       'POOLED': 'Pooled'}


# ---------------------------------------------------------------- S1
def s1():
    cov = pd.read_csv('results/coverage.csv')
    vars_ = ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd',
             'climb', 'walk', 'fall', 'grip', 'memory', 'p_grip', 'p_memory']
    lab = {'age': 'age', 'male': 'sex', 'educ': 'education', 'htn': 'hypertension',
           'dia': 'diabetes', 'hrt': 'heart disease', 'str': 'stroke',
           'shlt': 'self-rated health', 'cesd': 'depressive symptoms',
           'climb': 'stair climbing', 'walk': 'walking', 'fall': 'falls',
           'grip': 'grip strength', 'memory': 'memory', 'p_grip': 'prior grip',
           'p_memory': 'prior memory'}
    M = cov.pivot(index='var', columns='src', values='pct').reindex(vars_)[ORDER]

    fig, ax = plt.subplots(figsize=(W1 * 1.10, 78 * MM))
    cmap = LinearSegmentedColormap.from_list('avail', ['#f2f2f2', '#9ec5e0', '#1f6fb2'])
    im = ax.imshow(M.values, cmap=cmap, vmin=0, vmax=100, aspect='auto')
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([LAB[s] for s in ORDER], fontsize=6.4)
    ax.set_yticks(range(len(vars_)))
    ax.set_yticklabels([lab[v] for v in vars_], fontsize=6.3)
    for i in range(len(vars_)):
        for j in range(len(ORDER)):
            v = M.values[i, j]
            ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=6.2,
                    color='white' if v > 55 else DARK)
    cb = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.10)
    cb.set_label('available (%)', fontsize=6.3)
    cb.ax.tick_params(labelsize=6.2)
    ax.axhline(10.5, color='white', lw=1.6)
    for y0, y1, tag, col in [(0, 6, 'R', DARK), (7, 9, 'Q', SRC['CHARLS']),
                             (10, 15, 'E / L', SRC['ELSA'])]:
        ax.plot([len(ORDER) - 0.40, len(ORDER) - 0.40], [y0 - 0.4, y1 + 0.4],
                color=col, lw=1.6, clip_on=False, zorder=5)
        ax.text(len(ORDER) - 0.34, (y0 + y1) / 2, tag, fontsize=6.3, color=col,
                va='center', ha='left', clip_on=False, zorder=5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    save(fig, 'figS1_availability')
    plt.close(fig)


# ---------------------------------------------------------------- S2
def s2():
    sb = pd.read_csv('results/subgroup.csv')
    sb = sb[sb['strategy'].isin(['R', 'Q', 'E3'])]
    fig, axes = plt.subplots(1, 3, figsize=(W2, 66 * MM), sharey=True,
                             gridspec_kw={'wspace': 0.10})
    for ax, var in zip(axes, ['age', 'male', 'educ']):
        sub = sb[sb['var'] == var]
        ys = list(dict.fromkeys(sub['group']))
        yy = np.arange(len(ys))
        for st, col in [('R', '#9aa7b1'), ('Q', SRC['CHARLS']), ('E3', SRC['ELSA'])]:
            vals = []
            for g in ys:
                v = sub[(sub['group'] == g) & (sub['strategy'] == st)]
                vals.append(v['auc'].iloc[0] if len(v) else np.nan)
            ax.plot(yy, vals, 'o', ms=3.8, color=col, markeredgecolor='white',
                    markeredgewidth=0.4,
                    label={'R': 'background', 'Q': '+ questionnaire',
                           'E3': '+ grip + memory'}[st])
        ax.set_xticks(yy)
        ax.set_xticklabels(ys, fontsize=6.3)
        ax.set_xlabel({'age': 'age group', 'male': 'sex',
                       'educ': 'education'}[var], fontsize=6.4)
        tidy(ax)
        ax.margins(x=0.12, y=0.08)
    axes[0].set_ylabel('AUC, pooled over cohorts', fontsize=6.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=6.4, loc='lower center',
               ncol=3, bbox_to_anchor=(0.5, -0.10), handlelength=1.0,
               columnspacing=2.2, borderpad=0.1)
    save(fig, 'figS2_subgroups')
    plt.close(fig)


# ---------------------------------------------------------------- S3
def s3():
    se = pd.read_csv('results/sensitivity.csv')
    se = se[se['strategy'].isin(['R', 'Q', 'E3'])]
    an_order = ['S0_main', 'S1_age50', 'S2_alive_only', 'S3_add_walk',
                'S3_add_fall', 'S4_cesd_binary', 'S5_hrs_adl6', 'S6_short', 'S6_long']
    an_lab = {'S0_main': 'S0', 'S1_age50': 'S1', 'S2_alive_only': 'S2',
              'S3_add_walk': 'S3a', 'S3_add_fall': 'S3b',
              'S4_cesd_binary': 'S4', 'S5_hrs_adl6': 'S5',
              'S6_short': 'S6a', 'S6_long': 'S6b'}
    an_order = [a for a in an_order if a in set(se['analysis'])]
    fig, axes = plt.subplots(1, 3, figsize=(W2, 72 * MM), sharey=True,
                             gridspec_kw={'wspace': 0.09})
    for ax, src in zip(axes, ['CHARLS', 'ELSA', 'HRS']):
        sub = se[se['src'] == src]
        xs = np.arange(len(an_order))
        for st, col in [('R', '#9aa7b1'), ('Q', SRC['CHARLS']), ('E3', SRC['ELSA'])]:
            vals = []
            for a in an_order:
                v = sub[(sub['analysis'] == a) & (sub['strategy'] == st)]
                vals.append(v['auc'].iloc[0] if len(v) else np.nan)
            # marker-only: an analysis that was not run leaves a real gap, and a
            # connecting line across the gap would imply interpolation. Cells that
            # are structurally absent (the variable does not exist in that survey)
            # are marked n.d. so the gap cannot be read as a zero.
            ax.plot(xs, vals, 'o', ms=3.8, color=col,
                    markeredgecolor='white', markeredgewidth=0.4,
                    label={'R': 'background', 'Q': '+ questionnaire',
                           'E3': '+ grip + memory'}[st])
            if st == 'Q':
                # structural gaps (the analysis does not apply to that cohort) are
                # marked in axes coordinates so a gap cannot be read as a value of zero
                gaps = [i for i, v in enumerate(vals) if v != v]
                for gi in gaps:
                    ax.text(xs[gi], 0.045, 'n.d.', transform=ax.get_xaxis_transform(),
                            fontsize=6.0, color='#8a8a8a', ha='center', va='bottom')
        ax.set_xticks(xs)
        ax.set_xticklabels([an_lab[a] for a in an_order], fontsize=6.3)
        ax.set_title(LAB[src], loc='left', fontsize=6.6, color=SRC[src])
        ax.set_xlabel('analysis', fontsize=6.4)
        tidy(ax)
        ax.margins(x=0.05, y=0.08)
    axes[0].set_ylabel('AUC for two-year ADL difficulty', fontsize=6.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=6.4, loc='lower center',
               ncol=3, bbox_to_anchor=(0.5, -0.10), handlelength=1.0,
               columnspacing=2.2, borderpad=0.1)
    save(fig, 'figS3_sensitivity')
    plt.close(fig)


# ---------------------------------------------------------------- S4
def s4():
    d1 = pd.read_csv('results/paired_delta_b1000.csv')
    d2 = pd.read_csv('results/temporal_replication.csv')
    CMP = ['R->Q', 'Q->E1', 'Q->E2', 'Q->E3', 'E3->L']
    CLAB = {'R->Q': 'Q', 'Q->E1': '+grip', 'Q->E2': '+memory',
            'Q->E3': '+both', 'E3->L': '+prior wave'}
    d1 = d1[d1['comparison'].isin(CMP)]
    d1 = d1[d1['block'] == 'POOLED'].set_index('comparison')
    # the second-pair file also carries the stepped Q->E1/E2/E3 decomposition;
    # keep one row per comparison by matching the pair-set name
    PICK = {'R->Q': 'R_vs_Q', 'Q->E1': 'Q_vs_E1', 'Q->E2': 'Q_vs_E2',
            'Q->E3': 'Q_vs_E3', 'E3->L': 'E3_vs_L'}
    d2 = d2[d2['block'] == 'POOLED_2nd']
    d2 = d2[d2.apply(lambda r: PICK.get(r['comparison']) == r['set'], axis=1)]
    d2 = d2.set_index('comparison')

    fig, ax = plt.subplots(figsize=(W1 * 1.42, 62 * MM))
    ys = np.arange(len(CMP))[::-1]
    for i, c in enumerate(CMP):
        y = ys[i]
        a = d1.loc[c] if c in d1.index else None
        b = d2.loc[c] if c in d2.index else None
        if a is not None:
            ax.plot([a['dauc_lo'], a['dauc_hi']], [y + 0.11, y + 0.11], color=OK,
                    lw=1.0, solid_capstyle='round', zorder=2)
            ax.plot(a['dauc'], y + 0.11, 'o', ms=4.0, color=OK, zorder=3,
                    markeredgecolor='white', markeredgewidth=0.5)
        if b is not None:
            ax.plot([b['dauc_lo'], b['dauc_hi']], [y - 0.11, y - 0.11], color=ALERT,
                    lw=1.0, solid_capstyle='round', zorder=2)
            ax.plot(b['dauc'], y - 0.11, 's', ms=3.6, color=ALERT, zorder=3,
                    markeredgecolor='white', markeredgewidth=0.5)
        if a is not None and b is not None:
            ax.plot([a['dauc'], b['dauc']], [y + 0.11, y - 0.11], color='#cccccc',
                    lw=0.6, zorder=1)
    ax.axvline(0, color=GREY, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([CLAB[c] for c in CMP], fontsize=6.4)
    ax.set_xlabel('\u0394AUC (95% CI), pooled', fontsize=6.6)
    ax.set_xlim(-0.006, 0.095)
    ax.plot([], [], 'o', ms=4, color=OK, label='first baseline pair')
    ax.plot([], [], 's', ms=3.6, color=ALERT, label='second baseline pair')
    ax.legend(frameon=False, fontsize=6.2, loc='upper center', ncol=2,
              bbox_to_anchor=(0.5, -0.24), handlelength=1.0, borderpad=0.1,
              columnspacing=1.8)
    tidy(ax)
    ax.margins(x=0.02, y=0.10)
    save(fig, 'figS4_replication')
    plt.close(fig)


# ---------------------------------------------------------------- S5
def s5():
    co = pd.read_csv('results/coefficients.csv')
    co = co[co['strategy'] == 'Q']
    have = [b for b in ORDER if b in set(co['block'])]
    co = co[co['block'].isin(have)]
    M = co.pivot_table(index='term', columns='block', values='beta')
    M = M[have]
    keep = ['age', 'male', 'educ_mid', 'educ_high', 'htn', 'dia', 'hrt', 'str',
            'shlt', 'cesd_z', 'climb']
    keep = [k for k in keep if k in M.index]
    M = M.loc[keep]
    lab = {'age': 'age', 'male': 'male sex', 'educ_mid': 'education (mid)',
           'educ_high': 'education (high)', 'htn': 'hypertension', 'dia': 'diabetes',
           'hrt': 'heart disease', 'str': 'stroke', 'shlt': 'self-rated health',
           'cesd_z': 'depressive symptoms', 'climb': 'stair climbing'}

    fig, ax = plt.subplots(figsize=(W1 * 1.10, 64 * MM))
    vmax = np.abs(M.values).max()
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list('div',
                                             ['#c2571a', '#f5f5f5', '#1f6fb2'])
    im = ax.imshow(M.values, cmap=cmap, norm=norm, aspect='auto')
    ax.set_xticks(range(len(have)))
    ax.set_xticklabels([LAB[s] for s in have], fontsize=6.4)
    ax.set_yticks(range(len(keep)))
    ax.set_yticklabels([lab[k] for k in keep], fontsize=6.3)
    for i in range(len(keep)):
        for j in range(len(have)):
            v = M.values[i, j]
            ax.text(j, i, f'{v:+.2f}', ha='center', va='center', fontsize=6.2,
                    color='white' if abs(v) > vmax * 0.55 else DARK)
    cb = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02)
    cb.set_label('log-odds coefficient', fontsize=6.3)
    cb.ax.tick_params(labelsize=6.2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    save(fig, 'figS5_coefficients')
    plt.close(fig)


# ---------------------------------------------------------------- S6
def s6():
    sel = pd.read_csv('results/selection_diagnostic.csv')
    sel = sel[sel['strategy'].isin(['Q'])]
    fig, ax = plt.subplots(figsize=(W1 * 1.05, 58 * MM))
    subs = ['grip measured', 'grip not measured']
    for s in ['CHARLS', 'ELSA', 'HRS']:
        vals = []
        for sub in subs:
            r = sel[(sel['src'] == s) & (sel['subset'] == sub)]
            vals.append(r['auc'].iloc[0] if len(r) else np.nan)
        ax.plot([0, 1], vals, 'o-', ms=4.4, lw=1.4, color=SRC[s],
                markeredgecolor='white', markeredgewidth=0.5,
                label=LAB[s], zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['could complete\nthe grip test',
                        'could not complete\nthe grip test'], fontsize=6.3)
    ax.set_xlim(-0.30, 1.30)
    ax.set_ylabel('AUC of the questionnaire model', fontsize=6.6)
    ax.legend(frameon=False, fontsize=6.2, loc='upper left', handlelength=1.2,
              borderpad=0.2, labelspacing=0.35)
    tidy(ax)
    ax.margins(x=0.04, y=0.20)
    save(fig, 'figS6_selection')
    plt.close(fig)


if __name__ == '__main__':
    s1(); s2(); s3(); s4(); s5(); s6()
    print('supplementary figures written')
