"""09_unknown_outcome_bounds.py
Full-denominator bounds for the paired comparison under unknown outcome.

The main analysis is complete-case: participants whose 24-month state could not
be determined are dropped. This script keeps them and asks how much the
conclusion could move if they were all assigned the same outcome, i.e. a
worst/best-case bound rather than an assumption-free estimate.

For each paired comparison (R->Q, Q->E3) and each source we refit both models on
the full eligible denominator, with the unknown participants assigned:
  all independent / all disabled / all died
and report the paired dAUC and dBrier under each assignment alongside the
complete-case estimate.

Writes results/unknown_bounds.csv and results/unknown_bounds_summary.csv.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
SRCS = ['CHARLS', 'ELSA', 'HRS']
BIN = ('male', 'htn', 'dia', 'hrt', 'str', 'climb', 'cesd_z')
STRAT = {
    'R': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str'],
    'Q': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z', 'climb'],
    'E3': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z',
           'climb', 'grip', 'memory'],
}
CMP = [('R', 'Q'), ('Q', 'E3')]


def design(df, cols):
    X = pd.DataFrame(index=df.index)
    stats = {}
    for c in cols:
        v = pd.to_numeric(df[c], errors='coerce').astype(float)
        if c in BIN:
            X[c] = v.fillna(0.0)
        elif c == 'educ':
            X['educ_mid'] = (v == 2).astype(float)
            X['educ_high'] = (v == 3).astype(float)
        else:
            m, s = v.mean(), v.std(ddof=0)
            X[c] = (v - m) / (s if s > 1e-9 else 1.0)
    return X


def paired(df, a, b, y):
    cols = sorted(set(STRAT[a]) | set(STRAT[b]))
    Xa, Xb = design(df, STRAT[a]), design(df, STRAT[b])
    ya = (y == 1).astype(int)
    if ya.sum() < 15 or ya.sum() == len(ya):
        return None
    ma = LogisticRegression(C=1e6, max_iter=4000).fit(Xa.values, y)
    mb = LogisticRegression(C=1e6, max_iter=4000).fit(Xb.values, y)
    pa = ma.predict_proba(Xa.values)[:, 1]
    pb = mb.predict_proba(Xb.values)[:, 1]
    return dict(n=len(df), events=int(ya.sum()),
                auc_a=roc_auc_score(ya, pa), auc_b=roc_auc_score(ya, pb),
                brier_a=float(np.mean((pa - ya) ** 2)),
                brier_b=float(np.mean((pb - ya) ** 2)))


rows = []
for src in SRCS:
    d = pd.read_parquet(f'data/cohort_{src}.parquet')
    unk = pd.read_parquet(f'data/cohort_{src}_unknown.parquet')
    n_unk = len(unk)

    for a, b in CMP:
        cols = sorted(set(STRAT[a]) | set(STRAT[b]))
        # complete case: both models on the same analysable rows
        cc = d.dropna(subset=cols)
        y_cc = (cc['status'].astype(int) == 1).astype(int).values
        r = paired(cc, a, b, y_cc)
        if r:
            rows.append(dict(src=src, comparison=f'{a}->{b}', scenario='complete case',
                             n=r['n'], events=r['events'],
                             dauc=r['auc_b'] - r['auc_a'],
                             dbrier=r['brier_b'] - r['brier_a'],
                             n_unknown_excluded=n_unk))

        # bounds: add the unknown participants back, assign one outcome to all
        # The endpoint is binary (disabled vs not), so "all independent" and
        # "all died" map to the same value y=0. Only two distinct assignments exist.
        for scen, yv in [('unknown all not disabled', 0), ('unknown all disabled', 1)]:
            aug = pd.concat([d[cols + ['status']], unk[cols + ['status']]],
                            ignore_index=True)
            aug = aug.dropna(subset=cols)
            y = (aug['status'].astype(int) == 1).astype(int).values.copy()
            is_unk = aug.index >= len(d)
            y[is_unk] = 1 if yv == 1 else 0
            r = paired(aug, a, b, y)
            if r:
                rows.append(dict(src=src, comparison=f'{a}->{b}', scenario=scen,
                                 n=r['n'], events=r['events'],
                                 dauc=r['auc_b'] - r['auc_a'],
                                 dbrier=r['brier_b'] - r['brier_a'],
                                 n_unknown_excluded=n_unk))
    print('done', src, flush=True)

out = pd.DataFrame(rows)
out.to_csv('results/unknown_bounds.csv', index=False)

summ = []
for (src, cmp_), g in out.groupby(['src', 'comparison']):
    cc = g[g['scenario'] == 'complete case']
    sc = g[g['scenario'] != 'complete case']
    if cc.empty or sc.empty:
        continue
    summ.append(dict(
        src=src, comparison=cmp_, n_unknown=int(cc.iloc[0]['n_unknown_excluded']),
        dauc_cc=cc.iloc[0]['dauc'],
        dauc_min=sc['dauc'].min(), dauc_max=sc['dauc'].max(),
        dbrier_cc=cc.iloc[0]['dbrier'],
        dbrier_min=sc['dbrier'].min(), dbrier_max=sc['dbrier'].max()))
s = pd.DataFrame(summ)
s.to_csv('results/unknown_bounds_summary.csv', index=False)

pd.set_option('display.width', 220)
print(out.round(4).to_string(index=False))
print()
print(s.round(4).to_string(index=False))
print()
print('sign of dAUC preserved in all scenarios:',
      bool(((s['dauc_min'] > 0) == (s['dauc_cc'] > 0)).all()))
