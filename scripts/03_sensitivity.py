"""03_sensitivity.py
Prespecified sensitivity analyses S1-S6. S7 (MICE) registered as not run.
Writes results/sensitivity.csv.
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
RES = 'results'
RNG = np.random.default_rng(20261025)
SRCS = ['CHARLS', 'ELSA', 'HRS']
BIN = ('male', 'htn', 'dia', 'hrt', 'str', 'climb', 'walk', 'fall', 'cesd_z')

STRAT = {
    'R': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str'],
    'Q': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z', 'climb'],
    'E1': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z', 'climb', 'grip'],
    'E2': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z', 'climb', 'memory'],
    'E3': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z', 'climb', 'grip', 'memory'],
}


def design(df, cols, stats=None):
    X = pd.DataFrame(index=df.index)
    stats = {} if stats is None else stats
    for c in cols:
        v = pd.to_numeric(df[c], errors='coerce').astype(float)
        if c in BIN:
            X[c] = v.fillna(0.0)
        elif c == 'educ':
            X['educ_mid'] = (v == 2).astype(float)
            X['educ_high'] = (v == 3).astype(float)
        else:
            if stats is None or c not in stats:
                m, s = v.mean(), v.std(ddof=0)
                s = s if (s and s > 1e-9) else 1.0
                stats[c] = (m, s)
            m, s = stats[c]
            X[c] = (v - m) / s
    return X, stats


def run(df, strategies, outcome='status', tag='', src=''):
    """Fit each strategy on its own complete cases; return AUC/Brier rows."""
    out = []
    for s in strategies:
        cols = STRAT[s]
        sub = df.dropna(subset=cols)
        y = sub[outcome].astype(int).values
        if len(sub) < 200 or (y == 1).sum() < 15:
            out.append(dict(analysis=tag, src=src, strategy=s, n=len(sub),
                            events=int((y == 1).sum()), auc=np.nan, brier=np.nan,
                            note='too few events'))
            continue
        X, _ = design(sub, cols)
        m = LogisticRegression(C=1e6, max_iter=5000, solver='lbfgs')
        m.fit(X.values, y)
        p = m.predict_proba(X.values)
        out.append(dict(analysis=tag, src=src, strategy=s, n=len(sub),
                        events=int((y == 1).sum()),
                        auc=float(roc_auc_score((y == 1).astype(int), p[:, 1])),
                        brier=float(np.mean((p[:, 1] - (y == 1)) ** 2)),
                        note=''))
    return out


def main():
    coh = {s: pd.read_parquet(f'data/cohort_{s}.parquet') for s in SRCS}
    rows = []

    # ---- S0 reference: main 3-state outcome, alive-vs-disabled AUC on p(disabled)
    for s in SRCS:
        d = coh[s].copy()
        d['y2'] = (d['status'] == 1).astype(int)
        for r in run(d, ['R', 'Q', 'E1', 'E2', 'E3'], 'y2', 'S0_main', s):
            rows.append(r)

    # ---- S1: include ages 50-59 (rebuild cohorts without the age filter)
    import subprocess, sys
    for s in SRCS:
        full = pd.read_parquet(f'data/cohort_{s}_allages.parquet') \
            if os.path.exists(f'data/cohort_{s}_allages.parquet') else None
        if full is None:
            continue
        full['y2'] = (full['status'] == 1).astype(int)
        for r in run(full, ['R', 'Q', 'E3'], 'y2', 'S1_age50', s):
            rows.append(r)

    # ---- S2: two-state alive-only outcome (drop deaths), all four sources
    for s in SRCS + ['SHARE']:
        d = pd.read_parquet(f'data/cohort_{s}.parquet')
        d = d[d['status'].isin([0, 1])].copy()
        d['y2'] = (d['status'] == 1).astype(int)
        for r in run(d, ['R', 'Q', 'E1', 'E2', 'E3'], 'y2', 'S2_alive_only', s):
            rows.append(r)

    # ---- S3: Q with walking difficulty; ELSA/SHARE additionally falls
    for s in SRCS:
        d = coh[s].copy()
        d['y2'] = (d['status'] == 1).astype(int)
        STRAT['Q_walk'] = STRAT['Q'] + ['walk']
        for r in run(d, ['Q', 'Q_walk'], 'y2', 'S3_add_walk', s):
            rows.append(r)
    for s in ['ELSA', 'SHARE']:
        d = pd.read_parquet(f'data/cohort_{s}.parquet')
        d['y2'] = (d['status'] == 1).astype(int)
        STRAT['Q_fall'] = STRAT['Q'] + ['fall']
        for r in run(d, ['Q', 'Q_fall'], 'y2', 'S3_add_fall', s):
            rows.append(r)

    # ---- S4: depressive symptoms dichotomised instead of continuous z
    for s in SRCS:
        d = coh[s].copy()
        d['y2'] = (d['status'] == 1).astype(int)
        thr = {'CHARLS': 10, 'ELSA': 4, 'HRS': 4}[s]     # conventional cut-points
        d['cesd_bin'] = (d['cesd_z'] >= thr).astype(float)
        d.loc[d['cesd_z'].isna(), 'cesd_bin'] = np.nan
        STRAT['Q_bin'] = ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str',
                          'shlt', 'cesd_bin', 'climb']
        for r in run(d, ['Q', 'Q_bin'], 'y2', 'S4_cesd_binary', s):
            rows.append(r)

    # ---- S5: HRS outcome defined by adl6a instead of the shared five items
    if os.path.exists('data/cohort_HRS_adl6.parquet'):
        d = pd.read_parquet('data/cohort_HRS_adl6.parquet')
        d['y2'] = (d['status'] == 1).astype(int)
        for r in run(d, ['R', 'Q', 'E3'], 'y2', 'S5_hrs_adl6', 'HRS'):
            rows.append(r)

    # ---- S6: window strata (18-23 vs 24-30 months)
    for s in SRCS:
        d = coh[s].copy()
        d['y2'] = (d['status'] == 1).astype(int)
        for lo, hi, lab in [(18, 23, 'S6_short'), (24, 30, 'S6_long')]:
            sub = d[d['months'].between(lo, hi)]
            for r in run(sub, ['R', 'Q', 'E3'], 'y2', lab, s):
                rows.append(r)

    out = pd.DataFrame(rows)
    out.to_csv(f'{RES}/sensitivity.csv', index=False)
    piv = out.pivot_table(index=['analysis', 'strategy'], columns='src',
                          values='auc', aggfunc='mean')
    print(piv.round(3).to_string())


if __name__ == '__main__':
    main()
