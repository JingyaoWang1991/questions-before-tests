"""13_paired_bootstrap.py — paired differences at B = 1,000.
Percentile intervals for every tier comparison, refitting both models inside each
bootstrap resample. Writes results/paired_delta_b1000.csv.
"""
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RES = 'results'
SRCS = ['CHARLS', 'ELSA', 'HRS']
NBOOT = 1000
SEED = 20261025          # identical to scripts/01_main_analysis.py

STRAT = {
    'R':  ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str'],
    'Q':  ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z', 'climb'],
    'E1': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z', 'climb', 'grip'],
    'E2': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z', 'climb', 'memory'],
    'E3': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z', 'climb',
           'grip', 'memory'],
    'L':  ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z', 'climb',
           'grip', 'memory', 'p_grip', 'p_memory'],
}
SETS = [
    ('R_vs_Q',  ['R', 'Q'],  'Q'),
    ('Q_vs_E1', ['Q', 'E1'], 'E1'),
    ('Q_vs_E2', ['Q', 'E2'], 'E2'),
    ('Q_vs_E3', ['Q', 'E3'], 'E3'),
    ('E3_vs_L', ['E3', 'L'], 'L'),
]


def design(df, cols, stats=None):
    X = pd.DataFrame(index=df.index)
    stats = {} if stats is None else stats
    for c in cols:
        v = pd.to_numeric(df[c], errors='coerce').astype(float)
        if c in ('male', 'htn', 'dia', 'hrt', 'str', 'climb', 'cesd_z'):
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


def fit(X, y):
    return LogisticRegression(C=1e6, max_iter=5000, solver='lbfgs').fit(X.values, y)


def proba(m, X):
    p = m.predict_proba(X.values)
    out = np.zeros((len(X), 3))
    for j, cls in enumerate(m.classes_):
        out[:, int(cls)] = p[:, j]
    return out


def brier(P, y):
    return float(np.mean(np.sum((P - np.eye(3)[y]) ** 2, axis=1)))


def block(df, label, nboot=NBOOT):
    rows = []
    for setname, models, need in SETS:
        cols = sorted({c for m in models for c in STRAT[m]})
        sub = df.copy()
        for c in cols:
            sub = sub[sub[c].notna()]
        y = sub['status'].astype(int).values
        if len(sub) < 200 or (y == 1).sum() < 20 or (y == 2).sum() < 10:
            rows.append(dict(block=label, comparison=f'{models[0]}->{models[1]}',
                             n=len(sub), note='not estimable'))
            continue
        Xs = {m: design(sub, STRAT[m])[0] for m in models}
        P = {m: proba(fit(Xs[m], y), Xs[m]) for m in models}
        ybin = (y == 1).astype(int)
        a, b = models
        auc_a, auc_b = roc_auc_score(ybin, P[a][:, 1]), roc_auc_score(ybin, P[b][:, 1])
        br_a, br_b = brier(P[a], y), brier(P[b], y)

        rng = np.random.default_rng(SEED)
        da, db = [], []
        t0 = time.time()
        for _ in range(nboot):
            idx = rng.integers(0, len(sub), len(sub))
            ys = y[idx]
            yy = (ys == 1).astype(int)
            if yy.sum() < 20 or (ys == 2).sum() < 10:
                continue
            Xb = {m: design(sub.iloc[idx], STRAT[m])[0] for m in models}
            try:
                Pa = proba(fit(Xb[a], ys), Xb[a])
                Pb = proba(fit(Xb[b], ys), Xb[b])
                da.append(roc_auc_score(yy, Pb[:, 1]) - roc_auc_score(yy, Pa[:, 1]))
                db.append(brier(Pb, ys) - brier(Pa, ys))
            except Exception:
                continue
        da, db = np.array(da), np.array(db)
        rows.append(dict(
            block=label, comparison=f'{a}->{b}', n=len(sub),
            events_dis=int((y == 1).sum()), events_dead=int((y == 2).sum()),
            auc_a=auc_a, auc_b=auc_b, dauc=auc_b - auc_a,
            dauc_lo=np.percentile(da, 2.5), dauc_hi=np.percentile(da, 97.5),
            brier_a=br_a, brier_b=br_b, dbrier=br_b - br_a,
            dbrier_lo=np.percentile(db, 2.5), dbrier_hi=np.percentile(db, 97.5),
            nboot_ok=len(da), seconds=round(time.time() - t0, 1)))
        print(f'{label} {a}->{b}: {len(da)} resamples in {time.time()-t0:.0f}s',
              flush=True)
    return rows


def main():
    coh = {s: pd.read_parquet(f'data/cohort_{s}.parquet') for s in SRCS}
    for s in SRCS:
        coh[s]['src'] = s
    all_rows = []
    for s in SRCS:
        all_rows += block(coh[s], s)
    pooled = pd.concat(coh.values(), ignore_index=True)
    all_rows += block(pooled, 'POOLED')
    out = pd.DataFrame(all_rows)
    out.to_csv(f'{RES}/paired_delta_b1000.csv', index=False)
    print(out.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
