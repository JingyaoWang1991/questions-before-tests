"""08_transport_deciles.py
Decile calibration curves for every model (R, Q, E3) in every receiving cohort,
under no local update and after logistic recalibration.
Writes results/transport_deciles_all.csv (long format).
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')
RNG = np.random.default_rng(20261025)
SRCS = ['CHARLS', 'ELSA', 'HRS']
BIN = ('male', 'htn', 'dia', 'hrt', 'str', 'climb', 'cesd_z')
STRAT = {
    'R': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str'],
    'Q': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z', 'climb'],
    'E3': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z',
           'climb', 'grip', 'memory'],
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
            if c not in stats:
                m, s = v.mean(), v.std(ddof=0)
                stats[c] = (m, s if s > 1e-9 else 1.0)
            m, s = stats[c]
            X[c] = (v - m) / s
    return X, stats


coh = {s: pd.read_parquet(f'data/cohort_{s}.parquet') for s in SRCS}
rows = []
for test in SRCS:
    tr = pd.concat([coh[s] for s in SRCS if s != test], ignore_index=True)
    te = coh[test].copy()
    for strat, cols in STRAT.items():
        trc = tr.dropna(subset=cols)
        tec = te.dropna(subset=cols).reset_index(drop=True)
        Xtr, st = design(trc, cols)
        Xte, _ = design(tec, cols, st)
        ytr = trc['status'].astype(int).values
        yte = (tec['status'].astype(int) == 1).astype(int).values
        m = LogisticRegression(C=1e6, max_iter=5000, solver='lbfgs').fit(Xtr.values, ytr)
        p = m.predict_proba(Xte.values)[:, 1]

        # no local update
        q = pd.qcut(pd.Series(p).rank(method='first'), 10, labels=False)
        for g in range(10):
            msk = (q == g).values
            rows.append(dict(test=test, strategy=strat, update='none',
                             decile=g + 1, n=int(msk.sum()),
                             pred=float(p[msk].mean()), obs=float(yte[msk].mean())))

        # logistic recalibration, estimated on half, evaluated on the other half
        acc = {}
        for _ in range(20):
            idx = RNG.permutation(len(tec))
            h = len(tec) // 2
            a, b = idx[:h], idx[h:]
            if yte[a].sum() < 10 or yte[b].sum() < 10:
                continue
            lpa = np.log(np.clip(p[a], 1e-6, 1 - 1e-6) / (1 - np.clip(p[a], 1e-6, 1 - 1e-6)))
            mr = LogisticRegression(C=1e6, max_iter=4000).fit(lpa.reshape(-1, 1), yte[a])
            lpb = np.log(np.clip(p[b], 1e-6, 1 - 1e-6) / (1 - np.clip(p[b], 1e-6, 1 - 1e-6)))
            pb = mr.predict_proba(lpb.reshape(-1, 1))[:, 1]
            qb = pd.qcut(pd.Series(pb).rank(method='first'), 10, labels=False)
            for g in range(10):
                msk = (qb == g).values
                if msk.sum() == 0:
                    continue
                acc.setdefault(g, []).append((pb[msk].mean(), yte[b][msk].mean(), msk.sum()))
        for g, v in acc.items():
            arr = np.array(v)
            rows.append(dict(test=test, strategy=strat, update='logistic_recal',
                             decile=g + 1, n=int(arr[:, 2].mean()),
                             pred=float(arr[:, 0].mean()), obs=float(arr[:, 1].mean())))
    print('done', test, flush=True)

out = pd.DataFrame(rows)
out.to_csv('results/transport_deciles_all.csv', index=False)
print(out.groupby(['test', 'strategy', 'update']).size().to_string())
