"""05_transport_calibration.py
Decile calibration curves for the Q strategy under no updating vs logistic
recalibration, computed in leave-one-source-out transport.
Writes results/transport_calibration_deciles.csv.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')
RNG = np.random.default_rng(20261025)
SRCS = ['CHARLS', 'ELSA', 'HRS']
COLS = ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z', 'climb']
BIN = ('male', 'htn', 'dia', 'hrt', 'str', 'climb', 'cesd_z')


def design(df, stats=None):
    X = pd.DataFrame(index=df.index)
    stats = {} if stats is None else stats
    for c in COLS:
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


rows = []
coh = {s: pd.read_parquet(f'data/cohort_{s}.parquet') for s in SRCS}
for test in SRCS:
    tr = pd.concat([coh[s] for s in SRCS if s != test], ignore_index=True).dropna(subset=COLS)
    te = coh[test].dropna(subset=COLS).copy()
    Xtr, st = design(tr)
    Xte, _ = design(te, st)
    ytr = tr['status'].astype(int).values
    yte = te['status'].astype(int).values
    m = LogisticRegression(C=1e6, max_iter=5000, solver='lbfgs').fit(Xtr.values, ytr)
    p = m.predict_proba(Xte.values)
    P = np.zeros((len(te), 3))
    for j, c in enumerate(m.classes_):
        P[:, int(c)] = p[:, j]

    ybin = (yte == 1).astype(int)
    for lab, pp in [('none', P[:, 1].copy())]:
        q = pd.qcut(pd.Series(pp).rank(method='first'), 10, labels=False)
        for g in range(10):
            msk = (q == g).values
            rows.append(dict(test=test, update=lab, decile=g + 1, n=int(msk.sum()),
                             pred=float(pp[msk].mean()), obs=float(ybin[msk].mean())))

    # logistic recalibration: 50/50 split, evaluate on the held-out half, average
    acc = {}
    for _ in range(20):
        idx = RNG.permutation(len(te))
        h = len(te) // 2
        a, b = idx[:h], idx[h:]
        if (yte[a] == 1).sum() < 10 or (yte[b] == 1).sum() < 10:
            continue
        lpa = np.log(np.clip(P[a, 1], 1e-6, 1 - 1e-6) / (1 - np.clip(P[a, 1], 1e-6, 1 - 1e-6)))
        mr = LogisticRegression(C=1e6, max_iter=5000, solver='lbfgs')
        mr.fit(lpa.reshape(-1, 1), (yte[a] == 1).astype(int))
        lpb = np.log(np.clip(P[b, 1], 1e-6, 1 - 1e-6) / (1 - np.clip(P[b, 1], 1e-6, 1 - 1e-6)))
        pb = mr.predict_proba(lpb.reshape(-1, 1))[:, 1]
        qb = pd.qcut(pd.Series(pb).rank(method='first'), 10, labels=False)
        for g in range(10):
            msk = (qb == g).values
            if msk.sum() == 0:
                continue
            acc.setdefault(g, []).append((pb[msk].mean(), (yte[b][msk] == 1).mean(), msk.sum()))
    for g, v in acc.items():
        arr = np.array(v)
        rows.append(dict(test=test, update='logistic_recal', decile=g + 1,
                         n=int(arr[:, 2].mean()),
                         pred=float(arr[:, 0].mean()), obs=float(arr[:, 1].mean())))
    print('done', test, flush=True)

pd.DataFrame(rows).to_csv('results/transport_calibration_deciles.csv', index=False)
print(pd.DataFrame(rows).round(4).to_string(index=False))
