"""17_calibration_and_updating.py — calibration of the questionnaire model
under external transport, and paired evaluation of local updating.

Calibration-in-the-large is the intercept of a calibration model with slope fixed at 1,
that is the a solving mean(expit(a + logit(p))) = mean(y), found by one-dimensional root
finding. Local updating is fitted on one random half of the receiving cohort and scored
on the other half, twenty times, with the unchanged model scored on the same test rows.
Writes results/transport_q.csv, results/transport_q_ci.csv, results/local_updates.csv
and results/local_update_paired.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DATA = Path('data')
RES = Path('results')
RES.mkdir(exist_ok=True)

SRCS = ['CHARLS', 'ELSA', 'HRS']
BIN = ('male', 'htn', 'dia', 'hrt', 'str', 'climb', 'cesd_z')
STRAT = {
    'R': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str'],
    'Q': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z', 'climb'],
    'E3': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z',
           'climb', 'grip', 'memory'],
}
EPS = 1e-12
N_BOOT = 500
SPLIT_SEEDS = list(range(20261103, 20261123))


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


def lp(p):
    return logit(np.clip(p, EPS, 1 - EPS))


def offset_intercept(p, ybin):
    """Fixed-slope-1 intercept: the a with mean(expit(a + logit(p))) = mean(y)."""
    z = lp(p)
    target = float(ybin.mean())
    if target <= 0 or target >= 1:
        return np.nan
    return float(brentq(lambda a: float(np.mean(expit(z + a))) - target,
                        -50.0, 50.0, xtol=1e-12))


def logistic_calibration(p, ybin):
    """Free intercept and slope of a binary logistic calibration model.

    Fitted with scikit-learn's lbfgs on an essentially unpenalised model
    (C = 1e6), the same estimator used in the rest of the pipeline, so that the
    reported slope matches the analysis tables to the printed precision.
    """
    z = lp(p)
    m = LogisticRegression(C=1e6, max_iter=5000, solver='lbfgs')
    m.fit(np.column_stack([np.ones(len(z)), z]), ybin)
    return float(m.intercept_[0]), float(m.coef_[0][1])


def fit(X, y):
    return LogisticRegression(C=1e6, max_iter=5000, solver='lbfgs').fit(X.values, y)


def proba(m, X):
    p = m.predict_proba(X.values)
    out = np.zeros((len(X), 3))
    for j, cls in enumerate(m.classes_):
        out[:, int(cls)] = p[:, j]
    return out


def scores(P, y):
    ybin = (y == 1).astype(int)
    _, slope = logistic_calibration(P[:, 1], ybin)
    return dict(
        auc=float(roc_auc_score(ybin, P[:, 1])),
        brier=float(np.mean(np.sum((P - np.eye(3)[y]) ** 2, axis=1))),
        cal_slope=slope,
        cal_in_large=offset_intercept(P[:, 1], ybin),
        p_dis_pred=float(P[:, 1].mean()),
        p_dis_obs=float(ybin.mean()),
    )


def tenor(P, p_new):
    """Replace the ADL probability and keep the other two in their original ratio."""
    out = P.copy()
    out[:, 1] = p_new
    rest = np.clip(P[:, 0] + P[:, 2], 1e-12, None)
    out[:, 0] = (1 - p_new) * P[:, 0] / rest
    out[:, 2] = (1 - p_new) * P[:, 2] / rest
    return out


def transport_block(coh):
    """A. Leave-one-source-out transport with corrected CITL and bootstrap CI."""
    rng = np.random.default_rng(20261103)
    point, boots = [], []
    for test in SRCS:
        tr = pd.concat([coh[s] for s in SRCS if s != test], ignore_index=True)
        te = coh[test].copy()
        for strat, cols in STRAT.items():
            trc = tr.dropna(subset=cols)
            tec = te.dropna(subset=cols).reset_index(drop=True)
            Xtr, st = design(trc, cols)
            Xte, _ = design(tec, cols, st)
            m = fit(Xtr, trc['status'].astype(int).values)
            P = proba(m, Xte)
            y = tec['status'].astype(int).values
            s = scores(P, y)
            point.append(dict(test=test, strategy=strat, n=len(tec),
                              events=int((y == 1).sum()), protocol='full_receiving',
                              **s))

            draws = {k: [] for k in s}
            n = len(tec)
            for _ in range(N_BOOT):
                idx = rng.integers(0, n, n)
                yb = y[idx]
                if (yb == 1).sum() == 0 or (yb == 1).sum() == n:
                    continue
                b = scores(P[idx], yb)
                for k in draws:
                    draws[k].append(b[k])
            row = dict(test=test, strategy=strat, n=n,
                       events=int((y == 1).sum()), protocol='full_receiving')
            for k, v in draws.items():
                a = np.asarray(v, float)
                a = a[np.isfinite(a)]
                row[k] = s[k]
                row[f'{k}_lo'] = float(np.percentile(a, 2.5))
                row[f'{k}_hi'] = float(np.percentile(a, 97.5))
            boots.append(row)
    pd.DataFrame(point).to_csv(RES / 'transport_q.csv', index=False)
    pd.DataFrame(boots).to_csv(RES / 'transport_q_ci.csv', index=False)
    print('wrote results/transport_q.csv and results/transport_q_ci.csv')


def updating_block(coh):
    """B. Paired local updating on disjoint update/test halves."""
    rows = []
    for test in SRCS:
        tr = pd.concat([coh[s] for s in SRCS if s != test], ignore_index=True)
        te = coh[test].copy()
        for strat in ['Q']:
            cols = STRAT[strat]
            trc = tr.dropna(subset=cols)
            tec = te.dropna(subset=cols).reset_index(drop=True)
            Xtr, st = design(trc, cols)
            m = fit(Xtr, trc['status'].astype(int).values)
            Xte, _ = design(tec, cols, st)
            P0 = proba(m, Xte)
            y = tec['status'].astype(int).values

            for seed in SPLIT_SEEDS:
                idx = np.random.default_rng(seed).permutation(len(tec))
                h = len(tec) // 2
                a, b = idx[:h], idx[h:]

                base = scores(P0[b], y[b])
                upd = {'none': base}

                # intercept-only update, fitted on the update half only
                ai = offset_intercept(P0[a, 1], (y[a] == 1).astype(int))
                upd['intercept'] = scores(tenor(P0[b], expit(lp(P0[b, 1]) + ai)), y[b])

                # logistic recalibration: free intercept and slope
                c0, c1 = logistic_calibration(P0[a, 1], (y[a] == 1).astype(int))
                upd['logistic_recal'] = scores(
                    tenor(P0[b], expit(c0 + c1 * lp(P0[b, 1]))), y[b])

                # coefficient re-estimation on the update half
                Xa, sa = design(tec.iloc[a], cols)
                Xb, _ = design(tec.iloc[b], cols, sa)
                mc = fit(Xa, y[a])
                upd['coefficient_revision'] = scores(proba(mc, Xb), y[b])

                for scheme, s in upd.items():
                    rows.append(dict(test=test, strategy=strat, split_seed=seed,
                                     update=scheme, **s))

    df = pd.DataFrame(rows)
    df.to_csv(RES / 'local_updates.csv', index=False)
    # paired difference against no update, within the same split
    out = []
    for (test, seed), g in df.groupby(['test', 'split_seed']):
        base = g[g['update'] == 'none'].iloc[0]
        for _, r in g[g['update'] != 'none'].iterrows():
            out.append(dict(test=test, split_seed=seed, update=r['update'],
                            d_brier=r['brier'] - base['brier'],
                            d_auc=r['auc'] - base['auc']))
    pd.DataFrame(out).to_csv(RES / 'local_update_paired.csv', index=False)
    print('wrote results/local_updates.csv and local_update_paired.csv')


if __name__ == '__main__':
    coh = {s: pd.read_parquet(DATA / f'cohort_{s}.parquet') for s in SRCS}
    transport_block(coh)
    updating_block(coh)
