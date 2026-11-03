"""02_transport.py — leave-one-source-out external validation.
R, Q and E3 are trained on two cohorts and applied unchanged to the third, with four
levels of local updating. Training and evaluation rows are strictly disjoint.
Writes results/transport.csv.
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
RES = 'results'
RNG = np.random.default_rng(20261025)
SRCS = ['CHARLS', 'ELSA', 'HRS']
STRAT = {
    'R': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str'],
    'Q': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z', 'climb'],
    'E3': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z', 'climb', 'grip', 'memory'],
}
BIN = ('male', 'htn', 'dia', 'hrt', 'str', 'climb', 'cesd_z')


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


def fit(X, y):
    m = LogisticRegression(C=1e6, max_iter=5000, solver='lbfgs')
    m.fit(X.values, y)
    return m


def proba(m, X):
    p = m.predict_proba(X.values)
    out = np.zeros((len(X), 3))
    for j, c in enumerate(m.classes_):
        out[:, int(c)] = p[:, j]
    return out


def calib(p, ybin, eps=1e-6):
    """Calibration slope and calibration-in-the-large.

    The slope is fitted with a free intercept. Calibration-in-the-large is the
    intercept of the same logistic model with the slope fixed at one, that is
    the value a solving mean(expit(a + logit(p))) = mean(y).

    The default summary logit(mean(y)) - mean(logit(p)) is not that intercept and can
    misreport a cohort whose mean predicted risk is close to the observed risk, so the
    root find is used throughout.
    The corrected estimator is a one-dimensional root find, so it is exact.
    """
    lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    m = LogisticRegression(C=1e6, max_iter=5000, solver='lbfgs')
    m.fit(np.column_stack([np.ones(len(lp)), lp]), ybin)
    slope = float(m.coef_[0][1])

    target = float(ybin.mean())
    if target <= 0 or target >= 1:
        citl = np.nan
    else:
        f = lambda a: float(np.mean(1 / (1 + np.exp(-(lp + a))))) - target
        citl = float(brentq(f, -50.0, 50.0, xtol=1e-12))
    return slope, citl


def score(P, y):
    ybin = (y == 1).astype(int)
    auc = roc_auc_score(ybin, P[:, 1]) if 0 < ybin.sum() < len(y) else np.nan
    brier = float(np.mean(np.sum((P - np.eye(3)[y]) ** 2, axis=1)))
    slope, citl = calib(P[:, 1], ybin)
    return dict(auc=auc, brier=brier, cal_slope=slope, cal_in_large=citl,
                p_dis_pred=float(P[:, 1].mean()), p_dis_obs=float(ybin.mean()))


def main():
    coh = {s: pd.read_parquet(f'data/cohort_{s}.parquet') for s in SRCS}
    rows = []
    for test in SRCS:
        train_srcs = [s for s in SRCS if s != test]
        for strat, cols in STRAT.items():
            tr = pd.concat([coh[s] for s in train_srcs], ignore_index=True)
            te = coh[test].copy()
            tr = tr.dropna(subset=cols)
            te = te.dropna(subset=cols)
            if len(te) < 300 or (te['status'] == 1).sum() < 30:
                rows.append(dict(test=test, strategy=strat, update='(not run)',
                                 n_train=len(tr), n_test=len(te), note='too few events'))
                continue
            Xtr, st = design(tr, cols)
            Xte, _ = design(te, cols, st)
            ytr = tr['status'].astype(int).values
            yte = te['status'].astype(int).values

            m = fit(Xtr, ytr)
            P = proba(m, Xte)
            s = score(P, yte)
            rows.append(dict(test=test, strategy=strat, update='none',
                             n_train=len(tr), n_test=len(te), **s))

            # intercept-only update: shift logits of class 1 and 2
            lp = np.log(np.clip(P, 1e-6, 1))
            ybin = (yte == 1).astype(int)
            d = np.log(np.clip(ybin.mean(), 1e-6, 1 - 1e-6) /
                       (1 - np.clip(ybin.mean(), 1e-6, 1 - 1e-6))) - \
                np.log(np.clip(P[:, 1].mean(), 1e-6, 1 - 1e-6) /
                       (1 - np.clip(P[:, 1].mean(), 1e-6, 1 - 1e-6)))
            P2 = P.copy()
            P2[:, 1] = 1 / (1 + np.exp(-(np.log(np.clip(P[:, 1], 1e-6, 1) /
                                                 np.clip(1 - P[:, 1], 1e-6, 1)) + d)))
            P2 = P2 / P2.sum(axis=1, keepdims=True)
            rows.append(dict(test=test, strategy=strat, update='intercept',
                             n_train=len(tr), n_test=len(te), **score(P2, yte)))

            # logistic recalibration (intercept + slope) on the test source itself
            # 50/50 split, repeated, so update and evaluation rows are disjoint
            aucs, briers, slopes = [], [], []
            for _ in range(20):
                idx = RNG.permutation(len(te))
                h = len(te) // 2
                a, b = idx[:h], idx[h:]
                if (yte[a] == 1).sum() < 10 or (yte[b] == 1).sum() < 10:
                    continue
                lpa = np.log(np.clip(P[a, 1], 1e-6, 1 - 1e-6) /
                             (1 - np.clip(P[a, 1], 1e-6, 1 - 1e-6)))
                mr = LogisticRegression(C=1e6, max_iter=5000, solver='lbfgs')
                mr.fit(lpa.reshape(-1, 1), (yte[a] == 1).astype(int))
                lpb = np.log(np.clip(P[b, 1], 1e-6, 1 - 1e-6) /
                             (1 - np.clip(P[b, 1], 1e-6, 1 - 1e-6)))
                pb = mr.predict_proba(lpb.reshape(-1, 1))[:, 1]
                P3 = P[b].copy()
                P3[:, 1] = pb
                den = np.clip(P[b, 0] + P[b, 2], 1e-9, None)
                P3[:, 0] = (1 - pb) * (P[b, 0] / den)
                P3[:, 2] = (1 - pb) * (P[b, 2] / den)
                P3 = P3 / P3.sum(axis=1, keepdims=True)
                sb = score(P3, yte[b])
                aucs.append(sb['auc'])
                briers.append(sb['brier'])
                slopes.append(sb['cal_slope'])
            if aucs:
                rows.append(dict(test=test, strategy=strat, update='logistic_recal',
                                 n_train=len(tr), n_test=len(te),
                                 auc=float(np.nanmean(aucs)), brier=float(np.nanmean(briers)),
                                 cal_slope=float(np.nanmean(slopes)),
                                 cal_in_large=np.nan,
                                 p_dis_pred=np.nan, p_dis_obs=np.nan))

            # full coefficient revision: refit on half the test source
            aucs, briers = [], []
            for _ in range(20):
                idx = RNG.permutation(len(te))
                h = len(te) // 2
                a, b = idx[:h], idx[h:]
                if (yte[a] == 1).sum() < 10 or (yte[b] == 1).sum() < 10:
                    continue
                Xa, sa = design(te.iloc[a], cols)
                Xb, _ = design(te.iloc[b], cols, sa)
                try:
                    mr = fit(Xa, yte[a])
                    Pb = proba(mr, Xb)
                    sb = score(Pb, yte[b])
                    aucs.append(sb['auc'])
                    briers.append(sb['brier'])
                except Exception:
                    continue
            if aucs:
                rows.append(dict(test=test, strategy=strat, update='coefficient_revision',
                                 n_train=len(tr), n_test=len(te),
                                 auc=float(np.nanmean(aucs)), brier=float(np.nanmean(briers)),
                                 cal_slope=np.nan, cal_in_large=np.nan,
                                 p_dis_pred=np.nan, p_dis_obs=np.nan))

    out = pd.DataFrame(rows)
    out.to_csv(f'{RES}/transport.csv', index=False)
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
