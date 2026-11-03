"""06_share_multicountry.py  (A2)
Within-SHARE multi-country transport: train on all other countries, test on each
country, for the Q strategy (alive-only two-state outcome).

This extends the cross-national calibration-failure finding from 3 countries to
the ~20 European countries with usable sample size.

Writes results/share_multicountry.csv and results/share_multicountry_calib.csv.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
RNG = np.random.default_rng(20261025)
COLS = ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z', 'climb']
BIN = ('male', 'htn', 'dia', 'hrt', 'str', 'climb', 'cesd_z')

# SHARE country codes -> ISO labels (SHARE release convention)
ISO = {11: 'Austria', 12: 'Germany', 13: 'Sweden', 14: 'Netherlands', 15: 'Spain',
       16: 'Italy', 17: 'France', 18: 'Denmark', 19: 'Greece', 20: 'Switzerland',
       23: 'Belgium', 25: 'Israel', 28: 'Czechia', 29: 'Poland', 30: 'Ireland',
       31: 'Luxembourg', 32: 'Hungary', 33: 'Portugal', 34: 'Slovenia',
       35: 'Estonia', 47: 'Croatia', 48: 'Lithuania', 51: 'Bulgaria',
       53: 'Cyprus', 55: 'Finland', 57: 'Latvia', 59: 'Malta', 61: 'Romania',
       63: 'Slovakia'}


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


def calib(p, y, eps=1e-6):
    lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    m = LogisticRegression(C=1e6, max_iter=4000).fit(
        np.column_stack([np.ones(len(lp)), lp]), y)
    slope = float(m.coef_[0][1])
    citl = float(np.log(np.clip(y.mean(), 1e-6, 1 - 1e-6) /
                        (1 - np.clip(y.mean(), 1e-6, 1 - 1e-6))) - lp.mean())
    return slope, citl


d = pd.read_parquet('data/cohort_SHARE.parquet')
d = d[d['status'].isin([0, 1])].dropna(subset=COLS).copy()
d['y'] = (d['status'] == 1).astype(int)
d['ctry'] = pd.to_numeric(d['psu'], errors='coerce').astype('Int64')

rows = []
for ctry, g in d.groupby('ctry'):
    if len(g) < 400 or g['y'].sum() < 40:
        continue
    tr = d[d['ctry'] != ctry]
    Xtr, st = design(tr)
    Xte, _ = design(g, st)
    m = LogisticRegression(C=1e6, max_iter=5000, solver='lbfgs')
    m.fit(Xtr.values, tr['y'].values)
    p = m.predict_proba(Xte.values)[:, 1]
    y = g['y'].values
    auc = roc_auc_score(y, p)
    brier = float(np.mean((p - y) ** 2))
    slope, citl = calib(p, y)

    # within-country apparent AUC for comparison
    Xg, _ = design(g)
    mg = LogisticRegression(C=1e6, max_iter=5000, solver='lbfgs')
    mg.fit(Xg.values, y)
    auc_in = roc_auc_score(y, mg.predict_proba(Xg.values)[:, 1])

    # intercept-only update
    off = np.log(np.clip(y.mean(), 1e-6, 1 - 1e-6) /
                 (1 - np.clip(y.mean(), 1e-6, 1 - 1e-6))) - \
          np.log(np.clip(p.mean(), 1e-6, 1 - 1e-6) /
                 (1 - np.clip(p.mean(), 1e-6, 1 - 1e-6)))
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    p2 = 1 / (1 + np.exp(-(lp + off)))
    auc2 = roc_auc_score(y, p2)
    brier2 = float(np.mean((p2 - y) ** 2))

    # logistic recalibration on a random half, evaluate on the other half
    a1, b1 = [], []
    for _ in range(20):
        idx = RNG.permutation(len(g))
        h = len(g) // 2
        a, b = idx[:h], idx[h:]
        if y[a].sum() < 10 or y[b].sum() < 10:
            continue
        lpa = np.log(np.clip(p[a], 1e-6, 1 - 1e-6) / (1 - np.clip(p[a], 1e-6, 1 - 1e-6)))
        mr = LogisticRegression(C=1e6, max_iter=4000).fit(lpa.reshape(-1, 1), y[a])
        lpb = np.log(np.clip(p[b], 1e-6, 1 - 1e-6) / (1 - np.clip(p[b], 1e-6, 1 - 1e-6)))
        pb = mr.predict_proba(lpb.reshape(-1, 1))[:, 1]
        a1.append(roc_auc_score(y[b], pb))
        b1.append(np.mean((pb - y[b]) ** 2))

    rows.append(dict(country=int(ctry), name=ISO.get(int(ctry), str(ctry)),
                     n=len(g), events=int(y.sum()),
                     prev=float(y.mean()),
                     auc_transport=float(auc), auc_within=float(auc_in),
                     brier_none=brier, brier_intercept=brier2,
                     brier_recal=float(np.mean(b1)) if b1 else np.nan,
                     cal_slope=slope, cal_in_large=citl,
                     auc_intercept=float(auc2),
                     auc_recal=float(np.mean(a1)) if a1 else np.nan))

out = pd.DataFrame(rows).sort_values('n', ascending=False)
out.to_csv('results/share_multicountry.csv', index=False)
print(out.round(4).to_string(index=False))
print()
print('countries tested:', len(out))
print('calibration slope  range: %.3f to %.3f' % (out['cal_slope'].min(), out['cal_slope'].max()))
print('AUC transport      range: %.3f to %.3f' % (out['auc_transport'].min(), out['auc_transport'].max()))
print('AUC within         range: %.3f to %.3f' % (out['auc_within'].min(), out['auc_within'].max()))
print('mean |cal slope-1|: %.3f' % (out['cal_slope'] - 1).abs().mean())
print('countries with slope outside 0.8-1.2: %d / %d'
      % (((out['cal_slope'] < 0.8) | (out['cal_slope'] > 1.2)).sum(), len(out)))
