"""12_cross_validation.py  --  Task 3: five-fold stratified cross-validation

The main paired comparison fits and evaluates on the same participants, so the
increment is apparent. This script re-estimates the R->Q increment with five-fold
stratified cross-validation: models are trained on four folds and scored on the
held-out fold, and AUC is computed on the pooled out-of-fold predictions.

Because the same folds are used for R and Q, the optimism largely cancels in the
paired difference; this quantifies that directly.

Writes results/cv_performance.csv and results/cv_paired.csv.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')
SRCS = ['CHARLS', 'ELSA', 'HRS']
STRAT = {
    'R': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str'],
    'Q': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z', 'climb'],
}
BIN = ('male', 'htn', 'dia', 'hrt', 'str', 'climb', 'cesd_z')
SEED = 20261025


def design(train, test, cols):
    Xtr = pd.DataFrame(index=train.index)
    Xte = pd.DataFrame(index=test.index)
    for c in cols:
        v = pd.to_numeric(train[c], errors='coerce').astype(float)
        w = pd.to_numeric(test[c], errors='coerce').astype(float)
        if c in BIN:
            Xtr[c] = v.fillna(0.0)
            Xte[c] = w.fillna(0.0)
        elif c == 'educ':
            Xtr['educ_mid'] = (v == 2).astype(float)
            Xtr['educ_high'] = (v == 3).astype(float)
            Xte['educ_mid'] = (w == 2).astype(float)
            Xte['educ_high'] = (w == 3).astype(float)
        else:
            m, s = v.mean(), v.std(ddof=0)
            s = s if s > 1e-9 else 1.0
            Xtr[c] = (v - m) / s
            Xte[c] = (w - m) / s
    return Xtr, Xte


def run_cv(df, cols, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.full(len(df), np.nan)
    for tr_i, te_i in skf.split(df, y):
        tr, te = df.iloc[tr_i], df.iloc[te_i]
        Xtr, Xte = design(tr, te, cols)
        m = LogisticRegression(C=1e6, max_iter=5000, solver='lbfgs')
        m.fit(Xtr.values, y[tr_i])
        p = m.predict_proba(Xte.values)
        cls = list(m.classes_)
        oof[te_i] = p[:, cls.index(1)] if 1 in cls else 0.0
    return oof


rows, paired = [], []
for src in SRCS:
    d = pd.read_parquet(f'data/cohort_{src}.parquet')
    cols_all = sorted({c for v in STRAT.values() for c in v})
    d = d.dropna(subset=cols_all).reset_index(drop=True)
    y = d['status'].astype(int).values
    ybin = (y == 1).astype(int)
    oof = {k: run_cv(d, v, y) for k, v in STRAT.items()}
    for k in STRAT:
        rows.append(dict(src=src, strategy=k, n=len(d), events=int(ybin.sum()),
                         auc_cv=float(roc_auc_score(ybin, oof[k]))))
    paired.append(dict(src=src, n=len(d), events=int(ybin.sum()),
                       dauc_cv=float(roc_auc_score(ybin, oof['Q'])
                                     - roc_auc_score(ybin, oof['R']))))
    print('done', src, flush=True)

cv = pd.DataFrame(rows)
cv.to_csv('results/cv_performance.csv', index=False)
cp = pd.DataFrame(paired)
cp.to_csv('results/cv_paired.csv', index=False)

# pooled CV on the concatenation of the three cohorts
pool = pd.concat([pd.read_parquet(f'data/cohort_{s}.parquet') for s in SRCS],
                 ignore_index=True)
cols_all = sorted({c for v in STRAT.values() for c in v})
pool = pool.dropna(subset=cols_all).reset_index(drop=True)
yp = pool['status'].astype(int).values
ypb = (yp == 1).astype(int)
oofp = {k: run_cv(pool, v, yp) for k, v in STRAT.items()}
pooled_cv = dict(src='POOLED', n=len(pool), events=int(ypb.sum()),
                 auc_R=float(roc_auc_score(ypb, oofp['R'])),
                 auc_Q=float(roc_auc_score(ypb, oofp['Q'])),
                 dauc_cv=float(roc_auc_score(ypb, oofp['Q'])
                               - roc_auc_score(ypb, oofp['R'])))

pd.set_option('display.width', 200)
print()
print('=== apparent vs cross-validated ===')
app = pd.read_csv('results/paired_delta.csv')
app = app[(app['set'] == 'R_vs_Q') & (app['comparison'] == 'R->Q')]
for _, r in app.iterrows():
    c = cp[cp['src'] == r['block']]
    cvv = c.iloc[0]['dauc_cv'] if not c.empty else np.nan
    print(f"  {r['block']:8s} apparent dAUC = {r['dauc']:+.4f}   "
          f"5-fold CV dAUC = {cvv:+.4f}")
print(f"  {'POOLED':8s} apparent dAUC = {app[app.block=='POOLED'].iloc[0]['dauc']:+.4f}   "
      f"5-fold CV dAUC = {pooled_cv['dauc_cv']:+.4f}")
print()
print('pooled CV AUC: R = %.4f, Q = %.4f' % (pooled_cv['auc_R'], pooled_cv['auc_Q']))
import json
json.dump(pooled_cv, open('results/cv_pooled.json', 'w'), indent=1)
