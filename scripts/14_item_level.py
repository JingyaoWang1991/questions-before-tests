"""14_item_level.py — item-level decomposition of the questionnaire tier.
Background plus each domain separately and both single items together, with 1,000
paired resamples and five-fold cross-validation. Writes results/item_level_exploratory.csv.
"""
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

# run from the module directory so that 'data/' and 'results/' resolve everywhere
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RES = 'results'
SRCS = ['CHARLS', 'ELSA', 'HRS']
BIN = ('male', 'htn', 'dia', 'hrt', 'str', 'climb', 'cesd_z')

R = ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str']
MODELS = {
    'R': R,
    'R+SRH': R + ['shlt'],
    'R+climb': R + ['climb'],
    'R+mood': R + ['cesd_z'],
    'R+SRH+climb': R + ['shlt', 'climb'],
    'Q': R + ['shlt', 'climb', 'cesd_z'],
}
NBOOT = 1000
SEED = 20261103


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
    sub = df.dropna(subset=MODELS['Q']).copy()
    y = sub.status.astype(int).values
    ybin = (y == 1).astype(int)
    Xs = {k: design(sub, MODELS[k])[0] for k in MODELS}
    P = {k: proba(fit(Xs[k], y), Xs[k]) for k in MODELS}
    auc = {k: roc_auc_score(ybin, P[k][:, 1]) for k in MODELS}
    br = {k: brier(P[k], y) for k in MODELS}

    # 5-fold cross-validated AUC, same folds for every model
    cv = {k: np.full(len(sub), np.nan) for k in MODELS}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr, te in skf.split(np.zeros(len(y)), y):
        dtr, dte = sub.iloc[tr], sub.iloc[te]
        for k in MODELS:
            Xtr, st = design(dtr, MODELS[k])
            Xte, _ = design(dte, MODELS[k], st)
            cv[k][te] = proba(fit(Xtr, y[tr]), Xte)[:, 1]
    auc_cv = {k: roc_auc_score(ybin, cv[k]) for k in MODELS}

    rng = np.random.default_rng(SEED)
    boot = {k: [] for k in MODELS if k != 'R'}
    t0 = time.time()
    for _ in range(nboot):
        idx = rng.integers(0, len(sub), len(sub))
        ys = y[idx]
        yy = (ys == 1).astype(int)
        if yy.sum() < 20:
            continue
        Xb = {k: design(sub.iloc[idx], MODELS[k])[0] for k in MODELS}
        aR = roc_auc_score(yy, proba(fit(Xb['R'], ys), Xb['R'])[:, 1])
        for k in MODELS:
            if k == 'R':
                continue
            boot[k].append(roc_auc_score(yy, proba(fit(Xb[k], ys), Xb[k])[:, 1]) - aR)
    print(f'{label}: {nboot} resamples in {time.time()-t0:.0f}s', flush=True)

    rows = []
    for k in MODELS:
        if k == 'R':
            continue
        arr = np.array(boot[k])
        arr = arr[np.isfinite(arr)]
        rows.append(dict(
            block=label, model=k, n=len(sub),
            events_dis=int((y == 1).sum()), events_dead=int((y == 2).sum()),
            auc=auc[k], dauc_vs_R=auc[k] - auc['R'],
            dauc_lo=np.percentile(arr, 2.5) if len(arr) > 50 else np.nan,
            dauc_hi=np.percentile(arr, 97.5) if len(arr) > 50 else np.nan,
            brier=br[k], dbrier_vs_R=br[k] - br['R'],
            auc_cv=auc_cv[k], dauc_cv_vs_R=auc_cv[k] - auc_cv['R'],
            nboot_ok=len(arr)))
    return rows, dict(block=label, n=len(sub), auc_R=auc['R'], auc_Q=auc['Q'],
                      auc_cv_R=auc_cv['R'], auc_cv_Q=auc_cv['Q'])


def missingness(coh):
    rows = []
    for s, df in coh.items():
        rows.append(dict(source=s, n_cohort=len(df),
                         n_complete_Q=int(df.dropna(subset=MODELS['Q']).shape[0]),
                         **{f'miss_{v}': float(df[v].isna().mean()) for v in
                            ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str',
                             'shlt', 'climb', 'cesd_z', 'grip', 'memory']}))
    return pd.DataFrame(rows)


def main():
    coh = {s: pd.read_parquet(f'data/cohort_{s}.parquet') for s in SRCS}
    for s in SRCS:
        coh[s]['src'] = s
    pooled = pd.concat(coh.values(), ignore_index=True)

    all_rows, summary = [], []
    for s in SRCS:
        r, summ = block(coh[s], s)
        all_rows += r
        summary.append(summ)
    r, summ = block(pooled, 'POOLED')
    all_rows += r
    summary.append(summ)

    pd.DataFrame(all_rows).to_csv(f'{RES}/item_level_exploratory.csv', index=False)
    pd.DataFrame(summary).to_csv(f'{RES}/item_level_auc_R_Q_cv.csv', index=False)
    missingness(coh).to_csv(f'{RES}/item_level_missingness.csv', index=False)
    print(pd.DataFrame(all_rows).round(4).to_string(index=False))


if __name__ == '__main__':
    main()
