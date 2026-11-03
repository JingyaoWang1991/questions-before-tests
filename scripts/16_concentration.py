"""16_concentration.py — where new cases fall in the predicted-risk distribution.
Share of cases in the highest-risk fifth and tenth, by strategy. Writes
results/concentration.csv.
"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RES = 'results'
SRCS = ['CHARLS', 'ELSA', 'HRS']
BIN = ('male', 'htn', 'dia', 'hrt', 'str', 'climb', 'cesd_z')
R = ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str']
MODELS = {
    'Q': R + ['shlt', 'cesd_z', 'climb'],
    'TwoItems': R + ['shlt', 'climb'],
}
FRACTIONS = (0.20, 0.10)


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
                stats[c] = (m, s if (s and s > 1e-9) else 1.0)
            m, s = stats[c]
            X[c] = (v - m) / s
    return X, stats


def fit(X, y):
    return LogisticRegression(C=1e6, max_iter=5000, solver='lbfgs').fit(X.values, y)


def proba_dis(m, X):
    p = m.predict_proba(X.values)
    for j, cls in enumerate(m.classes_):
        if int(cls) == 1:
            return p[:, j]
    return np.zeros(len(X))


def block(df, label):
    rows = []
    # identical rows for both models, so the comparison is paired
    sub = df.dropna(subset=sorted(set(MODELS['Q']) | set(MODELS['TwoItems']))).copy()
    y = sub['status'].astype(int).values
    ybin = (y == 1).astype(int)
    n, ncase = len(sub), int(ybin.sum())
    for name, cols in MODELS.items():
        X, _ = design(sub, cols)
        p = proba_dis(fit(X, y), X)
        order = np.argsort(-p)
        for frac in FRACTIONS:
            k = int(round(n * frac))
            sel = order[:k]
            found = int(ybin[sel].sum())
            rows.append(dict(
                block=label, model=name, n=n, cases=ncase,
                top_fraction=frac, n_assessed=k, cases_found=found,
                share_of_cases=found / ncase if ncase else np.nan,
                cases_per_100_assessed=100 * found / k if k else np.nan,
                lift=(found / ncase) / frac if ncase else np.nan,
                cut_point_predicted_risk=float(p[order[k - 1]])))
    return rows


def main():
    coh = {s: pd.read_parquet(f'data/cohort_{s}.parquet') for s in SRCS}
    for s in SRCS:
        coh[s]['src'] = s
    rows = []
    for s in SRCS:
        rows += block(coh[s], s)
        print('concentration', s, flush=True)
    pooled = pd.concat(coh.values(), ignore_index=True)
    rows += block(pooled, 'POOLED')
    out = pd.DataFrame(rows)
    out.to_csv(f'{RES}/concentration.csv', index=False)
    print()
    print(out.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
