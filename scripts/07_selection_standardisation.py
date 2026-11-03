"""07_selection_standardisation.py  (A3 + A4 + A5)

A3  Measurement-subsample selection diagnostic: does Q perform differently in the
    participants who could complete a grip test vs those who could not? If the
    small grip/cognition increment is driven by selection, Q should already
    discriminate better in the grip-available subset.
    Also: low-grip threshold (categorical) instead of continuous kg.

A4  Bootstrap confidence intervals for the transport metrics (calibration slope,
    calibration-in-the-large, AUC) reported in Table 4.

A5  Age- and sex-standardised 24-month disability risk per source, to separate
    population age structure from genuine risk-level differences.

Writes results/selection_diagnostic.csv, results/transport_ci.csv,
results/standardised_risk.csv.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
RNG = np.random.default_rng(20261025)
SRCS = ['CHARLS', 'ELSA', 'HRS']
BIN = ('male', 'htn', 'dia', 'hrt', 'str', 'climb', 'cesd_z')
STRAT = {
    'R': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str'],
    'Q': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z', 'climb'],
    'E1': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z', 'climb', 'grip'],
    'E3': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z', 'climb',
           'grip', 'memory'],
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


def fit_auc(df, cols, ycol='y'):
    sub = df.dropna(subset=cols)
    y = sub[ycol].astype(int).values
    if len(sub) < 150 or y.sum() < 20 or y.sum() == len(sub):
        return np.nan, len(sub), int(y.sum())
    X, _ = design(sub, cols)
    m = LogisticRegression(C=1e6, max_iter=5000, solver='lbfgs').fit(X.values, y)
    return float(roc_auc_score(y, m.predict_proba(X.values)[:, 1])), len(sub), int(y.sum())


# ================================================== A3 selection diagnostic
rows = []
for s in SRCS:
    d = pd.read_parquet(f'data/cohort_{s}.parquet')
    d['y'] = (d['status'] == 1).astype(int)
    full = d
    has = d[d['grip'].notna()]
    non = d[d['grip'].isna()]
    for lab, sub in [('full sample', full), ('grip measured', has),
                     ('grip not measured', non)]:
        for st in ['R', 'Q']:
            a, n, ev = fit_auc(sub, STRAT[st])
            rows.append(dict(src=s, subset=lab, strategy=st, n=n, events=ev, auc=a))
    # low-grip threshold instead of continuous kg (sex-specific lowest quintile)
    d2 = d[d['grip'].notna()].copy()
    cut = d2.groupby('male')['grip'].transform(lambda v: v.quantile(0.20))
    d2['lowgrip'] = (d2['grip'] <= cut).astype(float)
    STRAT['Q_lowgrip'] = STRAT['Q'] + ['lowgrip']
    a, n, ev = fit_auc(d2, STRAT['Q_lowgrip'])
    a_q, _, _ = fit_auc(d2, STRAT['Q'])
    rows.append(dict(src=s, subset='grip measured', strategy='Q_lowgrip_threshold',
                     n=n, events=ev, auc=a))
    rows.append(dict(src=s, subset='grip measured', strategy='Q_continuous_grip_ref',
                     n=n, events=ev, auc=a_q))

sel = pd.DataFrame(rows)
sel.to_csv('results/selection_diagnostic.csv', index=False)
print('=== A3 selection diagnostic ===')
print(sel.pivot_table(index=['src', 'subset'], columns='strategy',
                      values='auc').round(3).to_string())

# ================================================== A4 transport bootstrap CI
STRAT_T = {'R': STRAT['R'], 'Q': STRAT['Q'], 'E3': STRAT['E3']}
coh = {s: pd.read_parquet(f'data/cohort_{s}.parquet') for s in SRCS}


def score_boot(p, y):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    try:
        m = LogisticRegression(C=1e6, max_iter=2000).fit(
            np.column_stack([np.ones(len(lp)), lp]), y)
        slope = float(m.coef_[0][1])
    except Exception:
        slope = np.nan
    citl = float(np.log(np.clip(y.mean(), 1e-6, 1 - 1e-6) /
                        (1 - np.clip(y.mean(), 1e-6, 1 - 1e-6))) - lp.mean())
    try:
        auc = roc_auc_score(y, p)
    except Exception:
        auc = np.nan
    brier = float(np.mean((p - y) ** 2))
    return auc, brier, slope, citl


rows = []
for test in SRCS:
    tr = pd.concat([coh[s] for s in SRCS if s != test], ignore_index=True)
    te = coh[test].copy()
    for st, cols in STRAT_T.items():
        trc = tr.dropna(subset=cols)
        tec = te.dropna(subset=cols)
        Xtr, stt = design(trc, cols)
        Xte, _ = design(tec, cols, stt)
        ytr = trc['status'].astype(int).values
        yte = tec['status'].astype(int).values
        m = LogisticRegression(C=1e6, max_iter=5000, solver='lbfgs').fit(Xtr.values, ytr)
        p = m.predict_proba(Xte.values)[:, 1]
        yb = (yte == 1).astype(int)
        point = score_boot(p, yb)
        boots = []
        n = len(tec)
        for _ in range(500):
            idx = RNG.integers(0, n, n)
            if yb[idx].sum() < 10 or yb[idx].sum() == n:
                continue
            boots.append(score_boot(p[idx], yb[idx]))
        boots = np.array([b for b in boots if np.all(np.isfinite(b))])
        rows.append(dict(
            test=test, strategy=st, n=n, events=int(yb.sum()),
            auc=point[0],
            auc_lo=np.percentile(boots[:, 0], 2.5),
            auc_hi=np.percentile(boots[:, 0], 97.5),
            brier=point[1],
            brier_lo=np.percentile(boots[:, 1], 2.5),
            brier_hi=np.percentile(boots[:, 1], 97.5),
            cal_slope=point[2],
            slope_lo=np.percentile(boots[:, 2], 2.5),
            slope_hi=np.percentile(boots[:, 2], 97.5),
            cal_in_large=point[3],
            citl_lo=np.percentile(boots[:, 3], 2.5),
            citl_hi=np.percentile(boots[:, 3], 97.5)))
    print('A4 done', test, flush=True)

tci = pd.DataFrame(rows)
tci.to_csv('results/transport_ci.csv', index=False)
print()
print('=== A4 transport bootstrap CI ===')
print(tci.round(3).to_string(index=False))

# ================================================== A5 age-sex standardisation
pooled = pd.concat([coh[s] for s in SRCS], ignore_index=True)
pooled = pooled[pooled['status'].isin([0, 1, 2])]
# standard population: pooled age-band x sex distribution
pooled['ageb'] = pd.cut(pooled['age'], [59, 64, 69, 74, 79, 84, 200],
                        labels=['60-64', '65-69', '70-74', '75-79', '80-84', '85+'])
w = pooled.groupby(['ageb', 'male'], observed=True).size()
w = w / w.sum()

rows = []
for s in SRCS + ['SHARE']:
    d = pd.read_parquet(f'data/cohort_{s}.parquet')
    d = d[d['status'].isin([0, 1, 2])].copy()
    d['ageb'] = pd.cut(d['age'], [59, 64, 69, 74, 79, 84, 200],
                       labels=['60-64', '65-69', '70-74', '75-79', '80-84', '85+'])
    d['dis'] = (d['status'] == 1).astype(float)
    d['dth'] = (d['status'] == 2).astype(float)
    crude_dis = d['dis'].mean()
    crude_dth = d['dth'].mean()
    g = d.groupby(['ageb', 'male'], observed=True).agg(
        p_dis=('dis', 'mean'), p_dth=('dth', 'mean'), n=('dis', 'size'))
    # direct standardisation onto the pooled standard population
    idx = w.index.intersection(g.index)
    ww = w.loc[idx] / w.loc[idx].sum()
    std_dis = float((g.loc[idx, 'p_dis'] * ww).sum())
    std_dth = float((g.loc[idx, 'p_dth'] * ww).sum())
    rows.append(dict(src=s, N=len(d),
                     crude_dis=float(crude_dis), std_dis=std_dis,
                     crude_dead=float(crude_dth), std_dead=std_dth,
                     mean_age=float(d['age'].mean()),
                     pct_80plus=float((d['age'] >= 80).mean() * 100)))

sr = pd.DataFrame(rows)
sr.to_csv('results/standardised_risk.csv', index=False)
print()
print('=== A5 age-sex standardised risk ===')
print(sr.round(4).to_string(index=False))
