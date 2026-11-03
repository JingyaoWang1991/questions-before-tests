"""10_transport_samewave.py — same-survey, cross-wave transport.
The same instrument and country two years apart, so the only change is calendar time.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
RNG = np.random.default_rng(20261025)
SRCS = ['CHARLS', 'ELSA', 'HRS']
STRAT = {
    'R': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str'],
    'Q': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z', 'climb'],
    'E3': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z',
           'climb', 'grip', 'memory'],
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
            if c not in stats:
                m, s = v.mean(), v.std(ddof=0)
                stats[c] = (m, s if s > 1e-9 else 1.0)
            m, s = stats[c]
            X[c] = (v - m) / s
    return X, stats


def score(p, y, eps=1e-6):
    lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    m = LogisticRegression(C=1e6, max_iter=4000).fit(
        np.column_stack([np.ones(len(lp)), lp]), y)
    slope = float(m.coef_[0][1])
    citl = float(np.log(np.clip(y.mean(), eps, 1 - eps) /
                        (1 - np.clip(y.mean(), eps, 1 - eps))) - lp.mean())
    return dict(auc=float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan,
                brier=float(np.mean((p - y) ** 2)),
                cal_slope=slope, cal_in_large=citl,
                p_pred=float(p.mean()), p_obs=float(y.mean()))


rows = []
for src in SRCS:
    tr = pd.read_parquet(f'data/cohort_{src}.parquet')      # 1st wave pair
    te = pd.read_parquet(f'data/cohort2_{src}.parquet')     # 2nd wave pair
    for strat, cols in STRAT.items():
        trc = tr.dropna(subset=cols)
        tec = te.dropna(subset=cols).reset_index(drop=True)
        if len(tec) < 300 or (tec['status'] == 1).sum() < 30:
            rows.append(dict(setting='same survey, other wave', test=src,
                             strategy=strat, n_train=len(trc), n_test=len(tec),
                             note='too few events'))
            continue
        Xtr, st = design(trc, cols)
        Xte, _ = design(tec, cols, st)
        ytr = trc['status'].astype(int).values
        yte = (tec['status'].astype(int) == 1).astype(int).values
        m = LogisticRegression(C=1e6, max_iter=5000, solver='lbfgs')
        m.fit(Xtr.values, ytr)
        p = m.predict_proba(Xte.values)[:, 1]
        s = score(p, yte)
        rows.append(dict(setting='same survey, other wave', test=src,
                         strategy=strat, n_train=len(trc), n_test=len(tec),
                         events=int(yte.sum()), **s, note=''))

out = pd.DataFrame(rows)
out.to_csv('results/transport_samewave.csv', index=False)

pd.set_option('display.width', 220)
print('=== same-survey, cross-wave transport ===')
print(out.round(3).to_string(index=False))

# ---- compare with the cross-survey (LOSO) transports on the same metrics
cross = pd.read_csv('results/transport.csv')
cross = cross[(cross['update'] == 'none') & (cross['strategy'].isin(['R', 'Q', 'E3']))]

def summarise(df, label):
    d = df.dropna(subset=['cal_slope'])
    return dict(layer=label, n=len(d),
                abs_slope_dev=(d['cal_slope'] - 1).abs().mean(),
                abs_citl=d['cal_in_large'].abs().mean(),
                n_outside_band=int(((d['cal_slope'] < 0.8) |
                                    (d['cal_slope'] > 1.2)).sum()),
                slope_min=d['cal_slope'].min(), slope_max=d['cal_slope'].max())

comp = pd.DataFrame([
    summarise(cross, 'cross-survey (different instrument)'),
    summarise(out, 'same survey, other wave (same instrument)'),
])
print()
print('=== layer comparison ===')
print(comp.round(3).to_string(index=False))

# ---- the prespecified decision rule
cw = (out['cal_slope'] - 1).abs().mean()
xs = (cross['cal_slope'] - 1).abs().mean()
print()
print(f'cross-wave mean |slope-1| = {cw:.3f}')
print(f'cross-survey mean |slope-1| = {xs:.3f}')
# Prespecified rule: the instrument-specific attribution survives only if
# cross-wave calibration is nearly perfect (|slope-1| <= 0.04) AND clearly better
# than cross-survey. Anything else means miscalibration is generic to transport.
if cw <= 0.04 and cw < xs:
    print('DECISION: cross-wave nearly calibrated -> instrument-specific attribution '
          'SUPPORTED. Report the three-layer contrast as the explanation.')
else:
    print(f'DECISION: cross-wave |slope-1| = {cw:.3f} (rule requires <= 0.040) -> '
          'instrument-specific attribution NOT supported. Report instead that '
          'miscalibration accompanies every kind of transport; this is what the '
          'manuscript states.')

# ---- Mann-Whitney on slope and CITL, as in the reviewer-style comparison
try:
    from scipy.stats import mannwhitneyu
    for col in ['cal_slope', 'cal_in_large']:
        a = cross[col].dropna()
        b = out[col].dropna()
        u = mannwhitneyu(a, b, alternative='two-sided')
        print(f'  Mann-Whitney {col}: p = {u.pvalue:.2f}')
except Exception as e:
    print('  (scipy test skipped:', e, ')')
