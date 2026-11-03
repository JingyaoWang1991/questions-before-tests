"""01_main_analysis.py — paired comparison of the six information strategies.
Apparent AUC, three-state Brier score and calibration for R, Q, E1, E2, E3 and L,
each pair refitted on identical participants. Writes results/main_performance.csv
and results/paired_delta.csv.
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
RES = 'results'
os.makedirs(RES, exist_ok=True)
RNG = np.random.default_rng(20261025)

SRCS = ['CHARLS', 'ELSA', 'HRS']
STRAT = {
    'R':  ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str'],
    'Q':  ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z', 'climb'],
    'E1': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z', 'climb', 'grip'],
    'E2': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z', 'climb', 'memory'],
    'E3': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z', 'climb', 'grip', 'memory'],
    'L':  ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z', 'climb', 'grip', 'memory',
           'p_grip', 'p_memory'],
}
# paired comparison sets: (name, models compared, required variables)
SETS = [
    ('R_vs_Q',  ['R', 'Q'],  'Q'),
    ('Q_vs_E1', ['Q', 'E1'], 'E1'),
    ('Q_vs_E2', ['Q', 'E2'], 'E2'),
    ('Q_vs_E3', ['Q', 'E3'], 'E3'),
    ('Q_vs_E3_steps', ['Q', 'E1', 'E2', 'E3'], 'E3'),
    ('E3_vs_L', ['E3', 'L'], 'L'),
]


# ------------------------------------------------------------------ helpers
def design(df, cols, train_stats=None):
    """Build a numeric design matrix; standardisation learned on training rows."""
    X = pd.DataFrame(index=df.index)
    stats = {} if train_stats is None else train_stats
    for c in cols:
        v = pd.to_numeric(df[c], errors='coerce').astype(float)
        if c in ('male', 'htn', 'dia', 'hrt', 'str', 'climb', 'cesd_z'):
            X[c] = v.fillna(0.0)
        elif c == 'educ':
            X['educ_mid'] = (v == 2).astype(float)
            X['educ_high'] = (v == 3).astype(float)
        else:
            if train_stats is None:
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
    # ensure column order 0,1,2
    out = np.zeros((len(X), 3))
    for j, cls in enumerate(m.classes_):
        out[:, int(cls)] = p[:, j]
    return out


def metrics(P, y):
    """P: n x 3 probability matrix; y in {0,1,2}."""
    n = len(y)
    ybin = (y == 1).astype(int)
    alive = (y != 2)
    d = dict(
        auc=roc_auc_score(ybin, P[:, 1]) if ybin.sum() and ybin.sum() < n else np.nan,
        auc_alive=roc_auc_score(ybin[alive], P[alive, 1])
        if (ybin[alive].sum() and ybin[alive].sum() < alive.sum()) else np.nan,
        brier=float(np.mean(np.sum((P - np.eye(3)[y]) ** 2, axis=1))),
        p_dis_pred=float(P[:, 1].mean()),
        p_dead_pred=float(P[:, 2].mean()),
        p_dis_obs=float(ybin.mean()),
        p_dead_obs=float((y == 2).mean()),
    )
    d.update(calibration(P[:, 1], ybin))
    return d


def calibration(p, ybin, eps=1e-6):
    """calibration intercept (slope fixed 1) and slope (intercept free)."""
    lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    Xa = np.column_stack([np.ones(len(lp)), lp])
    try:
        m = LogisticRegression(C=1e6, max_iter=5000, solver='lbfgs')
        m.fit(Xa, ybin)
        slope = float(m.coef_[0][1])
        icpt = float(m.intercept_[0])
    except Exception:
        slope, icpt = np.nan, np.nan
    try:
        m0 = LogisticRegression(C=1e6, max_iter=5000, solver='lbfgs')
        m0.fit(lp.reshape(-1, 1) * 0 + 1, ybin)   # intercept-only on a constant
        citl = float(m0.intercept_[0] + 0.0)
        # calibration-in-the-large: intercept with slope 1 -> mean(y) - mean(lp)
        citl = float(np.log(np.clip(ybin.mean(), 1e-6, 1 - 1e-6) /
                            (1 - np.clip(ybin.mean(), 1e-6, 1 - 1e-6))) - lp.mean())
    except Exception:
        citl = np.nan
    return dict(cal_slope=slope, cal_intercept=icpt, cal_in_large=citl)


def boot_delta(df, models, y, cols_map, nboot=200):
    """Paired bootstrap of AUC/Brier differences between consecutive strategies."""
    pairs = list(zip(models[:-1], models[1:]))
    n = len(df)
    out = {f'{a}->{b}': {'dauc': [], 'dbrier': []} for a, b in pairs}
    for _ in range(nboot):
        idx = RNG.integers(0, n, n)
        Xs, ys = {}, y[idx]
        for s in models:
            Xs[s] = design(df.iloc[idx], cols_map[s])[0]
        for a, b in pairs:
            try:
                ma, mb = fit(Xs[a], ys), fit(Xs[b], ys)
                Pa, Pb = proba(ma, Xs[a]), proba(mb, Xs[b])
                ybin = (ys == 1).astype(int)
                out[f'{a}->{b}']['dauc'].append(
                    roc_auc_score(ybin, Pb[:, 1]) - roc_auc_score(ybin, Pa[:, 1]))
                out[f'{a}->{b}']['dbrier'].append(
                    np.mean(np.sum((Pb - np.eye(3)[ys]) ** 2, axis=1)) -
                    np.mean(np.sum((Pa - np.eye(3)[ys]) ** 2, axis=1)))
            except Exception:
                continue
    res = {}
    for k, v in out.items():
        for m in ('dauc', 'dbrier'):
            arr = np.array(v[m])
            arr = arr[np.isfinite(arr)]
            res[(k, m)] = (np.percentile(arr, 2.5), np.percentile(arr, 97.5)) if len(arr) > 20 else (np.nan, np.nan)
    return res


# ------------------------------------------------------------------ main loop
def run_block(df, label):
    """Run all paired comparison sets on one dataset; return perf + delta rows."""
    perf, delta, coef = [], [], []
    for setname, models, need in SETS:
        cols = sorted({c for m in models for c in STRAT[m]})
        sub = df.copy()
        for c in cols:
            sub = sub[sub[c].notna()]
        y = sub['status'].astype(int).values
        if len(sub) < 200 or (y == 1).sum() < 20 or (y == 2).sum() < 10:
            delta.append(dict(block=label, set=setname, comparison='(not run)',
                              n=len(sub), events_dis=int((y == 1).sum()),
                              events_dead=int((y == 2).sum()), note='insufficient events'))
            continue
        fits, P = {}, {}
        for s in models:
            X, st = design(sub, STRAT[s])
            fits[s] = fit(X, y)
            P[s] = proba(fits[s], X)
        for s in models:
            mm = metrics(P[s], y)
            mm.update(block=label, set=setname, strategy=s, n=len(sub),
                      events_dis=int((y == 1).sum()), events_dead=int((y == 2).sum()))
            perf.append(mm)
            for c, v in zip(design(sub, STRAT[s])[0].columns, fits[s].coef_[1]):
                coef.append(dict(block=label, set=setname, strategy=s, term=c, beta=v))
        ci = boot_delta(sub, models, y, STRAT, nboot=200)
        for a, b in zip(models[:-1], models[1:]):
            ma = metrics(P[a], y)
            mb = metrics(P[b], y)
            lo, hi = ci.get((f'{a}->{b}', 'dauc'), (np.nan, np.nan))
            blo, bhi = ci.get((f'{a}->{b}', 'dbrier'), (np.nan, np.nan))
            delta.append(dict(
                block=label, set=setname, comparison=f'{a}->{b}', n=len(sub),
                events_dis=int((y == 1).sum()), events_dead=int((y == 2).sum()),
                auc_a=ma['auc'], auc_b=mb['auc'], dauc=mb['auc'] - ma['auc'],
                dauc_lo=lo, dauc_hi=hi,
                brier_a=ma['brier'], brier_b=mb['brier'], dbrier=mb['brier'] - ma['brier'],
                dbrier_lo=blo, dbrier_hi=bhi,
                slope_a=ma['cal_slope'], slope_b=mb['cal_slope']))
    return perf, delta, coef


def main():
    coh = {s: pd.read_parquet(f'data/cohort_{s}.parquet') for s in SRCS}
    for s in SRCS:
        coh[s]['src'] = s

    all_perf, all_delta, all_coef = [], [], []
    for s in SRCS:
        p, d, c = run_block(coh[s], s)
        all_perf += p
        all_delta += d
        all_coef += c
        print('done', s, len(p), len(d), flush=True)

    pooled = pd.concat(coh.values(), ignore_index=True)
    p, d, c = run_block(pooled, 'POOLED')
    all_perf += p
    all_delta += d
    all_coef += c
    print('done POOLED', flush=True)

    pd.DataFrame(all_perf).to_csv(f'{RES}/main_performance.csv', index=False)
    pd.DataFrame(all_delta).to_csv(f'{RES}/paired_delta.csv', index=False)
    pd.DataFrame(all_coef).to_csv(f'{RES}/coefficients.csv', index=False)

    d = pd.DataFrame(all_delta)
    cols = ['block', 'set', 'comparison', 'n', 'events_dis', 'events_dead',
            'auc_a', 'auc_b', 'dauc', 'dauc_lo', 'dauc_hi',
            'dbrier', 'dbrier_lo', 'dbrier_hi', 'slope_a', 'slope_b']
    print(d[cols].to_string(index=False))


if __name__ == '__main__':
    main()
