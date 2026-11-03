"""15_weighted_sensitivity.py — survey-weighted sensitivity.
Repeats the five paired comparisons of 13_paired_bootstrap.py under each survey's own
weights, with a cluster bootstrap for CHARLS. Writes results/weighted_sensitivity.csv.
"""
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RES = 'results'
SRCS = ['CHARLS', 'ELSA', 'HRS']
NBOOT = 1000
SEED = 20261103            # fixed so the cluster bootstrap is reproducible

STRAT = {
    'R':  ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str'],
    'Q':  ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z', 'climb'],
    'E1': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z', 'climb', 'grip'],
    'E2': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z', 'climb', 'memory'],
    'E3': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z', 'climb',
           'grip', 'memory'],
    'L':  ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z', 'climb',
           'grip', 'memory', 'p_grip', 'p_memory'],
}
SETS = [
    ('R_vs_Q',  ['R', 'Q']),
    ('Q_vs_E1', ['Q', 'E1']),
    ('Q_vs_E2', ['Q', 'E2']),
    ('Q_vs_E3', ['Q', 'E3']),
    ('E3_vs_L', ['E3', 'L']),
]
BIN = ('male', 'htn', 'dia', 'hrt', 'str', 'climb', 'cesd_z')


def design(df, cols, w, stats=None):
    """Same design matrix as scripts/41, with weighted standardisation."""
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
                ww = np.asarray(w, dtype=float)
                m = float(np.average(v.values, weights=ww))
                var = float(np.average((v.values - m) ** 2, weights=ww))
                s = np.sqrt(var)
                stats[c] = (m, s if s > 1e-9 else 1.0)
            m, s = stats[c]
            X[c] = (v - m) / s
    return X, stats


def fit(X, y, w):
    return LogisticRegression(C=1e6, max_iter=5000, solver='lbfgs').fit(
        X.values, y, sample_weight=w)


def proba(m, X):
    p = m.predict_proba(X.values)
    out = np.zeros((len(X), 3))
    for j, cls in enumerate(m.classes_):
        out[:, int(cls)] = p[:, j]
    return out


def wbrier(P, y, w):
    return float(np.sum(w * np.sum((P - np.eye(3)[y]) ** 2, axis=1)) / np.sum(w))


def wauc(ybin, p, w):
    return float(roc_auc_score(ybin, p, sample_weight=w))


def cluster_index(psu_values, rng, n_clusters):
    """Rows belonging to n_clusters clusters drawn with replacement."""
    pos = {}
    for i, g in enumerate(psu_values):
        pos.setdefault(g, []).append(i)
    keys = list(pos.keys())
    pick = rng.integers(0, len(keys), len(keys))
    idx = []
    for k in pick:
        idx.extend(pos[keys[k]])
    return np.array(idx, dtype=int)


def block(df, label, use_clusters, nboot=NBOOT, report_weighted=True):
    rows = []
    rng = np.random.default_rng(SEED)
    for setname, models in SETS:
        cols = sorted({c for m in models for c in STRAT[m]})
        sub = df.dropna(subset=cols).copy()
        y = sub['status'].astype(int).values
        w = sub['wt'].astype(float).values
        if len(sub) < 200 or (y == 1).sum() < 20 or (y == 2).sum() < 10 or np.sum(w) <= 0:
            rows.append(dict(block=label, comparison=f'{models[0]}->{models[1]}',
                             n=len(sub), note='not estimable'))
            continue
        Xs = {m: design(sub, STRAT[m], w)[0] for m in models}
        P = {m: proba(fit(Xs[m], y, w), Xs[m]) for m in models}
        ybin = (y == 1).astype(int)
        a, b = models
        auc_u, br_u = wauc(ybin, P[a][:, 1], w), wbrier(P[a], y, w)
        auc_w, br_w = wauc(ybin, P[b][:, 1], w), wbrier(P[b], y, w)

        psu_col = sub['psu'].values if use_clusters else None
        da, db = [], []
        t0 = time.time()
        for _ in range(nboot):
            if use_clusters:
                idx = cluster_index(psu_col, rng, None)
            else:
                idx = rng.integers(0, len(sub), len(sub))
            ys, ws = y[idx], w[idx]
            yy = (ys == 1).astype(int)
            if yy.sum() < 20 or (ys == 2).sum() < 10 or np.sum(ws) <= 0:
                continue
            try:
                Xb = {}
                for m in models:
                    Xb[m], _ = design(sub.iloc[idx], STRAT[m], ws)
                Pa = proba(fit(Xb[a], ys, ws), Xb[a])
                Pb = proba(fit(Xb[b], ys, ws), Xb[b])
                da.append(wauc(yy, Pb[:, 1], ws) - wauc(yy, Pa[:, 1], ws))
                db.append(wbrier(Pb, ys, ws) - wbrier(Pa, ys, ws))
            except Exception:
                continue
        da, db = np.array(da), np.array(db)
        if report_weighted:
            rows.append(dict(
                block=label, comparison=f'{a}->{b}', n=len(sub),
                events_dis=int((y == 1).sum()), events_dead=int((y == 2).sum()),
                w_auc_a=auc_u, w_auc_b=auc_w, w_dauc=auc_w - auc_u,
                w_dauc_lo=np.percentile(da, 2.5), w_dauc_hi=np.percentile(da, 97.5),
                w_brier_a=br_u, w_brier_b=br_w, w_dbrier=br_w - br_u,
                w_dbrier_lo=np.percentile(db, 2.5), w_dbrier_hi=np.percentile(db, 97.5),
                resample_unit='psu cluster' if use_clusters else 'individual',
                nboot_ok=len(da), seconds=round(time.time() - t0, 1), note=''))
        else:
            # pooled weighting is not interpretable: see the module docstring
            rows.append(dict(
                block=label, comparison=f'{a}->{b}', n=len(sub),
                events_dis=int((y == 1).sum()), events_dead=int((y == 2).sum()),
                w_auc_a=np.nan, w_auc_b=np.nan, w_dauc=np.nan,
                w_dauc_lo=np.nan, w_dauc_hi=np.nan,
                w_brier_a=np.nan, w_brier_b=np.nan, w_dbrier=np.nan,
                w_dbrier_lo=np.nan, w_dbrier_hi=np.nan,
                resample_unit='individual', nboot_ok=0, seconds=0.0,
                note='not reported: weights not comparable across surveys'))
        print(f'{label} {a}->{b}: {len(da)} replicates in {time.time()-t0:.0f}s '
              f'({"clusters" if use_clusters else "individuals"})'
              f'{"" if report_weighted else " [weighted not reported]"}', flush=True)
    return rows


def risk_block():
    """Weighted crude and direct age-sex standardised risk per cohort."""
    coh_all = {s: pd.read_parquet(f'data/cohort_{s}.parquet') for s in SRCS}
    pooled = pd.concat([coh_all[s] for s in SRCS], ignore_index=True)

    def bands(d):
        return pd.cut(d['age'], [59, 64, 69, 74, 79, 84, 200],
                      labels=['60-64', '65-69', '70-74', '75-79', '80-84', '85+'])

    # weighted standard population: pooled age-band x sex distribution
    pooled = pooled.copy()
    pooled['ageb'] = bands(pooled)
    pooled['w'] = pooled['wt'].astype(float)
    wp = pooled.groupby(['ageb', 'male'], observed=True)['w'].sum()
    wp = wp / wp.sum()

    rows = []
    for s in SRCS:
        d = coh_all[s].copy()
        d['ageb'] = bands(d)
        d['w'] = d['wt'].astype(float)
        d['dis'] = (d['status'] == 1).astype(float)
        d['dth'] = (d['status'] == 2).astype(float)
        crude_dis = float(np.sum(d['w'] * d['dis']) / np.sum(d['w']))
        crude_dth = float(np.sum(d['w'] * d['dth']) / np.sum(d['w']))
        g = d.groupby(['ageb', 'male'], observed=True).apply(
            lambda x: pd.Series({
                'p_dis': np.sum(x['w'] * x['dis']) / np.sum(x['w']),
                'p_dth': np.sum(x['w'] * x['dth']) / np.sum(x['w'])}),
            include_groups=False)
        idx = wp.index.intersection(g.index)
        ww = wp.loc[idx] / wp.loc[idx].sum()
        rows.append(dict(
            src=s, n=len(d),
            w_crude_dis=crude_dis,
            w_std_dis=float((g.loc[idx, 'p_dis'] * ww).sum()),
            w_crude_dead=crude_dth,
            w_std_dead=float((g.loc[idx, 'p_dth'] * ww).sum()),
            wt_gt0_pct=float(100 * (d['wt'] > 0).mean()),
            psu_released=bool(d['psu'].notna().any())))
    return pd.DataFrame(rows)


def main():
    coh = {s: pd.read_parquet(f'data/cohort_{s}.parquet') for s in SRCS}
    for s in SRCS:
        coh[s]['src'] = s
    all_rows = []
    for s in SRCS:
        use_clusters = bool(coh[s]['psu'].notna().any())
        all_rows += block(coh[s], s, use_clusters)
    pooled = pd.concat(coh.values(), ignore_index=True)
    all_rows += block(pooled, 'POOLED', False, report_weighted=False)

    out = pd.DataFrame(all_rows)
    # attach the unweighted estimates for side-by-side reporting
    uw = pd.read_csv(f'{RES}/paired_delta_b1000.csv')
    key = ['block', 'comparison']
    out = out.merge(uw[key + ['dauc', 'dauc_lo', 'dauc_hi', 'dbrier']],
                    on=key, how='left')
    out.to_csv(f'{RES}/weighted_sensitivity.csv', index=False)

    rk = risk_block()
    rk.to_csv(f'{RES}/weighted_risk.csv', index=False)

    print()
    print(out.round(4).to_string(index=False))
    print()
    print(rk.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
