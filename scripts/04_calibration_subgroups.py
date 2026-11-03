"""04_calibration_subgroups.py
Absolute risk, decile calibration and subgroup discrimination for R/Q/E1/E2/E3.
Writes results/absolute_risk.csv, results/calibration_deciles.csv,
results/subgroup.csv.
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
RES = 'results'
SRCS = ['CHARLS', 'ELSA', 'HRS', 'SHARE']
BIN = ('male', 'htn', 'dia', 'hrt', 'str', 'climb', 'cesd_z')
STRAT = {
    'R': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str'],
    'Q': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z', 'climb'],
    'E1': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z', 'climb', 'grip'],
    'E2': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z', 'climb', 'memory'],
    'E3': ['age', 'male', 'educ', 'htn', 'dia', 'hrt', 'str', 'shlt', 'cesd_z', 'climb', 'grip', 'memory'],
}


def design(df, cols):
    X = pd.DataFrame(index=df.index)
    for c in cols:
        v = pd.to_numeric(df[c], errors='coerce').astype(float)
        if c in BIN:
            X[c] = v.fillna(0.0)
        elif c == 'educ':
            X['educ_mid'] = (v == 2).astype(float)
            X['educ_high'] = (v == 3).astype(float)
        else:
            m, s = v.mean(), v.std(ddof=0)
            s = s if (s and s > 1e-9) else 1.0
            X[c] = (v - m) / s
    return X


def main():
    risk, calib, sub = [], [], []
    for s in SRCS:
        d = pd.read_parquet(f'data/cohort_{s}.parquet')
        three = s != 'SHARE'
        for st, cols in STRAT.items():
            sub_d = d.dropna(subset=cols)
            y = sub_d['status'].astype(int).values
            X = design(sub_d, cols)
            m = LogisticRegression(C=1e6, max_iter=5000, solver='lbfgs')
            m.fit(X.values, y)
            p = m.predict_proba(X.values)
            P = np.zeros((len(X), 3))
            for j, c in enumerate(m.classes_):
                P[:, int(c)] = p[:, j]

            risk.append(dict(src=s, strategy=st, n=len(sub_d),
                             obs_dis=float((y == 1).mean()),
                             pred_dis=float(P[:, 1].mean()),
                             obs_dead=float((y == 2).mean()),
                             pred_dead=float(P[:, 2].mean()),
                             obs_dis_alive=float((y[y != 2] == 1).mean()),
                             pred_dis_alive=float((P[y != 2, 1] /
                                                   np.clip(P[y != 2, 1] + P[y != 2, 0], 1e-9, None)).mean())))

            # decile calibration of p(disabled)
            ybin = (y == 1).astype(int)
            q = pd.qcut(pd.Series(P[:, 1]).rank(method='first'), 10, labels=False)
            for g in range(10):
                msk = q == g
                if msk.sum() == 0:
                    continue
                calib.append(dict(src=s, strategy=st, decile=g + 1, n=int(msk.sum()),
                                  pred=float(P[msk, 1].mean()),
                                  obs=float(ybin[msk].mean())))

            # subgroups
            for var, cuts, labels in [
                    ('age', [-np.inf, 70, 80, np.inf], ['60-69', '70-79', '80+']),
                    ('male', [-0.5, 0.5, 1.5], ['female', 'male']),
                    ('educ', [0.5, 1.5, 2.5, 3.5], ['low', 'mid', 'high'])]:
                v = pd.to_numeric(sub_d[var], errors='coerce').astype(float)
                grp = pd.cut(v, cuts, labels=labels)
                for lab in labels:
                    msk = (grp == lab).values
                    if msk.sum() < 150 or (y[msk] == 1).sum() < 15:
                        continue
                    try:
                        a = roc_auc_score((y[msk] == 1).astype(int), P[msk, 1])
                    except Exception:
                        continue
                    sub.append(dict(src=s, strategy=st, var=var, group=lab,
                                    n=int(msk.sum()), events=int((y[msk] == 1).sum()),
                                    auc=float(a)))
            print('done', s, st, flush=True)

    pd.DataFrame(risk).to_csv(f'{RES}/absolute_risk.csv', index=False)
    pd.DataFrame(calib).to_csv(f'{RES}/calibration_deciles.csv', index=False)
    pd.DataFrame(sub).to_csv(f'{RES}/subgroup.csv', index=False)

    r = pd.DataFrame(risk)
    print(r.round(4).to_string(index=False))
    sb = pd.DataFrame(sub)
    print(sb.pivot_table(index=['var', 'group'], columns=['src', 'strategy'],
                         values='auc').round(3).to_string())


if __name__ == '__main__':
    main()
