"""11_layer_comparison.py
Three transport layers on identical metrics:
  L1 same survey, other wave   (same instrument, same country, 2 years apart)
  L2 same instrument, other country  (SHARE, 13 countries)
  L3 different survey, other country (CHARLS/ELSA/HRS, LOSO)

Quantifies whether miscalibration is specific to instrument change (the original
attribution) or generic to any transport. Writes results/transport_layers.csv.
"""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

# ---- L1 same survey, other wave
l1 = pd.read_csv('results/transport_samewave.csv')
l1 = l1[l1['cal_slope'].notna()].copy()
l1['layer'] = 'same survey, other wave'
l1['layer_short'] = 'L1  other wave'
l1['auc_loss'] = np.nan   # within-cohort AUC for this pair is in temporal_performance

# within-cohort AUC for the 2nd pair, to compute L1 discrimination loss
tp = pd.read_csv('results/temporal_performance.csv')
tp = tp[tp['set'] == 'R_vs_Q']
for i, r in l1.iterrows():
    w = tp[(tp['block'] == r['test'] + '_2nd') & (tp['strategy'] == r['strategy'])]
    if not w.empty:
        l1.loc[i, 'auc_within'] = w.iloc[0]['auc']
l1['auc_loss'] = l1['auc_within'] - l1['auc']

# ---- L2 same instrument, other country (SHARE)
l2 = pd.read_csv('results/share_multicountry.csv').copy()
l2['layer'] = 'same instrument, other country'
l2['layer_short'] = 'L2  other country, same instrument'
l2['test'] = 'SHARE'
l2['strategy'] = 'Q'
l2['auc_within'] = l2['auc_within']
l2['auc'] = l2['auc_transport']
l2['auc_loss'] = l2['auc_within'] - l2['auc_transport']

# ---- L3 different survey (LOSO)
l3 = pd.read_csv('results/transport.csv')
l3 = l3[(l3['update'] == 'none') & (l3['strategy'].isin(['R', 'Q', 'E3']))].copy()
l3['layer'] = 'different survey, other country'
l3['layer_short'] = 'L3  other country, other survey'
# within-cohort AUC on the same rows
mp = pd.read_csv('results/main_performance.csv')
mp = mp[mp['set'] == 'R_vs_Q']
loss = []
for _, r in l3.iterrows():
    w = mp[(mp['block'] == r['test']) & (mp['strategy'] == r['strategy'])]
    loss.append(w.iloc[0]['auc'] - r['auc'] if not w.empty else np.nan)
l3['auc_loss'] = loss

keep = ['layer', 'layer_short', 'test', 'strategy', 'auc', 'cal_slope',
        'cal_in_large', 'auc_loss']
allr = pd.concat([l1[keep], l2[keep], l3[keep]], ignore_index=True)
allr.to_csv('results/transport_layers.csv', index=False)

pd.set_option('display.width', 200)
print('=== three transport layers ===')
summ = allr.groupby('layer_short').agg(
    n=('cal_slope', 'size'),
    slope_min=('cal_slope', 'min'),
    slope_max=('cal_slope', 'max'),
    abs_slope_dev=('cal_slope', lambda s: (s - 1).abs().mean()),
    abs_citl=('cal_in_large', lambda s: s.abs().mean()),
    n_outside_band=('cal_slope', lambda s: int(((s < 0.8) | (s > 1.2)).sum())),
    auc_loss_mean=('auc_loss', lambda s: s.mean()),
    auc_loss_max=('auc_loss', 'max')).round(3)
print(summ.to_string())

print()
print('=== pairwise Mann-Whitney on calibration slope ===')
groups = {k: v['cal_slope'].dropna().values
          for k, v in allr.groupby('layer_short')}
keys = list(groups)
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        u = mannwhitneyu(groups[keys[i]], groups[keys[j]], alternative='two-sided')
        print(f'  {keys[i]}  vs  {keys[j]}:  p = {u.pvalue:.2f}')

print()
print('=== interpretation ===')
dev = summ['abs_slope_dev']
print(f'all three layers show mean |slope-1| between {dev.min():.3f} and {dev.max():.3f};')
print('no layer is exempt, so miscalibration is generic to transport rather than')
print('specific to a change of instrument.')
