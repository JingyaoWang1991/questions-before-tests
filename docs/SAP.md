# P01 Statistical Analysis Plan (SAP) v1.0 — final, frozen

**Project:** Multidimensional assessment of disability risk in older adults across countries: the
predictive value of routine questioning, measured function and prior-wave change, and its
cross-national transportability.

**Frozen:** after data verification was complete. Not modified afterwards in response to the
direction of results.

**Pipeline:** `scripts/03_main_analysis.py` -> `scripts/04_transport.py` -> `scripts/05_sensitivity.py`
and the complementary analyses `11`–`24`, `40`–`45`. The upstream cohort-build and data-check stage
(`01_build_cohorts`, `02_datacheck`) predates this repository and is not redistributed here; the
harmonised cohorts it produced are the inputs described in the top-level README.

**Scope.** This plan was written before any analysis was run and fixes the framework at the
execution level: cohorts, outcome definitions, the six information strategies, the paired
comparison design and the metrics.

---

## 1. Research questions

Among community-dwelling older adults who are independent in basic self-care (no difficulty in any
of the five ADLs):

- **Q1.** Using routine background characteristics and shared chronic disease only (R), how reliably
  can 24-month ADL difficulty be identified?
- **Q2.** How much does adding low-cost questioning (self-rated health, depressive symptoms,
  stair-climbing difficulty: Q) improve discrimination, overall error and calibration?
- **Q3.** Does adding one measured grip strength and one cognitive assessment (separately and
  together: E1/E2/E3) still add anything **in the same people, from the same starting point, over the
  same window**?
- **Q4.** In the subgroup with genuine repeat measurements, does adding prior-wave grip and memory (L)
  beat using the current levels alone?
- **Q5.** When a model developed in one country is applied unchanged in another, how large is the
  calibration mismatch, and how much of it does local updating (intercept / slope / coefficient
  re-estimation) recover?

## 2. Design and data sources

A multi-source development and external-validation study using national ageing surveys (a prediction
model study, reported per TRIPOD+AI).

| Source | Country/region | Baseline wave (year) | Target wave (year) | Death ascertainment | Role in the main analysis |
|---|---|---|---|---|---|
| CHARLS | China | r1 (2011) | r2 (2013) | `r2iwstat` = 5/6 | three-state main analysis |
| ELSA | England | w4 (2008) | w5 (2010) | `r5iwstat` = 5/6 | three-state main analysis + strategy L |
| HRS | USA | w8 (2006) | w9 (2008) | death month − interview month <= 30 | three-state main analysis + strategy L |
| SHARE | multi-country Europe | w5 (2013) | w6 (2015) | **not ascertainable** | alive-only supplementary analysis only |

**Threshold met:** the three-state main analysis covers 3 independent surveys in 3 countries/regions
(China, England, USA), satisfying the prespecified "at least 3 sources, at least 2 regions".

**Why SHARE is downgraded (recorded in advance; not revised in the light of results):** `deceased` in
`wave_standard__share_wave*` is identically 0 in every wave (the respondent files contain survivors
only); `mort` is unrelated to appearance in later waves (45% of those with mort > 0 still appear in
w6–w8) and its units are undocumented. Death cannot be separated from loss to follow-up, so SHARE
does not enter the three-state main analysis.

## 3. Study population

**Inclusion:** baseline interview completed (`iwstat` = 1); no difficulty in any of the five ADLs; age
>= 60; baseline interview date known; 24-month state ascertainable.

**The five shared ADLs:** bathing, dressing, eating, toileting, transferring from bed to chair.
CHARLS and ELSA use the harmonised `adlfive`; HRS uses `dressa + batha + eata + beda + toilta`
(**not** `adl6a`, which includes a sixth item and reads systematically higher); SHARE uses the sum of
the five. The identity of `adlfive` and the five-item sum was verified value by value in ELSA and
SHARE.

**One row per person:** a single fixed baseline wave per source (table above). No cross-wave splicing
of "the earliest eligible wave", so that within a source the starting point and window are uniform and
strategy L's prior measurements are genuinely prior.

## 4. Outcome (24-month, three states)

Main window: baseline interview to target interview, **18–30 months**; median 24.

- **State 0, alive and independent:** target interview completed, no difficulty in any of the five ADLs
- **State 1, alive with difficulty:** target interview completed, difficulty in at least one ADL
- **State 2, dead:** died within the target window (CHARLS/ELSA target-wave `iwstat` = 5/6; HRS death
  month − interview month <= 30)

No first-onset dates are fabricated; the paper does not claim a first-event cumulative risk, only the
probability of a state at 24 months.

**Three denominators are reported separately:**

1. P(alive with difficulty) among all baseline-independent participants — **the primary risk**
2. P(dead) among all baseline-independent participants
3. P(difficulty | alive) among survivors

## 5. Information strategies (fixed in advance; not added or removed in the light of results)

| Strategy | Content | Variables |
|---|---|---|
| **R** | routine background + shared chronic disease | age, sex, education (3 levels), hypertension, diabetes, heart disease, stroke |
| **Q** | R + low-cost questioning | + self-rated health (1–5 ordered), depressive symptoms (within-source z), stair-climbing difficulty (0/1) |
| **E1** | Q + one grip measurement | + grip strength (kg, maximum of left and right, within-source z) |
| **E2** | Q + one cognitive assessment | + memory score (immediate + delayed, within-source z) |
| **E3** | Q + grip + cognition | + both |
| **L** | E3 + prior-wave information | + earlier grip, earlier memory, and their differences |

**Why Q is composed this way:** `climb` is the activity-limitation item with the most complete
coverage across the four sources (CHARLS 99.2%, ELSA 100%, HRS 91.7%, SHARE 100%); `walk` is available
in only 94.3% of CHARLS and only within the physical-examination subsample, so it is a sensitivity
variable. Falls are absent from the CHARLS harmonised file and cover only 81% of HRS, so falls are not
a main component of Q and appear only in ELSA/SHARE sensitivity analyses. The depressive instrument
differs by source (CESD-10 / CESD-8 / CESD-8 / EURO-D); no forced equating is attempted, and
within-source z-standardisation is used instead.

## 6. Model

**Multinomial logistic regression** (3 classes: independent / difficulty / dead), linear predictors,
no interactions, no regularisation. Reasons: the outcome has three states; absolute probabilities for
all three classes are required; and an interpretable linear predictor is needed for cross-national
calibration and updating. No algorithm comparison is performed — the question is the value of
information, not the choice of algorithm.

**Missing data (main analysis):** each paired comparison uses **its own complete cases**, and the
reference model is recomputed on the same rows (the paired rule). Measurements that were never
scheduled are not imputed (a person without a grip test is not given one). Continuous variables are
z-standardised within the training set.

**Paired rule (hard constraint):** the five comparisons R vs Q, Q vs E1, Q vs E2, Q vs E3 and E3 vs L
each recompute every compared model on **the same subsample, the same starting point and the same
window**. Comparing an AUC across different subsamples is forbidden.

## 7. Performance measures

- **Discrimination:** AUC for P(difficulty) (difficulty vs not, with death counted as non-difficulty);
  the survivor-restricted AUC (difficulty vs independent) is reported alongside
- **Overall error:** three-class Brier score (sum of squared probability errors / n)
- **Calibration:** calibration-in-the-large and calibration slope (logistic recalibration of the
  outcome on the logit of the predicted probability); decile calibration plots
- **Absolute risk:** predicted vs observed P(difficulty) and P(dead) by source and strategy
- **Increments:** delta AUC, delta Brier and delta calibration slope, with 95% CIs from a **paired
  bootstrap (1,000 resamples, resampling participants)**; a likelihood-ratio test for nested models is
  reported as model-level evidence

No referral threshold is prespecified, therefore **no decision-curve analysis** is performed and no
clinical-utility claim is made.

## 8. Cross-national transport and local updating

**Leave-one-source-out (LOSO) external validation:** train on two sources and test on the third
(three folds). All preprocessing and standardisation is learned in the training sources only.

For each test source and strategy, four updating levels are compared:

1. **no update** — applied unchanged
2. **intercept update** (recalibration in the large) — estimate the calibration intercept only
3. **logistic recalibration** — estimate intercept and slope
4. **coefficient re-estimation** (local revision) — a 50/50 random split of the test source; all
   coefficients re-estimated on one half and evaluated on the other, repeated 20 times and averaged

**The updating sample and the evaluation sample are strictly disjoint.**

> *Note.* All four updating levels are scored on the same held-out half of the receiving
> cohort, so the levels are paired by split. The prespecified estimator for each level is
> unchanged.

## 9. Sensitivity analyses (prespecified)

- S1: ages 50–59 included (broader applicability)
- S2: alive-only two-state outcome (difficulty vs independent, deaths removed) across the three
  sources plus SHARE
- S3: `walk` added in CHARLS; falls added in ELSA/SHARE
- S4: depressive symptoms dichotomised (any / none) instead of the continuous z score
- S5: HRS outcome defined on `adl6a` (construct sensitivity)
- S6: different windows (18–24 / 24–30 months)
- S7: multiple imputation (MICE, 20 datasets) instead of complete cases, for the E/L strategies only

## 10. Subgroups and applicability

AUC and calibration slope are reported by age (60–69 / 70–79 / >=80), sex and education (low / mid /
high). No claim of a significant subgroup difference is made unless the interaction test gives
P < 0.01 (conservative, to allow for multiple comparisons).

## 11. Stopping and reporting rules

- If a strategy shows no increment, a reversed direction, or worse calibration in a paired comparison
  it is **reported as found**, never deleted
- If a variable's construct is not fit for purpose in a given source, the reason is recorded in one
  line and the variable is removed from that strategy; the analysis is not re-run repeatedly
- If strategy L has too few participants or events, it is reported as an extension that was not
  released, and the main line is not blocked
- Sources, variables and windows are never selected on the basis of P values or the direction of the
  AUC

## 11b. Confirmatory extensions (added as v1.1, 2026-10-25)

Added after the main analysis was complete, to answer the three foreseeable weaknesses of that
analysis. **These are confirmatory extensions, not post hoc choices:** each was fixed by the analyst,
before the corresponding result was seen, as the most likely line of attack on the main analysis, and
none changes the strategy definitions of section 5, the model of section 6, the measures of section 7
or the reporting rules of section 11.

| ID | Extension | The challenge it answers | Result files |
|---|---|---|---|
| **A1** | A second wave pair per source (CHARLS r2->r3, ELSA w6->w7, HRS w10->w11), all paired comparisons re-run | "You used a single wave; the result may be an accident of that wave" | `temporal_replication.csv`, `temporal_performance.csv`, `temporal_flow.csv` |
| **A2** | Leave-one-country-out transport within SHARE (13 countries, one questionnaire, one grip protocol, one wave) | "Three countries is thin evidence for cross-national claims; the mismatch may just be population differences" | `share_multicountry.csv` |
| **A3** | Selection diagnostic for grip availability, plus a low-grip-threshold substitute for continuous kg | "The grip increment is small because only healthier people can be measured" | `selection_diagnostic.csv` |
| **A4** | Bootstrap CIs (500 resamples) for the transport metrics | "The calibration slope is a point estimate with no evidence that it is not noise" | `transport_ci.csv` |
| **A5** | Direct age–sex standardisation to a pooled standard population | "The Chinese risk is higher only because that sample is older" | `standardised_risk.csv` |

**Bonus from A1:** in the second wave pair CHARLS gains a genuine prior wave, which makes strategy L
estimable in all three countries and fills the `not estimable` cell of the main analysis.

**Interpretation limit of A2 (must be stated in the paper):** `deceased` is identically 0 in SHARE, so
A2 can only use the **alive-only two-state** outcome. It tests transport under one measurement
instrument across countries; it cannot replace the three-state main analysis. Its value is that it
attributes the mismatch to the instrument rather than to the population.

**A3–A5 do not change the main conclusions**; they change the strength and credibility of their
interpretation.

## 12. Prespecified structure of the main conclusions

1. The discrimination and calibration achieved by R (by country and pooled)
2. The increment of Q over R (paired delta AUC, delta Brier)
3. The increment of E1/E2/E3 over Q, and the marginal contribution of grip and of cognition
4. The increment of L over E3 (nested subgroup)
5. The size of the calibration mismatch under LOSO transport, and how much each updating level recovers
6. Agreement across the three countries: which increments point the same way everywhere, and which are
   source-specific
