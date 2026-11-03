# Causal / predictive structure (DAG) and the rationale for variable selection

This is a **prediction** study, not a causal-effect estimate. The DAG is used here to fix the
prediction time point, to rule out leakage of future information, to justify why particular
variables enter particular information tiers, and to explain the likely sources of cross-country
calibration mismatch. Every arrow denotes temporal availability and the direction of association;
none denotes an identified causal effect.

## 1. Temporal structure

```
                +------------- prediction time t0 (baseline interview) -------------+
                |                                                                   |
  prior ---------| information available at t0   ->  18-30 month window  -> outcome t1
  measures      | R / Q / E1 / E2 / E3                                              | independent /
  (t-2, t-4)    |                                                                   | difficulty / dead
                +-------------------------------------------------------------------+
                          strategy L additionally uses repeat measures taken before t0
```

**Hard constraint:** every variable entering a model must have been measured at or before t0. No
information arising inside the outcome window (t0, t1] may enter any model. The prior-wave measures
used by strategy L must be genuine repeat measurements taken before t0, never back-derived from the
outcome.

## 2. Variable tiers and their structure

```
        age ----------+---------------+---------------+---------------+
        sex ----------+               |               |               |
        education ----+               |               |               |
                      v               v               v
            shared chronic disease   self-rated     measured grip
            (hypertension /          health              |
             diabetes / heart            v               v
             disease / stroke)      depressive      cognition
                      |             symptoms        (memory)
                      |                 v               |
                      |            stair climbing /     |
                      |            walking difficulty   |
                      +-------+---------+---------------+
                              v
                24-month ADL difficulty / death
                              ^
                      prior-wave change (L)
```

### Rationale, tier by tier

| Tier | Variables | Why this tier | Why it cannot move to a higher tier |
|---|---|---|---|
| R (reference) | age, sex, education, four shared chronic diseases | Free in any primary-care geriatric assessment; the anchor for a "minimum implementable comparison" | — |
| Q (questionnaire) | self-rated health, depressive symptoms, stair-climbing difficulty | All obtained by asking once: no device, no trained assessor | Self-rated health and mood are subjective reports, possibly affected by unmeasured health status acting as a common cause, so they are used for **prediction only**, never for causal interpretation |
| E (one measurement) | grip strength (kg), memory score | Requires a device or a test procedure, hence extra time and staff cost; the **increment** must be demonstrated to justify it | Grip and cognition may both be affected by baseline chronic disease, and are different facets of one physical/cognitive-reserve construct, which is why E1, E2 and E3 are compared separately |
| L (prior wave) | earlier grip, earlier memory | Requires existing records, so coverage is intrinsically narrower; evaluated only in the nested subgroup with genuine repeat measurements | Using prior measurements introduces selection ("must survive and re-attend to be measured twice"), so it cannot be compared directly with R/Q in the full sample |

### Reverse arrows that must be acknowledged but cannot be handled inside a prediction framework

- **ADL difficulty -> grip / cognition.** Among people independent at baseline, subclinical disease
  may lower both grip and cognition while separately raising the risk of later difficulty. That makes
  grip and cognition **valid predictors**, but **not** modifiable causal targets. Every E-tier result
  here is read predictively.
- **Self-rated health <- depressive symptoms <- ADL difficulty.** The three are mutually causal and
  share unmeasured health. This is precisely why Q is framed as "low-cost information" rather than as
  a mechanism.
- **Death is a competing risk.** Treating death as its own third state rather than censoring avoids
  the bias of scoring death as "no difficulty", and reporting the survivor-conditional risk alongside
  keeps the denominator explicit.

## 3. Where cross-country calibration mismatch comes from, structurally

The mismatch seen in leave-one-source-out validation (CHARLS over-predicted 23.8% vs 16.4% observed;
HRS over-predicted 13.7% vs 9.6%; ELSA under-predicted 8.8% vs 10.5%) has three possible paths on the
DAG:

1. **A different baseline risk level** (intercept): the two-year incidence of ADL difficulty among
   people aged 60 or over genuinely differs between the three populations (CHARLS 16.2%, ELSA 10.7%,
   HRS 10.3%).
2. **A different distribution of predictors** (slope): the same construct is measured on different
   scales, for example different depressive-symptom instruments (CESD-10 / CESD-8 / EURO-D), or
   different versions of the self-rated-health question.
3. **A different strength of the predictor-outcome relationship** (coefficients): the same variable
   may predict differently in different cultures and health systems.

The four updating levels map onto these paths: intercept updating addresses path 1 only; logistic
recalibration addresses parts of paths 1 and 2; coefficient re-estimation addresses paths 1, 2 and 3.
The finding that coefficient re-estimation improves the AUC far more than intercept updating does
(for example HRS, strategy Q: 0.653 -> 0.748) indicates that **path 3 is the dominant source** — the
same questions carry a different predictive strength in different countries — rather than a simple
difference in baseline rates.

> **Note on the numbers in section 3.** They are quoted from the first-pass transport run of
> 2026-10-25 and are preserved for provenance. The final, reported transport estimates are in
> `results/transport.csv`, `results/transport_ci.csv` and Table 2b of the paper; the argument
> made here is unchanged between the two runs.

## 4. Variables deliberately kept out of the models

| Variable | Status | Reason |
|---|---|---|
| Fall history | sensitivity only | Absent from the CHARLS harmonised file; 81% coverage in HRS. Not comparable across the four sources, so it cannot enter the main Q tier |
| Walking difficulty | sensitivity only | Available in 94% of CHARLS and only within the physical-examination subsample; highly collinear with stair climbing |
| IADL | not used | Overlaps the main outcome (ADL) conceptually and would dilute the clinical question, which concerns early identification among people who are ADL-independent |
| BMI, smoking, activity level | not used | Prespecified as secondary candidates; adding them does not change the core question and increases missingness |
| Pain, marital status / living alone | not used | As above; also defined non-comparably across the four sources |
| Biomarkers, polypharmacy | not used | Inconsistent with the low-cost primary-care setting the study addresses |

All of these exclusions were fixed **before** the analysis (see section 5 of `SAP.md`); none was
added after seeing results.
