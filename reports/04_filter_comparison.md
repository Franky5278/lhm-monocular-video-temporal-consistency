# The filter we chose is not the best one — matched-fidelity comparison

This note exists because a reviewer's first question about any stabilisation result is
"compared to what?". The answer, measured on the same raw SMPL-X motion and with the
same metric definitions, is uncomfortable: **the causal EMA used by the pipeline is
Pareto-dominated by a zero-phase filter.**

## Setup

- Input: the raw fitted motion of ZJU-313 (60 frames, 30 fps) — the same array every
  method starts from.
- Metric: pose jitter = mean per-frame second difference of the 6D rotation vector;
  deviation = mean L2 distance from the raw motion, same representation.
- Baselines: centred moving average (no group delay), One-Euro filter
  (Casiez et al. 2012), and a zero-phase EMA (the pipeline's own operator applied
  forward then backward).
- Fairness: every method is swept over its strength, so the comparison is a
  Pareto front, not a single hand-picked point.

## Result

| method | deviation | pose jitter | jitter vs pipeline |
|---|---|---|---|
| raw SMPL-X motion | 0.000 | 0.47747 | 16.9× |
| moving average w=9 | 0.2656 | 0.05060 | 1.79× |
| moving average w=15 | 0.3189 | 0.02969 | 1.05× |
| **zero-phase EMA α=0.2** | **0.3224** | **0.01636** | **0.58×** |
| pipeline (causal EMA + trans-velocity EMA) | 0.4243 | 0.02826 | 1.00× |
| moving average w=25 | 0.4408 | 0.01703 | 0.60× |
| **zero-phase EMA α=0.1** | **0.5023** | **0.00569** | **0.20×** |

Interpolating the zero-phase front at the pipeline's deviation (0.4243) gives a jitter
of ≈0.010, i.e. **2.7× lower at the same fidelity cost**. At the pipeline's own
deviation, a plain centred moving average is also better (0.017 vs 0.028).

![Matched-fidelity filter comparison](figures/fig_filter_pareto.png)

## Why this happens (and why it is fixable)

A causal EMA has group delay: smoothing reduces high-frequency jitter but lags the
motion, and the lag itself shows up as deviation from the raw trajectory. The
pipeline therefore pays twice — once for the smoothing, once for the lag. Rendering is
an offline batch job, so causality buys nothing; applying the same recursion forward
and backward cancels the delay at no cost. That is the whole trick, and it is why the
green curve sits below the red star.

## What this changes about the paper's claims

- The reduction over the Gaussian-prefilter baseline (0.117 → 0.028, 4.1×) is real
  and reproducible. It stays.
- The claim that the *specific* causal EMA is a good design does **not** survive this
  comparison, and is withdrawn.
- The contribution that survives is: (a) localising instability to the monocular
  driver, (b) separating pose and translation channels (they remain orthogonal — the
  translation EMA adds exactly zero pose deviation), (c) the zero-phase design rule
  quantified here, and (d) the evaluation protocol that made this defect visible in
  the first place.

## Limits of this comparison

- Motion space only. Re-rendering the zero-phase variant needs a GPU; until then the
  claim is about the motion metrics the paper reports, not about pixels.
- One sequence (ZJU-313). The ordering is a property of the operators, so it should
  transfer, but it has not been measured on the 390 segments.
- Deviation is measured against the *estimated* raw motion, not against ground-truth
  motion capture. A filter that lowers jitter while departing from the estimate is
  not necessarily closer to reality; this metric cannot separate the two.
