# Statistical Interpretation (E13)

## Method
For each of the six method pairs, the per-recording clean-channel SNR (n=2620, E1) was
compared with a **paired t-test** (same recording serves as its own control, so the
between-recording scale is removed). Cohen's `d_z` (standardized paired difference) was
computed as `mean(diff) / sd(diff)` from the per-recording SNR difference (source:
`results/paper_analysis/statistics/E13_effect_sizes.csv`).

## Results

| Pair                 | mean ΔSNR (dB) | t      | p          | d_z    |
|----------------------|----------------|--------|------------|--------|
| joint vs semantic    | +2.07          | 203.0  | <1e-16     |  3.97  |
| joint vs spectral    | -1.79          | -126.4 | <1e-16     | -2.47  |
| joint vs random      | +1.12          | 71.9   | <1e-16     |  1.41  |
| semantic vs spectral | -3.86          | -213.9 | <1e-16     | -4.18  |
| random vs semantic   | +0.95          | 81.2   | <1e-16     |  1.59  |
| random vs spectral   | -2.91          | -193.1 | <1e-16     | -3.77  |

## How to report p
The canonical E13 json stores `"p": 0.0`. This is **numerical underflow**, not a
mathematically exact zero. Every paired difference with |t| >= 71.9 is far beyond the
double-precision tail threshold, so in the manuscript the p-value must be reported as:

> p < machine precision (< 1e-16; scipy double-precision tail underflow).

Do **NOT** state "p = 0".

## Practical significance caveat
Statistical significance with n=2620 detects even a 0.1 dB shift. The mean differences
here are 0.95-3.86 dB, so they are both significant and of comparable magnitude to the
cross-recording standard deviation (sigma~1.4-4.1 dB of the paired difference; see table).
However, SNR is an objective proxy of time-domain distortion; **no perceptual metric
(MUSHRA/PESQ) was measured**, so "statistically significant" must not be promoted to
"perceptually superior" or "practically better quality".

## Direction read
- Joint > semantic (+2.07 dB): adding spectral suitability on top of semantic selection
  increases clean SNR.
- Joint < spectral (-1.79 dB): spectral-only selection has the highest clean SNR.
- Joint > random (+1.12 dB): the joint mechanism beats random coordinate placement.
- Semantic-only has the **lowest** clean SNR of the four strategies (-3.86 dB vs spectral).
