# SHC 13 ring 4: why the fitted temperature changed

Reproducible line-by-line analysis of the two Excel tables in
`data/NGC253_HR/Results/Tables/`, using `HC3N_line_dict_builder` with a
0.020 × 0.022 arcsec beam and `Jselect=24` (the existing ±2 selection).
The script `python scripts/analyze_SHC13_rotational.py` regenerates the CSV and
figure in this directory. Numerical calculations below use the 12 selected
lines and the no-bootstrap fits. Source repository baseline: `031528b`;
corrected implementation: `5072041`.

## 1. Exact cause of the 462 K → 608 K change

The ordinate is `y=log10(Nu/gu)`. If the measured linear population is
`N ± sigma_N`, first-order propagation gives

` sigma_y = sigma_N / (N ln 10) `.

The old code stored `log10(sigma_N)` in a field named `logNu/gu_err`, then
calculated `log10(sigma_N) / [ln(10) log10(N)]` and passed this to `curve_fit`
as the y uncertainty. Logging a dimensional column-density error is not an
uncertainty transformation. In this dataset it made all 12 errors nearly
uniform, **0.375–0.426 dex**, so each line received about **7–9%** of the
relative fit weight regardless of its actual signal-to-noise ratio.

The corrected errors span **0.0084–0.266 dex**. In particular, the weakest
`v6=2, J=24` line at 1589 K goes from 7.0% of fit weight to **0.036%**;
`v7=1, J=26` gets **35.5%**, and `v=0, J=26` gets **21.3%**. The two
high-weight lines favor a flatter population-versus-energy slope. The old
slope was `-9.39969e-4 dex/K`, the new slope `-7.14853e-4 dex/K`. Since
`T = -1/(slope ln 10)`, this corresponds to **462.0 K** and **607.5 K**.
The previously reported bootstrap result near 455 K followed the same
near-equal weighting.

The intercept falls from 13.0283 to 12.9050. With this code's
`Ntot = [kT/(h×4.55 GHz)]×10^intercept`, a 31.5% rise in T is nearly
cancelled by a 24.7% fall in `10^intercept`. The column estimates remain
`2.258e16` and `2.236e16 cm^-2` respectively. The corrected column error
also includes the slope/intercept covariance and the missing scale factors;
the old `1.76e12 cm^-2` is **not** directly comparable to the corrected
formal `2.77e14 cm^-2`.

## 2. Each line's contribution to the corrected chi-square

`z = [observed log10(Nu/gu) - predicted log10(Nu/gu)] / sigma_y`.
For the single-slope fit, `chi2 = sum(z²) = 252.94` with 10 degrees of
freedom; reduced chi-square is **25.29**. Positive z means more population
than the fitted line predicts. The table shows direct residuals from the
fit to all 12 lines; deleting one line causes the fit to move, so its z²
is not the same as the reduction in chi-square after refitting.

| Vibrational state | J | Frequency (GHz) | Eu/k (K) | sigma_y (dex) | z | z² | Weight | T without line (K) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v=0 | 26 | 236.513 | 153 | 0.0109 | -8.27 | 68.42 | 21.3% | **501** |
| v=0 | 24 | 218.325 | 131 | 0.0184 | -0.22 | 0.05 | 7.4% | 606 |
| v7=1 | 24 | 219.174 | 452 | 0.0170 | +8.73 | 76.13 | 8.7% | 613 |
| v7=1 | 26 | 237.432 | 475 | 0.0084 | +4.32 | 18.63 | 35.5% | 611 |
| v7=2 | 24 | 219.675 | 774 | 0.0297 | +2.14 | 4.56 | 2.9% | 602 |
| v7=2 | 26 | 237.969 | 796 | 0.0154 | +0.85 | 0.73 | 10.6% | 602 |
| v6=1 | 24 | 218.683 | 849 | 0.0295 | +3.20 | 10.23 | 2.9% | 597 |
| v6=1 | 26 | 236.900 | 871 | 0.0234 | -2.93 | 8.59 | 4.6% | 621 |
| v5=1/v7=3 | 26 | 236.661 | 1107 | 0.0323 | -3.73 | 13.89 | 2.4% | 628 |
| v6=v7=1 | 26 | 237.815 | 1192 | 0.0280 | -4.72 | 22.32 | 3.2% | 647 |
| v4=1 | 26 | 236.184 | 1399 | 0.0711 | -5.30 | 28.11 | 0.5% | 626 |
| v6=2 | 24 | 219.024 | 1589 | 0.2659 | -1.14 | 1.29 | 0.04% | 609 |

The first and third rows account for **57.1%** of the total chi-square.
The `v=0, J=26` line has the greatest effect on temperature: deleting it
changes the fit by **-106 K**, to 501 K. Deleting both `v=0` lines yields
451 K, but the remaining single-slope fit still has reduced chi-square
**8.94** (8 degrees of freedom). This is not a problem solved by discarding
one obvious outlier.

## 3. A physical pattern in the paired transitions

Four states have both J=24 and J=26 lines. Within each state, the upper-state
energy increases by about 22.3 K. If both lines follow an optically thin,
common-excitation Boltzmann population, each pair estimates a rotational
slope through `Delta log10(Nu/gu) / Delta(Eu/k)`. Their **descriptive**
rotational temperatures are:

| State | Pair temperature | Difference from the 608 K single-slope prediction |
|---|---:|---:|
| v=0 | 95 K | 4.0 standard errors in the pair contrast |
| v7=1 | 76 K | 5.9 standard errors |
| v7=2 | 147 K | 1.5 standard errors |
| v6=1 | 54 K | 4.3 standard errors |

These are estimates from **only two transitions per state**, with a small
22 K energy separation. Their values can be distorted by optical depth,
line blending, non-LTE effects, errors shared across spectral windows, and
population or line-strength conventions. A fitted J effect is confounded
with observing frequency here: all J=24 lines are around 218–220 GHz and
all J=26 lines around 236–238 GHz. The pairs demonstrate that one slope
cannot describe both between-state and within-state behavior at the
quoted statistical precision; they do not establish a kinetic temperature.

For illustration only, weighted algebraic models give:

| Model | Parameters | Temperature from vibrational slope | Reduced chi-square |
|---|---:|---:|---:|
| Single slope over total Eu | 2 | 608 K | 25.29 (10 dof) |
| Separate `E_vib` and approximate `E_rot` slopes | 3 | 619 K (`T_rot` ≈ 83 K) | 20.55 (9 dof) |
| Two slopes plus independent `v=0` intercept | 4 | **458 K** (`T_rot` ≈ 81 K) | **2.69** (8 dof) |

The last model has a `v=0` offset of **-0.214 dex** relative to the excited
state trend, equivalent to a factor **0.61** in apparent population at the
same modeled energy. It is still statistically strained (`chi2=21.51`, 8 dof,
`p≈0.0059` under independent Gaussian errors). The 458 K resemblance to the
old 462 K fit is **not** evidence that the old error calculation was valid;
those values arise from different models. In the augmented model, `v4=1`
and `v7=2, J=26` remain about 2.9 and 2.8 stated errors from prediction.

## 4. Q3: physically justified rotational and vibrational model

Yes. For optically thin apparent upper-state populations, the implemented
model is

`log10(Nu/gu) = a - Evib/[ln(10) Tvib] - Erot/[ln(10) Trot] + delta_state`,

with `Erot/k = h B J(J+1)/k` and `B = 4.55 GHz`; `Evib = Eu - Erot`.
The catalog `gup` values for the selected transitions are 49 and 53, equal to
`2J+1`, so the default treats each selected l-sublevel as an individual
resolved state. If a population instead sums equivalent unresolved
components, the API accepts their multiplicity and divides that measurement
before fitting. It also accepts a known population correction such as
`tau/[1-exp(-tau)]` when opacity has been obtained independently.

For this physical fit, `Nu/gu` is recalculated with each transition's catalog
Einstein A coefficient. The Q1 reproduction retains the original rigid-rotor
A approximation. Most catalog corrections are small, but the selected
`v6=1, J=24` coefficient changes its population by about 0.10 dex and the
`v=0, J=24` coefficient by about 0.052 dex, so the catalog values matter for
the shallow within-state rotational slopes.

The generalized least-squares covariance contains a 5% multiplicative
calibration term shared by every line in the same observing band. Here that
term is also shared by J: all J=24 lines are near 219 GHz and all J=26 lines
are near 237 GHz. It therefore leaves the best-fit temperatures unchanged but
increases the formal `Trot` uncertainty from about 16 K to **43 K**.

Two versions are reported:

| Model | Tvib | Trot | v=0 apparent-population offset | Extra scatter | Fit statistic |
|---|---:|---:|---:|---:|---:|
| Stated errors + 5% band calibration | **449.9 ± 12.1 K** | **111.7 ± 42.6 K** | `-0.233 ± 0.017 dex` | none | chi2/dof = 26.84/8, p=0.00075 |
| Robust descriptive model | **424.8 ± 18.3 K** | **121.2 ± 56.5 K** | `-0.275 ± 0.035 dex` | **0.0236 dex** | chi2/dof = 10.85/7, p=0.145 |

The robust model profiles a common unexplained line-to-line scatter. Its
ground-state offset corresponds to multiplying the apparent v=0 populations
by **1.88 ± 0.15** to align them with the excited-state trend. This factor is
an empirical discrepancy, not a measured optical-depth correction. Blending,
source-filling differences and non-LTE effects can produce similar offsets.

This answers the single-temperature question directly. A straight line
against total `Eu` imposes **Trot = Tvib**. That is a special LTE condition,
not something a population diagram guarantees. The strong structured
residuals reject that condition for these measurements at their stated
precision. A single fit can still be quoted as an *effective excitation
temperature*, but it is not a unique physical temperature of the whole
diagram. The two-temperature result also explains why the vibrational
temperature returns near the original estimate: after the low rotational
temperature and v=0 discrepancy are represented explicitly, the between-state
slope is about 425–450 K. Its closeness to the old 462 K value is partly
coincidental because the old calculation used invalid uncertainties.

The model is implemented in `utiles/excitation.py`. The script writes
`SHC13_two_temperature_summary.csv`, line-level diagnostics, and
`SHC13_two_temperature_model.png`. It deliberately does not turn the fitted
intercept into total HC3N column density, because that requires consistent
rotational and vibrational partition functions and a specified emitting area.

## 5. Sensitivity and next checks

Giving all lines an error floor changes the *single-slope* estimate: 0.03,
0.05, and 0.10 dex floors yield about **525, 497, and 469 K**. The 0.10 dex
floor brings reduced chi-square to 1.37, but this is a sensitivity scenario,
not a measured calibration uncertainty. A meaningful covariance matrix
would be preferable to inventing independent floors. The 14.5 K error on
608 K is conditional on the single-slope model and supplied independent
measurement errors; the residual pattern contradicts those assumptions.

Priorities for diagnosing cause from the underlying cubes and fits:

1. Revisit the integrated flux and uncertainties of the strongest discrepant
   lines: `v=0, J=26` (52.63 ± 1.32 mJy km/s), `v7=1, J=24`
   (40.93 ± 1.61), and `v7=1, J=26` (41.96 ± 0.81), all in ring 4 in
   `*_beam_orig` workbook columns. Check baseline, blend subtraction,
   continuum treatment and velocity integration.
2. Verify line-to-line covariance from a shared spatial extraction,
   continuum subtraction, spectral windows and absolute calibration.
   Statistical errors in the Excel workbook alone do not encode it.
3. Confirm the matched beam and source-filling convention at each frequency.
   The same nominal 0.020 × 0.022 arcsec beam is passed for every line; the
   workbook also contains separate `_beam_345` alternatives, which were
   deliberately excluded by the current builder.
4. Check optical depth of `v=0` independently and verify the degeneracy and
   l-sublevel assignments for vibrational states, especially v7=1. An offset
   in apparent population alone cannot distinguish opacity from excitation
   or calibration.
5. Specify rotational and vibrational partition functions before labeling
   `Q_rot(T_vib)×10^intercept` as a total HC3N column, and compare with the
   existing LTE and non-LTE modelling of these rings.

The associated publication studies both J=24–23 and J=26–25 and derives
vibrational temperatures through 2D-LTE and non-local transfer analysis.
This spreadsheet diagnostic does **not** independently assess those models
or revise a published temperature.

Primary context: Rico-Villas et al., *On the thermal structure of the
proto-Super Star Cluster 13 in NGC 253*, [arXiv:2208.01941](https://arxiv.org/abs/2208.01941).
