# Scientific review — 26 September 2026

## Scope and status

Reviewed the reusable equations in `u_conversion.py`, `utiles_alma.py`,
`utiles_physical.py`, and `utiles_molecules.py`, with targeted numerical checks.
This is not a validation of every FITS operation, non-LTE model, plotted result,
or published analysis. Baseline for this review: commit `031528b`.

The fixes below are supported by unit algebra, analytical derivatives,
independent Astropy calculations, and synthetic examples. Run:

```bash
python -m unittest discover -s tests -v
```

All 22 tests pass on Python 3.12, NumPy 2.3.5, SciPy 1.17.0 and Astropy 8.0.1.
These include the six tests from the first code review.

## Corrected issues

| Area | Finding | Correction / consequence |
|---|---|---|
| ALMA angular scales | GHz was used as Hz; the baseline in km was used as meters. | Explicit units in `FOV_telescope` and `cellsize_ALMA`. Missing scale factors were 10^9 and 10^12 respectively, with a small additional change from using exact c. |
| ALMA primary beam | A bare antenna diameter left a length unit in an angular result. | Interpret bare diameters as meters in `HPBW`. |
| Gaussian brightness errors | The area term used S/sigma_area^2 instead of S*sigma_area/area^2. Some variants raised a units error; zero errors could give infinity. | All three variants share derivative-based propagation and support arrays. Frequency is exact and errors are independent. |
| Flux integration | Negative RA pixel increments could make positive integrated flux negative. | Use the absolute pixel area in the two standard Jy/beam conversion helpers; replace a missing beam-area helper in Jybeam_to_Jypx. |
| Flux conversion options | A supplied positional False became the nonempty tuple `(False,)`, activating wavelength mode. | An explicit `wavelength=False` parameter; select units before creating a Quantity. |
| Luminosity inverse | Quantity distances acquired Mpc twice in `flux_to_lum(..., z=False)`. | Normalize with `Quantity(distance, Mpc)`. |
| Planck wavelength function | Bare inputs failed and the radiance convention was undocumented. | Inputs are meters and K, output W/(m2 m sr); use expm1. |
| Escape velocity | Division by 1000 changed the value while retaining m/s. | Return a correctly converted km/s Quantity. Calling `.to(km/s)` now gives the intended speed. |
| Free-fall time | Removing G's units produced a non-time dimension. | Keep the gravitational constant as a Quantity and convert to years. |
| McCrady virial mass | Multiplication by a kg-to-Msun Quantity left an extra mass dimension. | Convert the computed mass using `.to(Msun)`; the existing coefficient of 10 is retained. |
| Dust blackbody | A class had been aliased as the old blackbody function; code accessed `.value` on a model. | Instantiate `BlackBody(temperature=...)`, evaluate at frequency and explicitly convert to CGS radiance. |
| Dust lower mass bound | Its numeric solar-mass conversion retained a gram unit. | Return the scalar in solar masses, matching the upper bound in this helper. |
| Rotational line selection | `Jselect` was overwritten by 24. | Honor the supplied J, retaining the existing J +/- 2 selection window. |
| Logarithmic errors | The builder stored log10(sigma_N); fitting divided this by log10(N). | Use sigma_log10(N) = sigma_N / (N ln(10)). Recompute from linear fields for older dictionaries. |
| Rotational N uncertainty | Factors of 10^b and Q were missing; slope/intercept covariance was discarded. | Apply the full Jacobian and covariance. |
| Fit statistics | Relative-weight covariance could obscure the meaning of supplied measurement errors. | Use absolute_sigma=True and expose chi2, degrees of freedom and reduced_chi2. |
| Rotational bootstrap | It fitted without the main fit's weights and mixed residual scatter with measurement errors. | A weighted parametric bootstrap, 1000 draws around the fitted model, with covariance propagation. It quantifies the stated measurement model, not intrinsic scatter. |
| Thin optical-depth limit | tau/(1-exp(-tau)) returned 0/0 at tau=0. | Its limit is one; use expm1 for small tau. |
| Output labels / compatibility | Column density was labelled cm^-1; NumPy's removed np.float stopped the fit output. | Labels use cm^-2; the formatting helper uses float. |

## Equations checked

For a Gaussian beam,

- Omega = pi theta_major theta_minor / (4 ln 2).
- T_B = c^2 S_nu / (2 k nu^2 Omega), the Rayleigh-Jeans brightness temperature.
- At fixed frequency and independent errors,
  sigma_T^2 = (T/S sigma_S)^2 + (T/theta_major sigma_major)^2
  + (T/theta_minor sigma_minor)^2. The implementation uses a derivative form
  that remains finite at zero flux.

For y = log10(N_u/g_u) = m (E_u/k) + b, with Q = kT/(hB),

- T = -1/(m ln 10), N = Q(T) 10^b.
- dN/dm = -N/m; dN/db = N ln 10.
- Var(N) = (dN/dm)^2 Var(m) + (dN/db)^2 Var(b)
  + 2 (dN/dm)(dN/db) Cov(m,b).

The Jacobian is checked against finite differences. The optically thin
column-density formula is independently checked using the SI dipole Einstein
coefficient, including epsilon_0. Agreement is within 0.2%, consistent with the
legacy rounded brightness-temperature prefactor. The Planck spectrum is checked
both against Astropy and by integrating pi B_nu to sigma_SB T^4.

## Bundled SHC 13 example: impact

Inputs: `SLIM_HC3N_info.xlsx` and `SHC_13_flujos_hc3n_python.xlsx` under
`data/NGC253_HR/Results/Tables/`, ring 4 selected by the builder, beam
0.020 x 0.022 arcsec, Jselect=24, bootstrap=False. Both fits use 12 lines.

| Quantity | Baseline implementation | Corrected implementation |
|---|---:|---:|
| Excitation temperature (K) | 462.03 | 607.53 |
| Formal temperature error (K) | 38.85 | 14.54 |
| Column density (cm^-2) | 2.2583e16 | 2.2356e16 |
| Formal column-density error (cm^-2) | 1.7578e12 | 2.7714e14 |

The corrected reduced chi-square is **25.29 for 10 degrees of freedom**.
The formal uncertainty is conditional on independent, correctly sized errors
and a suitable single-temperature linear model. The observed residuals are much
larger than that model predicts. A smaller formal error is not evidence of a
more physically precise result. Possible causes to investigate include model
inadequacy, blending, opacity, underestimated errors and correlated calibration.
The comparison does not establish which cause applies, nor whether the code
used for a publication was identical to this repository snapshot.

To reproduce the corrected fit from the repository root:

```python
from astrothesispy.utiles import utiles_molecules as m
lines = m.HC3N_line_dict_builder(
    'SHC_13', 'data/NGC253_HR/Results/',
    'data/NGC253_HR/Results/Tables/SLIM_HC3N_info.xlsx', .020, .022)
table, fits = m.rotational_diagram(lines, Jselect=24, bootstrap=False)
print(fits['NoBootstrap'])
```

A line-level follow-up with weights, residuals, leave-one-line-out fits and
alternative excitation models is in `analysis/SHC13_FIT_DIAGNOSIS.md`.

## Remaining scientific issues requiring an explicit convention

1. **Rotational versus vibrational partition function.** The line selection spans
   vibrational states, while Q=kT/(hB) contains only rotational states. Before
   interpreting N as the total HC3N column, specify vibrational degeneracies,
   the intended partition function and whether T_rot equals T_vib.
2. **Virial definitions.** `virial_parameter` strips velocity units and its
   argument is named FWHM, although a usual 5 sigma^2 R/(GM) expression uses
   one-dimensional dispersion. `virial_mass` also needs an explicit radius /
   diameter and linewidth convention. These functions were not scientifically
   validated or silently assigned a new convention.
3. **Dust mass convention.** `gas_mass_from_dust` names its input Jy/beam but uses
   a flux-based formula without a beam area. It also lacks the dust-to-gas factor
   present in the Leroy variants. The restored blackbody call does not establish
   whether its result represents dust mass or gas mass. The slab helpers assume
   filling factor one, no background term, and require I_nu < B_nu(T).
4. **Other molecular formulas.** `tau_HC3N`, `vibexc_HC3N`,
   `transition_citical_density` and `Trotational` mix bare values and unit-bearing
   constants or have ambiguous J / dipole / energy conventions. They remain
   unvalidated. `HC3N_levelpopulation` mixes a log population with a linear error
   and uses Jup+1; its intended quantum-number convention needs resolution.
5. **Gaussian source-size inversion.** `sourcesize_from_fit` has an ambiguous
   size argument and is not the general Gaussian beam-dilution inversion.
6. **Bandwidth versus line FWHM.** `wm2_to_jy` and `jy_to_wm2` retain their legacy
   equivalent rectangular bandwidth convention. A Gaussian line with peak S
   and frequency FWHM has area S*FWHM*sqrt(pi/(4 ln 2)); a wavelength interval
   is not converted to a frequency interval by c/delta_lambda.
7. **Mixed return types.** `lin_size` still returns a length Quantity and a bare
   numerical error when D_err is supplied. The numeric `_pd` helper differs.
   Standardizing these interfaces requires a documented compatibility change.
8. **Legacy Madcuba estimator.** `Jybeam_to_Jy_madcuba` calls a missing beam
   helper and squares the supplied flux; its intended estimator needs clarification.
9. **Coverage.** Fits/cube moments, non-LTE transfer, empirical calibrations,
   every input domain, and the complete publication figures remain unaudited.

## Compatibility notes

Scientific outputs and formal errors change where the equations were wrong.
`planckwave` now explicitly returns per-steradian radiance. The Gaussian aliases
share one implementation. The legacy fit dictionary `pcov` still holds the
parameter standard errors to avoid breaking existing callers; the new
`covariance` key holds the actual 2x2 matrix. The bootstrap uses NumPy's global
random state, so an explicit `np.random.seed(...)` remains reproducible.
CRLF endings in the legacy modules were preserved to keep diffs readable.

## Primary references

- NRAO, brightness temperature and Gaussian beam area:
  https://science.nrao.edu/facilities/vla/proposing/TBconv
- Astropy physical models, including BlackBody:
  https://docs.astropy.org/en/stable/modeling/physical_models.html
- Astropy unit and spectral-density equivalencies:
  https://docs.astropy.org/en/stable/units/equivalencies.html
