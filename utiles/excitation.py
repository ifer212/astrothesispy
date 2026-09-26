"""Rotational and vibrational excitation fits for molecular line populations.

The model in this module separates the upper-state energy into an approximate
rigid-rotor contribution and a vibrational contribution::

    log10(N_u/g_u) = a - E_vib/(ln(10) T_vib)
                       - E_rot/(ln(10) T_rot) + state offsets.

It is intended for population-diagram measurements for which ``g_u`` already
contains the rotational degeneracy.  It does not perform radiative transfer;
opacity or unresolved-component corrections must be supplied by the caller.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from astropy import constants as const, units as u
from scipy.optimize import minimize_scalar
from scipy.stats import chi2


_LN10 = np.log(10.0)


@dataclass(frozen=True)
class TwoTemperatureFit:
    """Result of a generalized least-squares two-temperature fit.

    Parameter errors are conditional on the fitted intrinsic scatter.  The
    state offsets use the sign convention ``observed - common-state model``;
    ``10**(-offset)`` is therefore the factor required to bring that state's
    apparent population onto the common trend.
    """

    coefficients: np.ndarray
    covariance: np.ndarray
    parameter_names: tuple[str, ...]
    vibrational_temperature_K: float
    vibrational_temperature_error_K: float
    rotational_temperature_K: float
    rotational_temperature_error_K: float
    state_offsets_dex: dict[str, float]
    state_offset_errors_dex: dict[str, float]
    state_population_correction_factors: dict[str, float]
    intrinsic_scatter_dex: float
    chi2: float
    dof: int
    reduced_chi2: float
    p_value: float
    log_likelihood: float
    aic: float
    bic: float
    diagnostics: pd.DataFrame


def _as_vector(value, data, name, default):
    if value is None:
        return np.full(len(data), default, dtype=float)
    if isinstance(value, str):
        if value not in data:
            raise KeyError(f"Column {value!r} requested for {name} is absent")
        value = data[value]
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(len(data), float(array))
    if array.shape != (len(data),):
        raise ValueError(f"{name} must be scalar or have one value per line")
    return array


def _temperature_from_slope(slope, variance):
    if slope >= 0:
        return np.nan, np.nan
    temperature = -1.0 / (_LN10 * slope)
    derivative = 1.0 / (_LN10 * slope**2)
    return temperature, abs(derivative) * np.sqrt(max(variance, 0.0))


def fit_two_temperature_excitation(
    data: pd.DataFrame,
    rotational_constant=4.55 * u.GHz,
    *,
    calibration_groups: Sequence | str | None = None,
    calibration_fraction: float | Mapping = 0.0,
    offset_states: Sequence[str] = (),
    component_multiplicity: Sequence[float] | str | float | None = None,
    population_correction: Sequence[float] | str | float | None = None,
    fit_intrinsic_scatter: bool = False,
    max_intrinsic_scatter_dex: float = 0.5,
) -> TwoTemperatureFit:
    """Fit distinct rotational and vibrational excitation temperatures.

    Parameters
    ----------
    data : pandas.DataFrame
        One row per measured line. Required columns are ``vib_state``, ``qn1``
        (upper J), ``EUP_K`` and ``logNu/gu``. Supply either
        ``logNu/gu_relerr`` or both ``Nu/gu`` and ``Nu/gu_err``.
    rotational_constant : `~astropy.units.Quantity`
        Rigid-rotor B constant. The default is the HC3N approximation used by
        this project. ``E_rot/k = h B J(J+1)/k``.
    calibration_groups : sequence, column name, optional
        Lines in the same group share a fully correlated multiplicative
        calibration error. A separate group per observing band is typical.
    calibration_fraction : float or mapping
        One-sigma fractional calibration error for every group, or a mapping
        from group labels to errors. Converted to dex by first-order error
        propagation.
    offset_states : sequence of str
        Vibrational states allowed an empirical population offset. This can
        describe a discrepant ground state but is not an opacity measurement.
    component_multiplicity : sequence, column name, scalar, optional
        Number of equivalent unresolved components summed by each population
        measurement. The default is one, appropriate for a resolved individual
        l-sublevel. The fitted ordinate is divided by this multiplicity.
    population_correction : sequence, column name, scalar, optional
        Known multiplicative correction to the apparent upper-state
        population, such as ``tau/(1-exp(-tau))``. The default is one.
    fit_intrinsic_scatter : bool
        Profile a common extra scatter in dex. This is a robustness term for
        unmodelled line-to-line variation, not a physical excitation process.

    Notes
    -----
    The two temperatures are identifiable only when the input contains both
    multiple vibrational energies and multiple rotational energies. If J and
    observing band coincide, correlated band calibration directly weakens the
    rotational-temperature constraint. Partition functions are not needed for
    the slopes, but are required to turn the intercept into a total column.
    """
    required = {"vib_state", "qn1", "EUP_K", "logNu/gu"}
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    if len(data) < 4:
        raise ValueError("At least four line measurements are required")

    frame = data.reset_index(drop=True).copy()
    y_observed = frame["logNu/gu"].to_numpy(dtype=float)
    if "logNu/gu_relerr" in frame:
        sigma_stat = frame["logNu/gu_relerr"].to_numpy(dtype=float)
    elif {"Nu/gu", "Nu/gu_err"}.issubset(frame.columns):
        sigma_stat = (frame["Nu/gu_err"] / frame["Nu/gu"] / _LN10).to_numpy(dtype=float)
    else:
        raise KeyError("Supply logNu/gu_relerr or both Nu/gu and Nu/gu_err")

    multiplicity = _as_vector(component_multiplicity, frame, "component_multiplicity", 1.0)
    correction = _as_vector(population_correction, frame, "population_correction", 1.0)
    numeric = np.column_stack([y_observed, sigma_stat, multiplicity, correction])
    if not np.all(np.isfinite(numeric)):
        raise ValueError("Populations, errors and corrections must be finite")
    if np.any(sigma_stat <= 0) or np.any(multiplicity <= 0) or np.any(correction <= 0):
        raise ValueError("Errors, multiplicities and corrections must be positive")
    y = y_observed - np.log10(multiplicity) + np.log10(correction)

    B = u.Quantity(rotational_constant)
    if not B.unit.is_equivalent(u.Hz):
        raise u.UnitConversionError("rotational_constant must have frequency units")
    J = frame["qn1"].to_numpy(dtype=float)
    energy_total = frame["EUP_K"].to_numpy(dtype=float)
    energy_rot = (const.h * B / const.k_B).to_value(u.K) * J * (J + 1)
    energy_vib = energy_total - energy_rot
    if not np.all(np.isfinite(np.column_stack([J, energy_total, energy_vib]))):
        raise ValueError("Quantum numbers and energies must be finite")

    states = frame["vib_state"].astype(str).to_numpy()
    offset_states = tuple(str(state) for state in offset_states)
    missing_states = set(offset_states).difference(states)
    if missing_states:
        raise ValueError(f"Offset states absent from data: {sorted(missing_states)}")
    columns = [np.ones(len(frame)), energy_vib, energy_rot]
    names = ["intercept", "vibrational_slope_dex_per_K", "rotational_slope_dex_per_K"]
    for state in offset_states:
        columns.append((states == state).astype(float))
        names.append(f"offset:{state}")
    design = np.column_stack(columns)
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("The requested temperatures and offsets are not identifiable from these lines")
    if len(frame) <= design.shape[1]:
        raise ValueError("More line measurements than fitted linear parameters are required")

    calibration_covariance = np.zeros((len(frame), len(frame)))
    if calibration_groups is not None:
        if isinstance(calibration_groups, str):
            if calibration_groups not in frame:
                raise KeyError(f"Calibration-group column {calibration_groups!r} is absent")
            groups = frame[calibration_groups].to_numpy()
        else:
            groups = np.asarray(calibration_groups)
        if groups.shape != (len(frame),):
            raise ValueError("calibration_groups must have one value per line")
        for group in pd.unique(groups):
            fraction = (calibration_fraction.get(group, 0.0)
                        if isinstance(calibration_fraction, Mapping)
                        else calibration_fraction)
            if not np.isfinite(fraction) or fraction < 0:
                raise ValueError("Calibration fractions must be finite and non-negative")
            selected = groups == group
            calibration_covariance[np.ix_(selected, selected)] += (fraction / _LN10) ** 2
    elif isinstance(calibration_fraction, Mapping) or float(calibration_fraction) != 0:
        raise ValueError("calibration_groups are required when calibration_fraction is non-zero")

    def solve(scatter):
        covariance_y = (np.diag(sigma_stat**2 + scatter**2) +
                        calibration_covariance)
        chol = np.linalg.cholesky(covariance_y)
        white_design = np.linalg.solve(chol, design)
        white_y = np.linalg.solve(chol, y)
        normal = white_design.T @ white_design
        if np.linalg.matrix_rank(normal) < normal.shape[0]:
            raise ValueError("The generalized least-squares design is singular")
        parameter_covariance = np.linalg.inv(normal)
        coefficients = parameter_covariance @ white_design.T @ white_y
        residual = y - design @ coefficients
        white_residual = np.linalg.solve(chol, residual)
        chisq = float(white_residual @ white_residual)
        logdet = 2.0 * np.log(np.diag(chol)).sum()
        log_likelihood = -0.5 * (len(y) * np.log(2 * np.pi) + logdet + chisq)
        return coefficients, parameter_covariance, residual, covariance_y, chisq, log_likelihood

    intrinsic_scatter = 0.0
    if fit_intrinsic_scatter:
        if max_intrinsic_scatter_dex <= 0:
            raise ValueError("max_intrinsic_scatter_dex must be positive")
        objective = lambda value: -solve(value)[-1]
        optimized = minimize_scalar(objective, bounds=(0.0, max_intrinsic_scatter_dex),
                                    method="bounded", options={"xatol": 1e-10})
        candidates = [(0.0, objective(0.0)),
                      (float(optimized.x), float(optimized.fun)),
                      (max_intrinsic_scatter_dex, objective(max_intrinsic_scatter_dex))]
        intrinsic_scatter = min(candidates, key=lambda item: item[1])[0]

    coefficients, covariance, residual, covariance_y, chisq, log_likelihood = solve(intrinsic_scatter)
    Tvib, Tvib_error = _temperature_from_slope(coefficients[1], covariance[1, 1])
    Trot, Trot_error = _temperature_from_slope(coefficients[2], covariance[2, 2])
    offsets = {}
    offset_errors = {}
    correction_factors = {}
    for index, state in enumerate(offset_states, start=3):
        offsets[state] = float(coefficients[index])
        offset_errors[state] = float(np.sqrt(max(covariance[index, index], 0.0)))
        correction_factors[state] = float(10 ** (-coefficients[index]))

    number_profiled = int(fit_intrinsic_scatter)
    number_parameters = design.shape[1] + number_profiled
    dof = len(y) - number_parameters
    predicted_corrected = design @ coefficients
    diagnostics = frame.copy()
    diagnostics["E_rot_K"] = energy_rot
    diagnostics["E_vib_K"] = energy_vib
    diagnostics["component_multiplicity"] = multiplicity
    diagnostics["population_correction"] = correction
    diagnostics["corrected_logNu/gu"] = y
    diagnostics["model_logNu/gu"] = predicted_corrected
    diagnostics["residual_dex"] = residual
    diagnostics["marginal_sigma_dex"] = np.sqrt(np.diag(covariance_y))
    diagnostics["marginal_standardized_residual"] = residual / np.sqrt(np.diag(covariance_y))

    return TwoTemperatureFit(
        coefficients=coefficients,
        covariance=covariance,
        parameter_names=tuple(names),
        vibrational_temperature_K=float(Tvib),
        vibrational_temperature_error_K=float(Tvib_error),
        rotational_temperature_K=float(Trot),
        rotational_temperature_error_K=float(Trot_error),
        state_offsets_dex=offsets,
        state_offset_errors_dex=offset_errors,
        state_population_correction_factors=correction_factors,
        intrinsic_scatter_dex=float(intrinsic_scatter),
        chi2=chisq,
        dof=dof,
        reduced_chi2=chisq / dof,
        p_value=float(chi2.sf(chisq, dof)),
        log_likelihood=float(log_likelihood),
        aic=float(2 * number_parameters - 2 * log_likelihood),
        bic=float(number_parameters * np.log(len(y)) - 2 * log_likelihood),
        diagnostics=diagnostics,
    )
