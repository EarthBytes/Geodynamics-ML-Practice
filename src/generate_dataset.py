# Geodynamic simulation database for surrogate-modelling practice

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED = 42
N_SIMULATIONS = 80

# Parameter bounds (dimensionless / stylised geodynamic knobs)
BOUNDS = {
    "Ra": (1e5, 1e8),  # Rayleigh number
    "viscosity_contrast": (1.0, 1e3),
    "initial_temperature": (0.5, 1.5),  # normalised mantle potential temperature
    "internal_heating": (0.0, 1.0),  # normalised radiogenic heating
    "activation_energy": (2.0, 6.0),  # log10(E_a / R) style grouping (stylised)
}

def _latin_hypercube_unit(n: int, d: int, seed: int) -> np.ndarray:
    """Simple LHS on [0, 1]^d without scipy."""
    rng = np.random.default_rng(seed)
    cuts = np.linspace(0, 1, n + 1)
    u = np.zeros((n, d))
    for j in range(d):
        points = rng.uniform(cuts[:n], cuts[1:])
        rng.shuffle(points)
        u[:, j] = points
    return u

def _scale_uniform(unit: np.ndarray, low: float, high: float) -> np.ndarray:
    return low + unit * (high - low)

def sample_parameters(n: int = N_SIMULATIONS, seed: int = RNG_SEED) -> pd.DataFrame:
    u = _latin_hypercube_unit(n, len(BOUNDS), seed)
    cols = list(BOUNDS.keys())
    data = {}
    for j, name in enumerate(cols):
        low, high = BOUNDS[name]
        if name == "Ra":
            log_low, log_high = np.log10(low), np.log10(high)
            data[name] = 10 ** _scale_uniform(u[:, j], log_low, log_high)
        elif name == "viscosity_contrast":
            log_low, log_high = np.log10(low), np.log10(high)
            data[name] = 10 ** _scale_uniform(u[:, j], log_low, log_high)
        else:
            data[name] = _scale_uniform(u[:, j], low, high)

    df = pd.DataFrame(data)
    df.insert(0, "simulation_id", np.arange(1, n + 1, dtype=int))
    return df

def surface_heat_flux_truth(params: pd.DataFrame) -> np.ndarray:
    """
    Stylised nonlinear map from inputs to surface heat flux (arbitrary units).

    Not a physical model — mimics coupled nonlinear simulation response.
    """
    ra = params["Ra"].to_numpy()
    visc = params["viscosity_contrast"].to_numpy()
    t0 = params["initial_temperature"].to_numpy()
    h = params["internal_heating"].to_numpy()
    ea = params["activation_energy"].to_numpy()

    flux = (
        np.sin(np.log10(ra))
        + 0.5 * np.log10(visc)
        + t0**2
        + 0.4 * h
        - 0.15 * ea
        + 0.08 * np.log10(ra) * t0
    )
    return flux

def add_run_noise(flux: np.ndarray, seed: int = RNG_SEED) -> np.ndarray:
    """Small heteroscedastic 'numerical noise' on each expensive run."""
    rng = np.random.default_rng(seed + 1)
    sigma = 0.02 * (1.0 + np.abs(flux))
    return flux + rng.normal(0.0, sigma)

def build_simulation_table(
    n: int = N_SIMULATIONS,
    seed: int = RNG_SEED,
    include_noise: bool = True,
) -> pd.DataFrame:
    params = sample_parameters(n=n, seed=seed)
    flux = surface_heat_flux_truth(params)
    if include_noise:
        flux = add_run_noise(flux, seed=seed)
    params["surface_heat_flux"] = flux
    return params

def save_dataset(path: Path, **kwargs) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = build_simulation_table(**kwargs)
    df.to_csv(path, index=False)
    return df