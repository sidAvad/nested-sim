"""Nested low-fi / high-fi 8-compartment cardiovascular simulator.

simulate(theta)            -> low-fi model (Python port of Cv8SimApp cv8Eed)
simulate(theta, phi=...)   -> high-fi model; phi=PHI_OFF reproduces low-fi
"""

from .hifi import HiFiModel, apply_wedge
from .lowfi import LowFiModel
from .measurement import measure
from .phi import PHI_OFF, full_phi, sample_phi
from .solver import SimResult, SolverSettings, simulate_beats, simulate_model


def simulate(theta=None, phi=None, settings=None) -> SimResult:
    """Simulate one steady-state beat. theta uses ASCII names (params.THETA_NAMES).

    phi=None runs the low-fi code path; any phi dict (missing keys = off) runs
    the high-fi model. Only structural phi affect this call; measurement phi
    are applied by measurement.measure().
    """
    if phi is None:
        return simulate_model(LowFiModel(theta), settings)
    model = HiFiModel(theta, phi)
    return apply_wedge(simulate_model(model, settings), model.phi)


def simulate_breath(theta, phi, settings=None):
    """High-fi: every beat of one respiratory period (list of SimResult)."""
    model = HiFiModel(theta, phi)
    return [apply_wedge(r, model.phi) for r in simulate_beats(model, settings)]


__all__ = ["simulate", "simulate_breath", "measure", "SimResult", "SolverSettings", "PHI_OFF", "full_phi", "sample_phi",
           "LowFiModel", "HiFiModel"]
