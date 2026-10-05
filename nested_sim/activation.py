"""Chamber activation function epsilon(t).

Port of `epsilon_smooth` (reference/julia/dynamic_elastance.jl), the default
contraction timer of the cv8Eed model: the sine rise and exponential decay of
Burkhoff's driving function, blended by a logistic weight
L(x) = 1 / (1 + exp(-a (x - 1.5 Tmax))^b) with a = 0.03, b = 3, and shifted so
that epsilon(0) = 0. t is local time since that chamber's contraction onset,
in [0, T); the timer resets every cycle.
"""

import math

A_BLEND = 0.03
B_BLEND = 3.0


def _eps_raw(x, Tmax, tau):
    L = 1.0 / (1.0 + math.exp(-A_BLEND * (x - 1.5 * Tmax)) ** B_BLEND)
    rise = 0.5 * (math.sin(math.pi * x / Tmax - math.pi / 2.0) + 1.0)
    decay = 0.5 * math.exp((1.5 * Tmax - x) / tau)
    return (1.0 - L) * rise + L * decay


def make_epsilon(Tmax, tau):
    """epsilon(t) for one chamber; the epsilon(0) offset is computed once."""
    eps0 = _eps_raw(0.0, Tmax, tau)

    def epsilon(t):
        return _eps_raw(t, Tmax, tau) - eps0

    return epsilon
