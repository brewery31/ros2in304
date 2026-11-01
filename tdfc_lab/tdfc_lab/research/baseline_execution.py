"""Explicit physical interface for discrete acceleration-based baseline laws.

The native law is still evaluated at the declared control period. Its velocity
increment is interpreted as a nominal acceleration, capped, then translated to
the same lagged velocity-setpoint interface as the surface controller. This is
an execution adaptation, not an exact reproduction of native trajectories.
"""
import numpy as np


def acceleration_setpoint(native_velocity, observed_velocity, actual_velocity, *, dt, tau, nominal_limit):
    if dt <= 0 or tau < 0 or nominal_limit <= 0:
        raise ValueError('positive timestep/nominal limit and nonnegative lag required')
    acceleration = (native_velocity - observed_velocity) / dt
    size = np.linalg.norm(acceleration, axis=1)
    acceleration = acceleration * np.minimum(1., nominal_limit / np.maximum(size, 1e-30))[:, None]
    gain = dt if tau == 0 else dt / (-np.expm1(-dt / tau))
    return actual_velocity + gain * acceleration
