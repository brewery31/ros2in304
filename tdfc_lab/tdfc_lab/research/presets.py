"""Frozen parameter presets.

``SIM``       : the configuration used for the large simulation study.
``CRAZYFLIE`` : the hardware configuration validated in
                ``scripts/plot_hw_ring.py`` / ``docs/05`` §7.3d (50 Hz loop,
                3 m/s^2, 1 m/s, 1 cm mocap noise, 40 ms delay, 100 ms lag).
``CRAZYFLIE_NO_DELAY`` : same without delay/lag (best case on a good mocap).
"""
from __future__ import annotations

from .dynamics import SimConfig

SIM = dict(a_max=3.0, v_max=1.0, dt=0.02, t_end=60.0)   # must match the safety layer assumptions
CRAZYFLIE = dict(a_max=3.0, v_max=1.0, dt=0.02, t_end=50.0, sat_rate=10.0,
                 meas_noise=0.01, latency=2)
CRAZYFLIE_NO_DELAY = dict(a_max=3.0, v_max=1.0, dt=0.02, t_end=50.0,
                          meas_noise=0.01)

#: hardware gains (dimensionless) recommended for the Crazyflie preset:
#: damping c >= 4 is what restores the ring under 40 ms delay + 100 ms lag
CRAZYFLIE_GAINS = dict(kappa_s=10.0, kappa_p=1.0, gamma_hat=4.0, c_hat=4.0,
                       k_al_hat=0.0, h_kernel=1.5, c_pair=1.6, r_vis=2.5, mu=1.6)


def make_cfg(preset: str, n_agents: int, **over) -> SimConfig:
    base = {"SIM": SIM, "CRAZYFLIE": CRAZYFLIE,
            "CRAZYFLIE_NO_DELAY": CRAZYFLIE_NO_DELAY}[preset]
    kw = dict(base)
    kw.update(over)
    return SimConfig(n_agents=n_agents, **kw)
