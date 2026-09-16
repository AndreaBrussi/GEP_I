# =============================================================================
# --- START OF FILE Brussi_2026_GEP_I_disk_escape_trajectory_file1.py
#
# Copyright (C) 2026 Andrea Brussi - https://andreabrussi.it
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://gnu.org>.
#
# =============================================================================


#!/usr/bin/env python3
"""Numerical evaluation of the average photon escape distance from a thin disk.

This script computes the Freeman-profile-weighted disk average:

    <t>_disk = [\int_0^R <t(r)> exp(-r/h) r dr] /
               [h^2 (1 - exp(-R/h) (1 + R/h))]

with

    <t(r)> = (1/pi) \int_0^pi sqrt(R^2 - r^2 cos^2(theta))
             E( r sin(theta) / sqrt(R^2 - r^2 cos^2(theta)) )
             sin(theta) dtheta

E is the complete elliptic integral of the second kind.

Implementation details:
- scipy.special.ellipe is used for E
- Both theta and radial integrals are computed numerically with scipy.integrate.quad
- R/h is scanned from 3 to 5 with configurable step

The output includes per-ratio values of <t>_disk / R and the full min-max range.
"""

# =========================
# User-editable parameters
# =========================
R_OVER_H_MIN = 3.0
R_OVER_H_MAX = 5.0
R_OVER_H_STEP = 0.1  # 21 sample points (~20 steps)
SCALE_LENGTH_H = 1.0

# Numerical integration tolerances
EPSABS_R = 1e-7
EPSREL_R = 1e-6
EPSABS_THETA = 1e-9
EPSREL_THETA = 1e-8

import numpy as np
from scipy.integrate import quad
from scipy.special import ellipe


def theta_integrand(theta: float, r: float, R: float) -> float:
    """Integrand of the angular average <t(r)>.

    Notes
    -----
    scipy.special.ellipe expects parameter m = k^2, while the physics
    notation commonly writes E(k). We therefore pass m = k^2.
    """
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    root_term = np.sqrt(max(R * R - r * r * cos_t * cos_t, 0.0))
    if root_term == 0.0:
        return 0.0

    k = (r * sin_t) / root_term
    m = np.clip(k * k, 0.0, 1.0)

    return root_term * ellipe(m) * sin_t


def average_escape_at_radius(
    r: float,
    R: float,
    epsabs: float = EPSABS_THETA,
    epsrel: float = EPSREL_THETA,
) -> float:
    """Compute <t(r)> via numerical integration over theta in [0, pi]."""
    integral, _ = quad(
        theta_integrand,
        0.0,
        np.pi,
        args=(r, R),
        epsabs=epsabs,
        epsrel=epsrel,
        limit=200,
    )
    return integral / np.pi


def radial_integrand(r: float, R: float, h: float, epsabs_theta: float, epsrel_theta: float) -> float:
    """Integrand of the Freeman-weighted radial average."""
    tr = average_escape_at_radius(r, R, epsabs=epsabs_theta, epsrel=epsrel_theta)
    return tr * np.exp(-r / h) * r


def disk_averaged_escape(
    R: float,
    h: float,
    epsabs_r: float = EPSABS_R,
    epsrel_r: float = EPSREL_R,
    epsabs_theta: float = EPSABS_THETA,
    epsrel_theta: float = EPSREL_THETA,
) -> float:
    """Compute Freeman-weighted disk average <t>_disk."""
    numerator, _ = quad(
        radial_integrand,
        0.0,
        R,
        args=(R, h, epsabs_theta, epsrel_theta),
        epsabs=epsabs_r,
        epsrel=epsrel_r,
        limit=200,
    )

    denominator = h * h * (1.0 - np.exp(-R / h) * (1.0 + R / h))
    return numerator / denominator


def compute_normalized_range(rh_min: float, rh_max: float, rh_step: float, h: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return sampled R/h values and corresponding <t>_disk/R values."""
    rh_values = np.arange(rh_min, rh_max + 0.5 * rh_step, rh_step)
    normalized_values = []

    for rh in rh_values:
        R = rh * h
        t_disk = disk_averaged_escape(R=R, h=h)
        normalized_values.append(t_disk / R)

    return rh_values, np.array(normalized_values)


def main() -> None:
    """Run parameter scan and print the numerical range needed for the paper."""
    rh_values, t_over_r = compute_normalized_range(
        R_OVER_H_MIN,
        R_OVER_H_MAX,
        R_OVER_H_STEP,
        h=SCALE_LENGTH_H,
    )

    print("Freeman-weighted average escape distance from thin disk")
    print(
        f"Scanned R/h in [{R_OVER_H_MIN:.1f}, {R_OVER_H_MAX:.1f}] "
        f"with step {R_OVER_H_STEP:.1f}"
    )
    print("\nR/h\t<t>_disk / R")
    for rh, val in zip(rh_values, t_over_r):
        print(f"{rh:>4.1f}\t{val:.8f}")

    print("\nNormalized range over scanned R/h:")
    print(f"min(<t>_disk/R) = {t_over_r.min():.8f}")
    print(f"max(<t>_disk/R) = {t_over_r.max():.8f}")


if __name__ == "__main__":
    main()


# =============================================================================
# --- END OF FILE ---
# =============================================================================
