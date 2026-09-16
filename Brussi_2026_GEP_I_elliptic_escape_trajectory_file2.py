# =============================================================================
# --- START OF FILE Brussi_2026_GEP_I_elliptic_escape_trajectory_file2.py
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
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad

# ====
# USER-EDITABLE PARAMETERS
# ====
R_OVER_RE = 4.0
B4 = 7.6692
E_MIN = 0
E_MAX = 7
# ====

def local_average_t(r, R_halo):
    """
    Mean escape path length from galactocentric distance r to the surface
    of a sphere of radius R_halo, averaged over all isotropic directions.
    """
    if np.isclose(r, 0.0):
        return R_halo
    if r >= R_halo:
        return R_halo / 2.0

    ratio = np.clip(r / R_halo, 0.0, 0.999999999999)
    return (
        R_halo / 2.0
        + (R_halo**2 - r**2) / (4.0 * r) * np.log((1.0 + ratio) / (1.0 - ratio))
    )

def profile_phi(r, re):
    """
    de Vaucouleurs-like radial weighting used in the current model.
    """
    return np.exp(-B4 * ((r / re) ** 0.25))

def compute_series(mode="oblate"):
    """
    Returns arrays for:
    - E type index
    - axis ratio q
    - geometric average <t>_geom / R
    - luminosity-weighted average <t>_ell / R
    """
    R_halo = 1.0
    re = R_halo / R_OVER_RE

    e_values = []
    q_values = []
    t_geom_values = []
    t_ell_values = []

    for e in range(E_MIN, E_MAX + 1):
        q = 1.0 - e / 10.0

        if mode == "oblate":
            a, c = R_halo, R_halo * q
        elif mode == "prolate":
            c, a = R_halo, R_halo * q
        else:
            raise ValueError("mode must be 'oblate' or 'prolate'")

        def rho_limit(z):
            return a * np.sqrt(max(0.0, 1.0 - (z / c) ** 2))

        def numerator_geometric(rho, z):
            r = np.sqrt(rho**2 + z**2)
            return local_average_t(r, R_halo) * (2.0 * np.pi * rho)

        def numerator_weighted(rho, z):
            r = np.sqrt(rho**2 + z**2)
            return local_average_t(r, R_halo) * profile_phi(r, re) * (2.0 * np.pi * rho)

        def denominator_weighted(rho, z):
            r = np.sqrt(rho**2 + z**2)
            return profile_phi(r, re) * (2.0 * np.pi * rho)

        num_geom, _ = dblquad(numerator_geometric, -c, c, lambda z: 0.0, rho_limit)
        num_ell, _ = dblquad(numerator_weighted, -c, c, lambda z: 0.0, rho_limit)
        den_ell, _ = dblquad(denominator_weighted, -c, c, lambda z: 0.0, rho_limit)

        v_ell = (4.0 / 3.0) * np.pi * a**2 * c

        t_geom = num_geom / v_ell
        t_ell = num_ell / den_ell

        e_values.append(e)
        q_values.append(q)
        t_geom_values.append(t_geom / R_halo)
        t_ell_values.append(t_ell / R_halo)

    return (
        np.array(e_values),
        np.array(q_values),
        np.array(t_geom_values),
        np.array(t_ell_values),
    )

# Compute data
e_obl, q_obl, geom_obl, ell_obl = compute_series("oblate")
e_pro, q_pro, geom_pro, ell_pro = compute_series("prolate")

# Print table in terminal
print("\nOBLATE")
print("E\tq=c/a\t<t>_geom/R\t<t>_ell/R")
for e, q, tg, te in zip(e_obl, q_obl, geom_obl, ell_obl):
    print(f"E{e}\t{q:.1f}\t{tg:.8f}\t{te:.8f}")

print("\nPROLATE")
print("E\tq=a/c\t<t>_geom/R\t<t>_ell/R")
for e, q, tg, te in zip(e_pro, q_pro, geom_pro, ell_pro):
    print(f"E{e}\t{q:.1f}\t{tg:.8f}\t{te:.8f}")

# Plot
plt.figure(figsize=(9, 6))

plt.plot(e_obl, geom_obl, marker="o", linewidth=2.0, label="Oblate  ⟨t⟩geom / R", color="tab:blue")
plt.plot(e_obl, ell_obl, marker="o", linewidth=2.0, linestyle="--", label="Oblate  ⟨t⟩ell / R", color="tab:blue")

plt.plot(e_pro, geom_pro, marker="s", linewidth=2.0, label="Prolate ⟨t⟩geom / R", color="tab:red")
plt.plot(e_pro, ell_pro, marker="s", linewidth=2.0, linestyle="--", label="Prolate ⟨t⟩ell / R", color="tab:red")

plt.xticks(e_obl, [f"E{i}" for i in e_obl])
plt.xlabel("Elliptical morphology")
plt.ylabel("Normalized mean escape trajectory")
plt.title(r"Mean escape trajectory vs morphology ($R/R_{\mathrm{e}} = 4.0$)")
plt.grid(True, alpha=0.3)
plt.legend(frameon=True)
plt.tight_layout()
plt.show()


# =============================================================================
# --- END OF FILE ---
# =============================================================================
