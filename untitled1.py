

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 13:19:04 2026

@author: boris
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path

def driven_pendulum(t, y, b, ω0, ωd, A):
    x, v = y
    return np.array([v, -b * v - ω0**2 * x - A * np.sin(ωd * t)])

def placeholder_amplitudes(ω0, b, A, y0, tf, n_samples=1000, n_freqs=3000):
    driving_f = np.linspace(0.4, 2.0 * ω0, n_freqs)
    amps = []

    t_full = np.linspace(0.0, tf, n_samples)

    for ωd in driving_f:
        lfun = lambda t, y: driven_pendulum(t, y, b, ω0, ωd, A)
        result = solve_ivp(fun=lfun, t_span=(0, tf), y0=y0,
                           method="RK45", t_eval=t_full,)
        x, v = result.y
        tail = int(0.8 * len(result.t))
        xs = x[tail:]
        amps.append(0.5 * (xs.max() - xs.min()))
    return driving_f, np.array(amps)

x0, v0 = 0.0, 1.0
y0 = np.array([x0, v0])
tf = 80.0
ω0 = 1.2
A  = 0.1

out_dir = Path.home() / "documents"
out_dir.mkdir(parents=True, exist_ok=True)

b_single = 0.1
driving_f_r7, amps_r7 = placeholder_amplitudes(ω0, b_single, A, y0, tf)

fig_r7, ax_r7 = plt.subplots(figsize=(10, 6))
ax_r7.plot(driving_f_r7, amps_r7, '-', color='tab:blue', label=rf"\(b={b_single}\)")
ax_r7.set_xlabel(r"Driving frequency \(ωd)")
ax_r7.set_ylabel("Steady-state amplitude")
ax_r7.set_title(rf"Resonance curve for (A={A}, (b={b_single}), (ω0={ω0})")
ax_r7.legend()
plt.tight_layout()

out_file_r7 = out_dir / f"R7_resonance_A{A}_b{b_single}_w0_{ω0}.png"
plt.savefig(out_file_r7, dpi=150)
print(f"Saved R7 figure to: {out_file_r7}")
plt.show()

b_values = [0.05, 0.1, 0.2, 0.5, 1.0]
fig_r8, ax_r8 = plt.subplots(figsize=(10, 6))

for bs in b_values:
    driving_f, amps = placeholder_amplitudes(ω0, bs, A, y0, tf)
    ax_r8.plot(driving_f, amps, '-', label=rf"\(b={bs}\)")

ax_r8.set_xlabel(r"Driving frequency \(ωd)")
ax_r8.set_ylabel("Steady-state amplitude")
ax_r8.set_title(rf"Driving frequency scan for (A={A}), (ω0={ω0})")
ax_r8.legend()
plt.tight_layout()

out_file_r8 = out_dir / f"R8_resonance_A{A}_w0_{ω0}.png"
plt.savefig(out_file_r8, dpi=150)
print(f"Saved R8 figure to: {out_file_r8}")
plt.show()