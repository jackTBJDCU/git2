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

def placeholder_amplitudes(ω0, b, A, y0, tf, n_samples=100, n_freqs=300):
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
b  = [0.1,0.05,0.2,0.5,1,1.2]
ω0 = 1.2
A  = 0.1
i = 1
out_dir = Path.home() / "documents"
out_dir.mkdir(parents=True, exist_ok=True)
    
fig, ax = plt.subplots(figsize=(10, 6))
for bs in b:

    driving_f, amps = placeholder_amplitudes(ω0, bs, A, y0, tf)
    
    ax.plot(driving_f, amps, '-',label = rf"$b={b}$")
    print(f"the count is {i}")
    i = 1 + i
    label=rf"$b={b}$"
ax.set_xlabel(r"Driving frequency $\omega_d$")
ax.set_ylabel("Steady-state amplitude")
ax.set_title(rf"Resonance curve: $A={A}$, $b={b}$, $\omega_0={ω0}$")
ax.legend()
plt.tight_layout()

out_file = out_dir / f"R7_resonance_A{A}_b{b}_w0_{ω0}.png"
plt.savefig(out_file, dpi=150)
print(f"Saved figure to: {out_file}")
plt.show()