# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path

def driven_pendulum(t, y, b, ω0, ωd, A):
    """RHS of the driven damped harmonic oscillator."""
    x, v = y
    dxdt = v
    dvdt = -b * v - ω0**2 * x - A * np.sin(ωd * t)
    return np.array([dxdt, dvdt])

def phase_space(x, v, tit="Phase space diagram of x and v"):
    plt.figure(figsize=(6, 6))
    plt.plot(x, v, 'g', label="x vs v")
    plt.axis('equal')
    plt.xlabel("x")
    plt.ylabel("v")
    plt.title(tit)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



x0, v0 = 0.0, 1.0
y0 = np.array([x0, v0])
t0, tf = 0.0, 50.0
i = 10000
t_eval = np.linspace(t0, tf, i)
b0 = 0.1
ω0 = 1.2
ωd = 1.0
A = 0.1

common_tol = dict(method="RK45", rtol=1e-10, atol=1e-12)

# Output directory
out_dir = Path.home() / "documents"
out_dir.mkdir(parents=True, exist_ok=True)



# Fig 1

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

result = solve_ivp(
    fun=lambda tt, yy: driven_pendulum(tt, yy, b0, ω0, ωd, A),
    t_span=(t0, tf), y0=y0, t_eval=t_eval, **common_tol
)
x, v = result.y

ax1.plot(result.t, x, "r", label="x (RK45)")
ax1.plot(result.t, v, "blue", label="v (RK45)")
ax1.set_xlabel("Time t (s)")
ax1.set_ylabel("x, v")
ax1.set_title(
    f"Driven damped oscillator\n"
    rf"$x_0={x0}$, $v_0={v0}$, $b={b0}$, $\omega_0={ω0}$, "
    rf"$\omega_d={ωd}$, $A={A}$"
)
ax1.legend()

ax2.plot(x, v, 'k', label="RK45")
ax2.axis('equal')
ax2.set_xlabel("x")
ax2.set_ylabel("v")
ax2.set_title("Phase space: x vs v")
ax2.legend()

plt.tight_layout()
out_file = out_dir / f"R6_single_b{b0}_w0_{ω0}_wd_{ωd}_A_{A}.png"
plt.savefig(out_file, dpi=150)
print(f"Saved figure to: {out_file}")
plt.show()


# Fi 2 damping
phase_space(x, v, tit=f"Phase space: b={b0}, ω0={ω0}, ωd={ωd}, A={A}")

b_sets = [
    (0.1, "underdamped, b=0.1"),
    (2 * ω0, f"critically damped, b={2*ω0:.1f}"),
    (10.0, "overdamped, b=5.0"),
]

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for bp, lbl in b_sets:
    res_p = solve_ivp(
        fun=lambda tt, yy: driven_pendulum(tt, yy, bp, ω0, ωd, A),
        t_span=(t0, tf), y0=y0, t_eval=t_eval, **common_tol
    )
    xp, vp = res_p.y
    ax1.plot(res_p.t, xp, label=lbl)
    ax2.plot(xp, vp, label=lbl)

ax1.set_xlabel("Time t (s)")
ax1.set_ylabel("x")
ax1.set_title(rf"$x(t)$ for different damping regimes ($\omega_0={ω0}$)")
ax1.legend(fontsize=9)

ax2.set_xlabel("x")
ax2.set_ylabel("v")
ax2.set_title("Phase space for different damping regimes")
ax2.axis('equal')
ax2.legend(fontsize=9)

plt.tight_layout()
out_file2 = out_dir / "R6_dho_damping_regimes.png"
plt.savefig(out_file2, dpi=150)
print(f"Saved figure to: {out_file2}")
plt.show()


# Fig 3resonance curves
omega_d_vals = np.linspace(0.5, 1.5, 200)
b_res = [0.05, 0.1, 0.2, 0.5, 1.0]

plt.figure(figsize=(10, 6))

for b_scan in b_res:
    amplitudes = []
    for wd in omega_d_vals:
        sol = solve_ivp(
            fun=driven_pendulum, t_span=(t0, tf), y0=y0,
            t_eval=t_eval, args=(b_scan, ω0, wd, A), **common_tol
        )
        x_ss = sol.y[0]
        amplitudes.append(x_ss.max() - x_ss.min())
    plt.plot(omega_d_vals, amplitudes, 'o-', markersize=3,
             label=rf"$b = {b_scan}$")

plt.axvline(ω0, color='k', linestyle=':', alpha=0.5,
            label=rf"$\omega_0 = {ω0}$")
plt.xlabel(r"Driving frequency $\omega_d$")
plt.ylabel("Steady-state amplitude")
plt.title(rf"Resonance curves, $A={A}$, $\omega_0={ω0}$")
plt.legend()
plt.tight_layout()
out_file3 = out_dir / "R6_resonance_curves.png"
plt.savefig(out_file3, dpi=150)
print(f"Saved figure to: {out_file3}")
plt.show()