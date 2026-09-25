# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 12:34:24 2026

@author: boris
"""
import numpy as np 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path

def damped_pendulum(t, y, b=0.1, ω0=1.0):
    x, v = y  # extracts the x and v values from the tuple
    dydt = np.array([v, -b*v - ω0**2 * x])  # dxdt = v, dvdt = -b*v - ω0^2*x
    return dydt  # returns the array

def phase_space(x,v,tit = "phase space diagram of x and v"):
    plt.figure()
    plt.plot(v, x, 'g',label = "X v V")
    plt.axis('equal')
    plt.xlabel("v")
    plt.ylabel("x")
    plt.title(tit)
    plt.legend()
    plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2,)
fig.set_size_inches(14, 6)
x0 = 0
v0 = 1
y0 = np.array([x0,v0])
t0 = 0
tf = 50
i = 10000
b = 0.1
ω0 = 1.0

t = np.linspace(t0, tf, i)

x_f = [x0]
v_f = [v0]

result = solve_ivp(fun=lambda tt, yy: damped_pendulum(tt, yy, b, ω0),
                   t_span=(t0, tf), y0=y0, method="RK45", t_eval=t)

x, v = result.y
t = result.t

#fig one left 

ax1.plot(t,x,"r",label = "x (RK45)")
ax1.plot(t,v,"blue",label = "v (RK45)")
ax1.set_xlabel("t")
ax1.set_ylabel("x, v")
ax1.set_title(f"Damped harmonic oscillator: x0={x0}, v0={v0}, b={b}, \u03c9,0={ω0}")
ax1.legend()

#fig 1 right

ax2.plot(x, v, 'k', label = "RK45")
ax2.axis('equal')
ax2.set_xlabel("x")
ax2.set_ylabel("v")
ax2.set_title("Phase space: x vs v")
ax2.legend()

plt.tight_layout()
out_dir = Path.home() / "documents"
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / f"R4_dho_x0_{x0}_v0_{v0}_b_{b}_w0_{ω0}.png"
plt.savefig(out_file, dpi=150)
print(f"Saved figure to: {out_file}")

plt.show()

#fig 2 standard phase space

phase_space(x,v)

#different values for b in damped sho

b_sets = [
    (0.1, "underdamped, b=0.1"),
    (2.0, "critically damped, b=2.0"),
    (5.0, "overdamped, b=5.0"),
]

fig2, (ax1, ax2) = plt.subplots(1, 2,)
fig2.set_size_inches(14, 6)

for bp, lbl in b_sets:
    res_p = solve_ivp(fun=lambda tt, yy: damped_pendulum(tt, yy, bp, ω0),
                      t_span=(t0, tf), y0=y0,
                      method="RK45", t_eval=t,
                      rtol=1e-10, atol=1e-12)
    xp, vp = res_p.y
    ax1.plot(res_p.t, xp, label = lbl)
    ax2.plot(xp, vp, label = lbl)

ax1.set_xlabel("t")
ax1.set_ylabel("x")
ax1.set_title("x(t) for different damping regimes / \u03c9,0")
ax1.legend(fontsize=8)

ax2.set_xlabel("x")
ax2.set_ylabel("v")
ax2.set_title("Phase space for different damping regimes")
ax2.axis('equal')
ax2.legend(fontsize=8)

plt.tight_layout()
out_file2 = out_dir / "R4_dho_damping_regimes.png"
plt.savefig(out_file2, dpi=150)
print(f"Saved figure to: {out_file2}")
plt.show()