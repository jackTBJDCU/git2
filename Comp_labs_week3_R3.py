# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 12:34:24 2026

@author: boris
"""
import numpy as np 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pathlib import Path

def simple_pendulum(t, y):
    x, v = y  # extracts the x and v values from the tuple
    dydt = np.array([v, -x])  # generates an array with the rates of change: dxdt = v, dvdt = -x
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
def exact_pendlumx(t):
    x = x0*np.cos(t) + (v0 * np.sin(t))
    return x
def exact_pendlumv(t):
    x = (-x0 * np.sin(t)) + (v0 * np.cos(t))
    return x

fig, (ax1, ax2) = plt.subplots(1, 2,)
fig.set_size_inches(14, 6)
x0 = 0
v0 = 1
y0 = np.array([x0,v0])
t0 = 0
tf = 30
i = 10000

t = np.linspace(t0, tf, i)

x_f = [x0]
v_f = [v0]

result = solve_ivp(fun=simple_pendulum,t_span=(t0, tf),y0=y0,method="RK45",t_eval=t)

x, v = result.y
t = result.t

#fig one left 

ax1.plot(t,x,"r",label = "x (RK45)")
ax1.plot(t,v,"blue",label = "v (RK45)")
ax1.plot(t,exact_pendlumx(t),"--g",label = "exact x = sin(t)")
ax1.plot(t,exact_pendlumv(t),"--m",label = "exact v = cos(t)")
ax1.set_xlabel("t")
ax1.set_ylabel("x, v")
ax1.set_title(f"Simple harmonic oscillator: x0={x0}, v0={v0}")
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
out_file = out_dir / f"R3_sho_x0_{x0}_v0_{v0}.png"
plt.savefig(out_file, dpi=150)
print(f"Saved figure to: {out_file}")

plt.show()

#fig 2 standard and diffrence phase space
phase_space(x,v)
phase_space(x -exact_pendlumx(t), v -exact_pendlumv(t),tit = "the phase diagram of \u0394 x and \u0394 v")

#diffrent values for omage in sho

def simple_pendulum_ω(t, y, omeg0=1.0):
    x, v = y
    dydt = np.array([v, -omeg0**2 * x])
    return dydt

param_sets = [
    (0.0, 1.0, 1.0, "standard: x=sin(t)"),
    (1.0, 0.0, 1.0, "x0=1, v0=0 -> x=cos(t)"),
    (0.0, 1.0, 2.0, "\u03c9,0=2 -> double frequency"),
    (0.0, 1.0, 0.5, "\u03c9,0=0.5 -> half frequency"),
    (0.0, 0.0, 1.0, "x0=v0=0 -> stays at origin"),
]

fig2, (ax1, ax2) = plt.subplots(1, 2,)
fig2.set_size_inches(14, 6)

for xp0, vp0, ω, lbl in param_sets:
    res_p = solve_ivp(fun=lambda tt, yy: simple_pendulum_ω(tt, yy, ω),
                      t_span=(t0, tf), y0=[xp0, vp0],
                      method="RK45", t_eval=t,
                      rtol=1e-10, atol=1e-12)
    xp, vp = res_p.y
    ax1.plot(res_p.t, xp, label = lbl)
    ax2.plot(xp, vp, label = lbl)

ax1.set_xlabel("t")
ax1.set_ylabel("x")
ax1.set_title("x(t) for different initial conditions / \u03c9")
ax1.legend(fontsize=8)

ax2.set_xlabel("x")
ax2.set_ylabel("v")
ax2.set_title("Phase space for different parameters")
ax2.axis('equal')
ax2.legend(fontsize=8)

plt.tight_layout()
out_file2 = out_dir / "R3_sho_parameter_exploration.png"
plt.savefig(out_file2, dpi=150)
print(f"Saved figure to: {out_file2}")
plt.show()