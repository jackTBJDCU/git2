# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 10:54:23 2026

@author: boris
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def f(t, I, V, R, L):
    didt = (V - (R * I)) / L
    return didt

I0 = np.array([0.0])
t0 = 0.0
tf = 50.0
h_f = [0.5, 0.1, 0.01]

#Figure 1
V = 10.0
R = 50.0
L = 100.0

fig, (ax1, ax2) = plt.subplots(2, 1)
fig.set_size_inches(10, 11)

t_fine = np.linspace(t0, tf, 2000)
I_exact_fine = (V / R) * (1 - np.exp(-(R * t_fine / L)))
ax1.plot(t_fine, I_exact_fine, "k--", label="Exact solution")

for h in h_f:
    t_e = np.arange(t0, tf + h, h)
    I_exact = (V / R) * (1 - np.exp(-(R * t_e / L)))
    results = solve_ivp(fun=f, t_span=(t0, tf), y0=I0, t_eval=t_e,
                        method="RK45", args=(V, R, L), max_step=h)
    I = results.y[0]
    t = results.t
    ax1.plot(t, I, alpha=0.7, label=f"Approximation, h = {h}")
    diff = I_exact - I
    ax2.plot(t_e, diff, alpha=0.5, label=f"Difference, h = {h}")

ax1.set_xlabel("t")
ax1.set_ylabel("I")
ax2.set_xlabel("t")
ax2.set_ylabel(r"$\Delta I$")
ax1.set_title(f"Runge-Kutta RK45 solution of I when the circuit is closed and its difference at V = {V}, R = {R}, L = {L}")
ax1.legend()
ax2.legend()
plt.show()

#Figure 2
R_f = [25.0, 50.0, 100.0]
V = 10.0
L = 100.0
h = 0.01

fig, ax1 = plt.subplots(1, 1)
fig.set_size_inches(10, 6)

t_fine = np.linspace(t0, tf, 2000)
for R in R_f:
    I_exact_fine = (V / R) * (1 - np.exp(-(R * t_fine / L)))
    ax1.plot(t_fine, I_exact_fine, "k--", alpha=0.5)

for R in R_f:
    results = solve_ivp(fun=f, t_span=(t0, tf), y0=I0,
                        method="RK45", args=(V, R, L), max_step=h)
    I = results.y[0]
    t = results.t
    ax1.plot(t, I, alpha=0.7, label=f"Approximation, R = {R}")

ax1.set_xlabel("t")
ax1.set_ylabel("I")
ax1.set_title(f"Runge-Kutta RK45 solution of I when the circuit is closed at varying R, V = {V}, L = {L}, h = {h}")
ax1.legend()
plt.show()

#Figure 3
V_f = [5.0, 10.0, 20.0]
R = 50.0
L = 100.0
h = 0.01

fig, ax1 = plt.subplots(1, 1)
fig.set_size_inches(10, 6)

t_fine = np.linspace(t0, tf, 2000)
for V in V_f:
    I_exact_fine = (V / R) * (1 - np.exp(-(R * t_fine / L)))
    ax1.plot(t_fine, I_exact_fine, "k--", alpha=0.5)

for V in V_f:
    results = solve_ivp(fun=f, t_span=(t0, tf), y0=I0,
                        method="RK45", args=(V, R, L), max_step=h)
    I = results.y[0]
    t = results.t
    ax1.plot(t, I, alpha=0.7, label=f"Approximation, V = {V}")

ax1.set_xlabel("t")
ax1.set_ylabel("I")
ax1.set_title(f"Runge-Kutta RK45 solution of I when the circuit is closed at varying V, R = {R}, L = {L}, h = {h}")
ax1.legend()
plt.show()