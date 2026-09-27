#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 08:52:15 2026

@author: boristbj
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def sho(t, y):
    x, v = y
    return [v, -x]

x0, v0 = 0, 1
t0, tf, n = 0, 20, 101

t_e = np.linspace(t0, tf, n)
sol = solve_ivp(sho, (t0, tf), [x0, v0], t_eval=t_e, method="RK45")

x_num = sol.y[0]
x_exact = np.sin(t_e)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 11))
ax1.plot(sol.t, x_num, alpha=0.7, label=f"Approx x, n={n}")
ax1.plot(t_e, x_exact, "--g", label="Exact x = sin(t)")
ax1.legend(); ax1.set_xlabel("t"); ax1.set_ylabel("x")

ax2.plot(sol.t, x_num - x_exact, label="Error")
ax2.legend(); ax2.set_xlabel("t"); ax2.set_ylabel("x_num - x_exact")
plt.tight_layout()
plt.show()