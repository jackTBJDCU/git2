#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

h_list = [0.02, 0.01, 0.005, 0.001]
tmax = 32
x0, y0 = 0, 1

def dx_f(y):
    return y

def dy_f(x):
    return -x

def exact_solution(h):
    t = 0.0
    X_exact = [np.sin(t)]
    T_exact = [t]
    while t < tmax:
        t = t + h
        X_exact.append(np.sin(t))
        T_exact.append(t)
    return X_exact, T_exact

fig, (ax1, ax2) = plt.subplots(2, 1)
fig.set_size_inches(9, 10)

T_exact_fine = np.linspace(0, tmax, 2000)
X_exact_fine = np.sin(T_exact_fine)
#plotting the Euler method using diffrent jumps 
for h in h_list:
    t = 0.0
    x = x0
    y = y0
    T_v = [t]
    X_v = [x]
    while t < tmax:
        dx = dx_f(y)
        dy = dy_f(x)
        x = x + dx * h
        y = y + dy * h
        t = t + h
        T_v.append(t)
        X_v.append(x)
        
# Exact solution
    X_exact_grid, T_exact_grid = exact_solution(h)
    X_diff = np.array(X_v) - np.array(X_exact_grid)

    ax1.plot(T_v, X_v, label=f"Euler h={h}")
    ax2.plot(T_v, X_diff, label=f"Euler h={h}")

ax1.plot(T_exact_fine, X_exact_fine, 'k--', linewidth=2, label="exact sin(t)")

ax1.set_ylabel("x")
ax1.set_title(f"Euler method E3 : divergence with step size (t-max={tmax})")
ax1.legend(loc='upper left',fontsize = 7.8)

ax2.set_xlabel("t")
ax2.set_ylabel("x_num - x_exact")
ax2.legend()
plt.show()