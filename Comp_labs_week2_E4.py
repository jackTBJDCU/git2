#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 20:48:47 2026

@author: boristbj
"""
import numpy as np
import matplotlib.pyplot as plt

h_list = [ 0.005, 0.1, 0.05,0.01,0.001]
tmax = 32

t0 = 0
x0 = 0
v0 = 1

fig, (ax1, ax2) = plt.subplots(2, 1,)
fig.set_size_inches(9, 10)


T_exact_fine = np.linspace(0, tmax, 2000)
X_exact_fine = np.sin(T_exact_fine)
for h in h_list:
    t = 0.0
    x = x0
    v = v0
    T_v = [t]
    X_v = [x]
    V_v = [v]
    diff_v = [0]
    while t < tmax:    
        xinit = x + (h * v)
        vinit = v - (h * x)
        xi = x + h * (v + vinit) / 2
        vi = v - h * ((x+xinit)/2)
        x = xi
        v = vi
        t += h
        X_v.append(x)
        V_v.append(v)
        diff_v.append(x - np.sin(t))
        T_v.append(t)
        
        
    ax1.plot(T_v,X_v,label = f"X h = {h}")
    ax2.plot(T_v,diff_v,label = f"diffrence at {h}")
ax1.plot(T_exact_fine,X_exact_fine,'k--', linewidth=2,label = "Exact" ,color = "black")
ax1.legend()
ax1.set_xlabel("t")
ax1.set_ylabel("x")
ax1.set_title("Modfied Euler equation")

ax2.legend()
ax2.set_xlabel("t")
ax2.set_ylabel("x")
ax2.set_title("Error vs exact solution")
plt.show()