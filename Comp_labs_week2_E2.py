# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:38:32 2026

@author: boris
"""

import numpy as np
import matplotlib.pyplot as plt

def f(x,t):
    return t - x*x

tmax = 9
h = 0.05
x0_v = [3, 1.5 ,1,0,-0.5 ,-0.7,-0.71]

for x0 in x0_v:
    t = 0
    x = x0
    T_v = [t]
    X_v = [x]
    while t < tmax:
        x = x + h * f(x,t)
        t += h
        T_v.append(t)
        X_v.append(x)
    plt.plot(T_v, X_v, label=f"x0 = {x0}")

plt.xlabel("t")
plt.ylabel("x")
plt.title(f"Euler method, h = {h}, tmax = {tmax}")
plt.legend()
plt.show()

#Part 2 longer times
for tmax in [50, 100, 500, 1000]:
    plt.figure()
    for x0 in x0_v:
        t = 0.0
        x = x0
        T_v = [t]
        X_v = [x]
        while t < tmax:
            x = x + h * f(x, t)
            t += h
            T_v.append(t)
            X_v.append(x)
        plt.plot(T_v, X_v, label=f"x0 = {x0}")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title(f"Euler method, h = {h}, tmax = {tmax}")
    plt.legend()
    plt.show()

#Part 3 smaller step size
h = 0.01
tmax = 9
plt.figure()
for x0 in x0_v:
    t = 0.0
    x = x0
    T_v = [t]
    X_v = [x]
    while t < tmax:
        x = x + h * f(x, t)
        t += h
        T_v.append(t)
        X_v.append(x)
    plt.plot(T_v, X_v, label=f"x0 = {x0}")
plt.xlabel("t")
plt.ylabel("x")
plt.title(f"Euler method, h = {h}, tmax = {tmax}")
plt.legend()
plt.show()

#Part 4 
tmax_large = 50
x0 = 1
h = 0.05
prev = None
for i in range(10):
    t = 0.0
    x = x0
    T_v = [t]
    X_v = [x]
    while t < tmax_large:
        x = x + h * f(x, t)
        t += h
        T_v.append(t)
        X_v.append(x)
        
    prev = (T_v, X_v)
    h /= 2