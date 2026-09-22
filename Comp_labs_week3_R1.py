
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 08:54:53 2026

@author: boris
"""
import numpy as np 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def f(t,y):
    dydt = -a*y**3 + b*np.sin(t)
    return dydt

fig, (ax) = plt.subplots(1, 1,)
fig.set_size_inches(10, 9)  
a_f = [1,10,100]
b_f = [1,10]
y0 = np.array([0])
x0 = 0
t0 = 0
tf = 20
i = 101

t = np.linspace(t0, tf, i)
for a in a_f:
    for b in b_f:
        results = solve_ivp(fun=f,t_span=(t0,tf),y0=y0,t_eval=t,method="RK45")
        y = results.y[0]
        t = results.t
        ax.plot(t, y, label=f"a = {a} b = {b}")
ax.set_xlabel("t")
ax.set_ylabel("y")
ax.set_title("Runge-Kutta RK45 solution of dy/dt = -a y³ + b sin(t)")
plt.legend()
plt.show()
         

