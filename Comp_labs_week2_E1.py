# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:23:07 2026

@author: boris
"""

import numpy as np
import matplotlib.pyplot as plt

Dt = 0.001
tmax = 10
tau = 2
t = 0
t0 = 0
N0 = 10

T_v = [t0]
N_v = [N0]

diff = []
while abs(t - tmax) > Dt/2:
    N0 = N0 - (N0 / tau) * Dt
    t += Dt
    T_v.append(t)
    N_v.append(N0)

N_exact = 10 * np.exp(-np.array(T_v) / tau)

for i in range(len(N_v)):
    diff.append(N_exact[i] - N_v[i])

plt.figure()
plt.plot(T_v, N_v, 'o-', label='Euler')
plt.plot(T_v, N_exact, 'k--', label='Exact')
plt.xlabel('t')
plt.ylabel('N(t)')
plt.legend()
plt.show()

plt.figure()
plt.plot(T_v, diff, 'r-')
plt.xlabel('t')
plt.ylabel('N_exact - N_Euler')
plt.title('Difference vs time')
plt.show()

print(diff)