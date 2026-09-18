# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:23:07 2026

@author: boris
"""

import numpy as np
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1,)
fig.set_size_inches(9, 10)
 

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


ax1.plot(T_v, N_v, "o-", label="Euler")
ax1.plot(T_v, N_exact, "k--", label="Exact")
ax1.set_xlabel("t")
ax1.set_ylabel("Euler Approximation N(t)",fontsize = 10)
ax1.set_title("Euler approximation of exponetinal decay",fontsize = 14)
ax1.legend()

ax2.plot(T_v, diff, "r-")
ax2.set_xlabel("t")
ax2.set_ylabel("Exact N(t) - Euler aproximation N(t) ")
ax2.set_title("Difference vs time" ,fontsize = 14)
plt.show()

print(diff)