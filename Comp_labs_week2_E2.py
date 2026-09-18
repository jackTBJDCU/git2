import numpy as np
import matplotlib.pyplot as plt

def f(x,t):
    return t - x*x
fig, (ax) = plt.subplots(1, 1,)
fig.set_size_inches(9, 10)
 
fig2, (ax1, ax2) = plt.subplots(2, 2,)
fig2.set_size_inches(9, 10)
fig2.tight_layout(pad = 4)
 
fig3, (ax5 ) = plt.subplots(1, 1,)
fig3.set_size_inches(9, 10)

fig4, (ax6 ) = plt.subplots(1, 1,)
fig4.set_size_inches(9, 10)


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
    ax.plot(T_v, X_v, label=f"x0 = {x0}")

ax.set_xlabel("t",fontsize = 14)
ax.set_ylabel("x", fontsize = 14)
ax.set_title(f"Euler method, h = {h}, tmax = {tmax}",fontsize = 14)
ax.legend()

#Part 2 longer times

axes2 = [ax1, ax2]
for tmax_i, ax_i in zip([50, 100, 500, 1000], [ax1[0], ax1[1], ax2[0], ax2[1]]):
    for x0 in x0_v:
        t = 0.0
        x = x0
        T_v = [t]
        X_v = [x]
        while t < tmax_i:
            x = x + h * f(x, t)
            t += h
            T_v.append(t)
            X_v.append(x)
        ax_i.plot(T_v, X_v, label=f"x0 = {x0}")
    ax_i.set_xlabel("t")
    ax_i.set_ylabel("x")
    ax_i.set_title(f"Euler method, h = {h}, tmax = {tmax_i}")
    ax_i.legend()

#Part 3 smaller step size
h = 0.01
tmax = 9
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
    ax5.plot(T_v, X_v, label=f"x0 = {x0}")
ax5.set_xlabel("t",fontsize = 14)
ax5.set_ylabel("x",fontsize = 14)
ax5.set_title(f"Euler method, h = {h}, tmax = {tmax}",fontsize = 14)
ax5.legend(fontsize = 14)

#part 4
tmax_large =100
x0 = 1
h = 0.05
prev_T = prev_X = None
for i in range(10):
    t = 0.0
    x = x0
    T_v = [t]; X_v = [x]
    while t < tmax_large:
        x = x + h * f(x, t)
        t += h
        T_v.append(t); X_v.append(x)
    ax6.plot(T_v, X_v, label=f"h = {h}")
    if prev_T is not None:
        d = np.max(np.abs(np.array(X_v) - np.interp(T_v, prev_T, prev_X)))
        print(f"h = {h:.5f}   max change from previous = {d:.2e}")
    prev_T, prev_X = T_v, X_v
    h /= 2

ax6.set_xlabel("t")
ax6.set_ylabel("x")
ax6.set_title(f"Iterative halving of h, tmax = {tmax_large}")
ax6.legend()
plt.show()
