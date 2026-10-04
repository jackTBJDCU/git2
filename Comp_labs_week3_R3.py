import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def sho(t, y):
    x, v = y
    return [v, -x]


t_max = 30.0
t_eval = np.linspace(0, t_max, 2000)
y0 = [0.0, 1.0]

sol = solve_ivp(sho, (0, t_max), y0, t_eval=t_eval, rtol=1e-9, atol=1e-12)

x_num = sol.y[0]
v_num = sol.y[1]
t = sol.t
x_exact = np.sin(t)
v_exact = np.cos(t)


fig1, (axL, axR) = plt.subplots(1, 2, figsize=(11, 7))
axL.plot(t, x_num, color='red', lw=2, label=r'$x(t)$')
axL.plot(t, v_num, color='blue', lw=2, label=r'$v(t)$')
axL.plot(t, x_exact, 'k--', lw=1, label=r'exact $x=\sin(t)$')
axL.plot(t, v_exact, 'g--', lw=1, label=r'exact $v=\cos(t)$')
axL.set_xlabel('time  t  (s)')
axL.set_ylabel('x(t),  v(t)')
axL.set_title('(a) Time dependence')
axL.legend(loc='upper right', fontsize=8, framealpha=0.9)

axR.plot(x_num, v_num, color='green', lw=1.8, label='orbit')
axR.set_xlabel('x')
axR.set_ylabel('v')
axR.set_title('(b) Phase space')
axR.set_aspect('equal')
axR.legend(loc='upper right')

fig1.suptitle(f'Simple harmonic oscillator (x0=0, v0=1, t_max={t_max:g})',
              fontsize=17)
fig1.tight_layout(rect=[0, 0, 1, 0.95])
fig1.savefig('R3_fig1_combined.png', dpi=150)


def integrate(x0, v0, omega0, t_max=30.0, n=2000):
    def deriv(t, y):
        x, v = y
        return [v, -(omega0**2) * x]
    te = np.linspace(0, t_max, n)
    s = solve_ivp(deriv, (0, t_max), [x0, v0], t_eval=te,
                  rtol=1e-9, atol=1e-12)
    return s.t, s.y[0], s.y[1]


cases = [
    (r'$\omega_0=1,\ x_0=0,\ v_0=1$', 0.0, 1.0, 1.0, 'tab:blue'),
    (r'$\omega_0=1,\ x_0=1,\ v_0=0$', 1.0, 0.0, 1.0, 'tab:orange'),
    (r'$\omega_0=2$',               0.0, 1.0, 2.0, 'tab:green'),
    (r'$\omega_0=0.5$',               0.0, 1.0, 0.5, 'tab:red'),
    (r'$x_0=v_0=0$ ',              0.0, 0.0, 1.0, 'tab:purple'),
]

fig2, (axL, axR) = plt.subplots(1, 2, figsize=(12, 7))
for label, x0, v0, w0, colour in cases:
    tt, xx, vv = integrate(x0, v0, w0)
    axL.plot(tt, xx, color=colour, lw=1.8, label=label)
    if x0 == 0 and v0 == 0:
        axR.plot(xx, vv, 'o', color=colour, ms=6, label=label)
    else:
        axR.plot(xx, vv, color=colour, lw=1.8, label=label)

axL.set_xlabel('time  t  (s)')
axL.set_ylabel('x(t)')
axL.set_title(r'(a) x(t) for different initial conditions / $\omega_0$')
axL.legend(loc='upper right', framealpha=0.9)

axR.set_xlabel('x')
axR.set_ylabel('v')
axR.set_title('(b) Phase space for different parameters')
axR.set_aspect('equal')
axR.legend(loc='upper right', framealpha=0.9)

fig2.suptitle('Parameter exploration of the simple harmonic oscillator',
              fontsize=16)
fig2.tight_layout(rect=[0, 0, 1, 0.95])
fig2.savefig('R3_fig2_parameter_study.png', dpi=150)


dx = x_num - x_exact
dv = v_num - v_exact

fig3, (axL, axR) = plt.subplots(1, 2, figsize=(11, 7))
axL.plot(t, dx, color='crimson', lw=1.8, label=r'$\Delta x$')
axL.plot(t, dv, color='royalblue', lw=1.8, label=r'$\Delta v$')
axL.set_xlabel('time  t  (s)')
axL.set_ylabel(r'$\Delta x,\ \Delta v$')
axL.set_title('(a) Residual vs time')
axL.legend(loc='upper right', fontsize=8, framealpha=0.9)

axR.plot(dx, dv, color='darkgreen', lw=1.5, label='RK45 - exact')
axR.set_xlabel(r'$\Delta x = x_{\rm num} - x_{\rm exact}$')
axR.set_ylabel(r'$\Delta v = v_{\rm num} - v_{\rm exact}$')
axR.set_title('(b) Residual phase space')
axR.set_aspect('equal')
axR.legend(loc='upper right', fontsize=8)

fig3.suptitle('RK45 residual against the exact solution', fontsize=12)
fig3.tight_layout(rect=[0, 0, 1, 0.95])
fig3.savefig('R3_fig3_residual.png', dpi=150)

plt.show()