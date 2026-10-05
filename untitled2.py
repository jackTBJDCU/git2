# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def vdp(t, state):
    x, y = state
    return [y, -x + (1 - x**2) * y]


def nullclines(x):
    y_x = np.zeros_like(x)
    y_y = x / (1 - x**2)
    return y_x, y_y


def main():
    plt.close()

    coords = np.linspace(-4, 4, 101)
    X, Y = np.meshgrid(coords, coords)
    dXdt = Y
    dYdt = -X + (1 - X**2) * Y
    speed = np.sqrt(dXdt**2 + dYdt**2)

    x_iso = np.linspace(-4, 4, 4000)
    y_xn, y_yn = nullclines(x_iso)

    mask = np.abs(np.abs(x_iso) - 1) > 0.05
    x_m = x_iso[mask]
    y_m = y_yn[mask]
    breaks = np.where(np.diff(mask.astype(int)) != 0)[0]
    segments = np.split(np.column_stack([x_m, y_m]), breaks + 1)
    segs = [s for s in segments if len(s) > 1]

    fig1, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    ax.set_aspect('equal')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)

    strm = ax.streamplot(X, Y, dXdt, dYdt, density=1,
                         color=speed, cmap='plasma',
                         linewidth=0.8, arrowsize=1.0)
    fig1.colorbar(strm.lines, ax=ax, label=r'$|\dot{x}, \dot{y}|$',
                  fraction=0.046, pad=0.04)

    skip = 8
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
              dXdt[::skip, ::skip], dYdt[::skip, ::skip],
              color='black', scale=500)

    ax.plot(x_iso, y_xn, 'k--', lw=1.2, label=r'$\dot{x}=0$  (x-nullcline)')

    for i, seg in enumerate(segs):
        ax.plot(seg[:, 0], seg[:, 1], color='grey', ls='--', lw=1.2,
                label=r'$\dot{y}=0$  (y-nullcline)' if i == 0 else None)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Van der Pol vector field and nullclines')
    ax.legend(loc='lower right', fontsize=8)

    initial_conditions = [
        (3, 3), (-2, 2), (0.1, 0.1),
        (0.5, 1), (2.5, -2), (-3, -1),
    ]
    t_span = (0, 30)
    t_eval = np.linspace(*t_span, 5000)

    solutions = []
    for ic in initial_conditions:
        sol = solve_ivp(vdp, t_span, ic, t_eval=t_eval,
                        method='RK45', rtol=1e-6, atol=1e-8)
        solutions.append(sol)

    fig2, axes2 = plt.subplots(3, 2, figsize=(12, 10),
                               constrained_layout=True)
    axes2 = axes2.flatten()

    for i, ((x0, y0), sol) in enumerate(zip(initial_conditions, solutions)):
        ax = axes2[i]
        ax.plot(sol.t, sol.y[0], lw=1.5, label='x(t)')
        ax.plot(sol.t, sol.y[1], lw=1.5, label='y(t)')
        ax.set_title(f'(x0, y0) = ({x0}, {y0})', fontsize=10)
        ax.set_xlabel('time  t')
        ax.set_ylabel('value')
        ax.legend(loc='upper right', fontsize=8)

    fig2.suptitle('Time series x(t) and y(t) for all initial conditions',
                  fontsize=13)

    fig3, axes3 = plt.subplots(3, 2, figsize=(10, 13),
                               constrained_layout=True)
    axes3 = axes3.flatten()

    for i, ((x0, y0), sol) in enumerate(zip(initial_conditions, solutions)):
        ax = axes3[i]
        ax.plot(sol.y[0], sol.y[1], lw=1.5, color='crimson')
        ax.plot(x0, y0, 'go', ms=8, label='start')
        ax.set_title(f'(x0, y0) = ({x0}, {y0})', fontsize=10)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.legend(loc='upper right', fontsize=8)

    fig3.suptitle('Individual phase portraits y vs x',
                  fontsize=13)

    fig4, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    ax.set_aspect('equal')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)

    for (x0, y0), sol in zip(initial_conditions, solutions):
        ax.plot(sol.y[0], sol.y[1], lw=1.5, label=f'({x0}, {y0})')

    ax.plot(x_iso, y_xn, 'k--', lw=1.0, alpha=0.7, label=r'$\dot{x}=0$')
    for i, seg in enumerate(segs):
        ax.plot(seg[:, 0], seg[:, 1], color='grey', ls='--', lw=1.0,
                alpha=0.7, label=r'$\dot{y}=0$' if i == 0 else None)

    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('All trajectories converge to the limit')
    ax.legend(loc='lower right', fontsize=8)

    fig5, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    ax.set_aspect('equal')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    for (x0, y0), sol in zip(initial_conditions, solutions):
        ax.plot(sol.y[0], sol.y[1], lw=1.2, alpha=0.8,
                label=f'({x0}, {y0})')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Zoomed-in view of the limit cycle')
    ax.legend(loc='lower right', fontsize=8)

    plt.show()


if __name__ == '__main__':
    main()