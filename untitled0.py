# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 09:42:30 2026

@author: boris
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.ticker import MultipleLocator


def vdp(t, state):
    x, y = state
    return [y, -x + (1 - x**2) * y]


def main():
    plt.close()

    coords = np.linspace(-4, 4, 101)
    X, Y = np.meshgrid(coords, coords)

    dXdt = Y
    dYdt = -X + (1 - X**2) * Y

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect('equal')

    speed = np.sqrt(dXdt**2 + dYdt**2)
    strm = ax.streamplot(X, Y, dXdt, dYdt,
                         density=1.5,
                         color=speed, cmap='plasma',
                         linewidth=0.8, arrowsize=1.0)

    skip = 5
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
              dXdt[::skip, ::skip], dYdt[::skip, ::skip],
              color='white', alpha=0.4, scale=30)

    plt.colorbar(strm.lines, ax=ax, label='speed')

    x_iso = np.linspace(-4, 4, 400)
    ax.plot(x_iso, np.zeros_like(x_iso), 'k--', linewidth=1.2,
            label='x-isocline (y=0)')

    x_nonzero = x_iso[np.abs(x_iso) > 1e-6]
    ax.plot(x_nonzero, x_nonzero / (1 - x_nonzero**2),
            'w--', linewidth=1.2, label='y-isocline')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Van der Pol: vector field and isoclines')
    ax.legend(loc='upper right', fontsize=8)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))

    plt.tight_layout()
    plt.show()

    initial_conditions = [
        (3, 3),
        (-2, 2),
        (0.1, 0.1),
        (0.5, 1),
        (2.5, -2),
        (-3, -1),
    ]

    t_span = (0, 30)
    t_eval = np.linspace(0, 30, 3000)

    fig, axes = plt.subplots(len(initial_conditions), 2,
                             figsize=(12, 2.5 * len(initial_conditions)),
                             sharex='col')

    for i in range(len(initial_conditions)):
        x0 = initial_conditions[i][0]
        y0 = initial_conditions[i][1]

        sol = solve_ivp(vdp, t_span, [x0, y0],
                        t_eval=t_eval, method='RK45',
                        rtol=1e-8, atol=1e-10)

        ax = axes[i, 0]
        ax.plot(sol.t, sol.y[0], linewidth=1.5, label='x(t)')
        ax.plot(sol.t, sol.y[1], linewidth=1.5, label='y(t)')
        ax.set_ylabel('value')
        ax.set_title(f'Time series: (x0, y0) = ({x0}, {y0})', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)

        ax = axes[i, 1]
        ax.plot(sol.y[0], sol.y[1], linewidth=1.5, color='crimson')
        ax.plot(x0, y0, 'go', markersize=8, label='start')
        ax.set_ylabel('y')
        ax.set_title(f'Phase: (x0, y0) = ({x0}, {y0})', fontsize=10)
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=8)

    axes[-1, 0].set_xlabel('time  t')
    axes[-1, 1].set_xlabel('x')

    plt.suptitle('Van der Pol oscillator: time series and phase portraits',
                 fontsize=13)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect('equal')

    t_span_long = (0, 50)
    t_eval_long = np.linspace(0, 50, 5000)

    for i in range(len(initial_conditions)):
        x0 = initial_conditions[i][0]
        y0 = initial_conditions[i][1]

        sol = solve_ivp(vdp, t_span_long, [x0, y0],
                        t_eval=t_eval_long, method='RK45',
                        rtol=1e-8, atol=1e-10)

        ax.plot(sol.y[0], sol.y[1], linewidth=1.5, label=f'({x0}, {y0})')

    ax.plot(x_iso, np.zeros_like(x_iso), 'k--', linewidth=1.0,
            alpha=0.6, label='x-isocline')
    ax.plot(x_nonzero, x_nonzero / (1 - x_nonzero**2),
            'grey', linestyle='--', linewidth=1.0,
            alpha=0.6, label='y-isocline')

    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Van der Pol: all trajectories converge to the limit cycle')
    ax.legend(loc='upper right', fontsize=8)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()