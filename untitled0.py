# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.ticker import MultipleLocator


def vdp(t, state):
    x, y = state
    return [y, -x + (1 - x**2) * y]


def main():
    plt.close('all')

    coords = np.linspace(-4, 4, 101)
    X, Y = np.meshgrid(coords, coords)

    dXdt = Y
    dYdt = -X + (1 - X**2) * Y
    speed = np.sqrt(dXdt**2 + dYdt**2)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect('equal')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)

    strm = ax.streamplot(X, Y, dXdt, dYdt,
                         density=1,
                         color=speed, cmap='plasma',
                         linewidth=0.8, arrowsize=1.0)

    skip = 5
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
              dXdt[::skip, ::skip], dYdt[::skip, ::skip],
              color='black', alpha=0.35, scale=20)

    plt.colorbar(strm.lines, ax=ax, label='speed')
    x_iso = np.linspace(-4, 4, 2000)
    ax.plot(x_iso, np.zeros_like(x_iso), 'k--', linewidth=1.2,
            label='x-isocline (y = 0)')

    mask = np.abs(np.abs(x_iso) - 1) > 0.05
    x_iso_m = x_iso[mask]
    y_iso_m = x_iso_m / (1 - x_iso_m**2)

    breaks = np.where(np.diff(mask.astype(int)) != 0)[0]
    segments = np.split(np.column_stack([x_iso_m, y_iso_m]),
                        breaks + 1)
    for i, seg in enumerate(segments):
        if len(seg) > 1:
            ax.plot(seg[:, 0], seg[:, 1], 'w--', linewidth=1.2,
                    label='y-isocline' if i == 0 else None)

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
                             figsize=(12, 2.5 * len(initial_conditions)))

    for i, (x0, y0) in enumerate(initial_conditions):
        sol = solve_ivp(vdp, t_span, [x0, y0],
                        t_eval=t_eval, method='RK45',
                        rtol=1e-6, atol=1e-8)

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
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.legend(loc='upper right', fontsize=8)

    axes[-1, 0].set_xlabel('time  t')
    axes[-1, 1].set_xlabel('x')

    plt.suptitle('Van der Pol: time series and phase portraits', fontsize=13)
    plt.tight_layout()
    plt.show()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect('equal')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)

    t_span_long = (0, 50)
    t_eval_long = np.linspace(0, 50, 5000)

    for x0, y0 in initial_conditions:
        sol = solve_ivp(vdp, t_span_long, [x0, y0],
                        t_eval=t_eval_long, method='RK45',
                        rtol=1e-6, atol=1e-8)
        ax.plot(sol.y[0], sol.y[1], linewidth=1.5, label=f'({x0}, {y0})')

    ax.plot(x_iso, np.zeros_like(x_iso), 'k--', linewidth=1.0,
            alpha=0.6, label='x-isocline')
    for i, seg in enumerate(segments):
        if len(seg) > 1:
            ax.plot(seg[:, 0], seg[:, 1], 'grey', linestyle='--',
                    linewidth=1.0, alpha=0.6,
                    label='y-isocline' if i == 0 else None)

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