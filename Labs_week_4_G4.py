import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def main():
    plt.close()
    r = 1.0
    K = 10.0

    def f(t, x):
        return r * x * (1 - x / K)

    x_range = np.linspace(-2, 15, 400)
    dxdt = r * x_range * (1 - x_range / K)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.plot(x_range, dxdt, 'b-', linewidth=2, label=r"$x' = r x (1-x/K)$")
    ax.axhline(0, color='black', linewidth=0.8)
    ax.plot([0, K], [0, 0], 'ro', markersize=10)
    ax.annotate('$x^* = 0$ (unstable)', xy=(0, 0), xytext=(1.6, 1),
                color='red', fontsize=11)
    ax.annotate(f'$x^* = K = {K}$ (stable)', xy=(K, 0), xytext=(6.6, -2),
                color='red', fontsize=11)
    for xp in [-1, 1, 5, 12]:
        val = r * xp * (1 - xp / K)
        direction = np.sign(val)
        ax.arrow(xp, -0.2, 0.7 * direction, 0,
                 head_width=0.4, head_length=0.3, color='navy', alpha=0.7)
    ax.set_xlabel('x')
    ax.set_ylabel("dx/dt")
    ax.set_title(f'Verhulst: r={r}, K={K}')
    ax.legend()

    t_span = (0, 30)
    t_eval = np.linspace(t_span[0], t_span[1], 600)
    x0_values = [0, 0.1, 1, 5, 10, 12,15]

    ax = axes[0, 1]
    for x0 in x0_values:
        sol = solve_ivp(f, t_span, [x0], t_eval=t_eval, method='RK45')
        ax.plot(sol.t, sol.y[0], linewidth=1.8, label=f'$x_0$ = {x0}')
    ax.axhline(K, color='red', linestyle='--', linewidth=1,
               label=f'K = {K} (capacity)')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('time  t')
    ax.set_ylabel('population  x(t)')
    ax.set_title('Verhulst model: solution for different $x_0$')
    ax.legend(loc='lower right', fontsize=9)

    r_values = [0.3, 0.7, 1.0, 2.0]
    colors = plt.cm.plasma(np.linspace(0, 1, len(r_values)))

    ax = axes[1, 0]
    for rv, col in zip(r_values, colors):
        def f_r(t, x, rv=rv):
            return rv * x * (1 - x / K)
        sol = solve_ivp(f_r, (0, 20), [1.0],
                        t_eval=np.linspace(0, 20, 400),
                        method='RK45', rtol=1e-8, atol=1e-10)
        ax.plot(sol.t, sol.y[0], color=col, linewidth=1.8, label=f'r = {rv}')
    ax.axhline(K, color='red', linestyle='--', linewidth=1, label=f'K = {K}')
    ax.set_xlabel('time  t')
    ax.set_ylabel('population  x(t)')
    ax.set_title(f'Effect of growth rate r  (K = {K}, $x_0$ = 1.0)')
    ax.legend()

    ax = axes[1, 1]
    for rv in [-0.5, -1.0,0.5]:
        def f_neg(t, x, rv=rv):
            return rv * x * (1 - x / K)
        sol = solve_ivp(f_neg, (0, 30), [5],
                        t_eval=np.linspace(0, 10, 400),
                        method='RK45')
        ax.plot(sol.t, sol.y[0], linewidth=1.8, label=f'r = {rv}')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axhline(K, color='red', linestyle='--', linewidth=1, label=f'K = {K}')
    ax.set_xlabel('time  t')
    ax.set_ylabel('population  x(t)')
    ax.set_title('Negative r value (Die_out)')
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()