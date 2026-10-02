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

    plt.figure(figsize=(7, 5))
    plt.plot(x_range, dxdt, 'b-', linewidth=2,
             label=r"$x' = r x (1-x/K)$")
    plt.axhline(0, color='black', linewidth=0.8)

    plt.plot([0, K], [0, 0], 'ro', markersize=10)
    plt.annotate('x* = 0 (unstable)', xy=(0, 0), xytext=(1.6, 1),
                 color='red', fontsize=11)
    plt.annotate('x* = K = 10 (stable)', xy=(K, 0), xytext=(6.6, -2),
                 color='red', fontsize=11)

    for xp in [-1, 1, 5, 12]:
        val = r * xp * (1 - xp / K)
        direction = np.sign(val)
        plt.arrow(xp, -0.2, 0.7 * direction, 0,
                  head_width=0.4, head_length=0.3,
                  color='navy', alpha=0.7)

    plt.xlabel('x')
    plt.ylabel("dx/dt")
    plt.title('Verhulst model: r=' + str(r) + ', K=' + str(K))
    plt.legend()
    plt.tight_layout()
    plt.show()

    t_span = (0, 30)
    t_eval = np.linspace(t_span[0], t_span[1], 600)

    x0_values = [0,0.1, 1, 5, 9.5, 10, 12]
    plt.figure(figsize=(8, 5))
    for i in range(len(x0_values)):
        x0 = x0_values[i]
        sol = solve_ivp(f, t_span, [x0], t_eval=t_eval,
                        method='RK45')
        plt.plot(sol.t, sol.y[0], linewidth=1.8,
                 label='x0 = ' + str(x0))

    plt.axhline(K, color='red', linestyle='--', linewidth=1,
                label=f'K = {K} capacity)')
    plt.axhline(0, color='black', linewidth=0.5)

    plt.xlabel('time  t')
    plt.ylabel('population  x(t)')
    plt.title('Verhulst model (RK45) souliton for different x0')
    plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.show()

    r_values = [0.3, 0.7, 1.0,2.0]
    colors = plt.cm.plasma(np.linspace(0, 1, len(r_values)))

    plt.figure(figsize=(8, 5))
    for i in range(len(r_values)):
        rv = r_values[i]

        def f_r(t, x, rv=rv):
            return rv * x * (1 - x / K)

        sol = solve_ivp(f_r, (0, 20), [1.0],
                        t_eval=np.linspace(0, 20, 400),
                        method='RK45', rtol=1e-8, atol=1e-10)
        plt.plot(sol.t, sol.y[0], linewidth=1.8,
                 label='r = ' + str(rv))

    plt.axhline(K, color='red', linestyle='--', linewidth=1,
                label='K = ' + str(K))
    plt.xlabel('time  t')
    plt.ylabel('population  x(t)')
    plt.title(f'Effect of growth rate r  (K = {K}, x0 = {x0})')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()