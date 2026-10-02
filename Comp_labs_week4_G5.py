import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def main():
    plt.close()

    a = 4.0
    b = 2.0
    c = 1.0 / 3.0
    d = 1.0

    def lv(t, state, a=a, b=b, c=c, d=d):
        x, y = state
        dxdt = a * x - b * x * y
        dydt = c * x * y - d * y
        return [dxdt, dydt]

    x_vals = np.linspace(0.5, 4.0, 101)
    y_vals = np.linspace(0.5, 3.0, 101)
    X, Y = np.meshgrid(x_vals, y_vals)

    U = a * X - b * X * Y
    V = c * X * Y - d * Y

    step = 5
    Xc = X[::step, ::step]
    Yc = Y[::step, ::step]
    Uc = U[::step, ::step]
    Vc = V[::step, ::step]

    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, np.sqrt(U**2 + V**2),
                 levels=200, cmap='Blues')
    plt.streamplot(X, Y, U, V, color='black',
                   density=1.2, linewidth=1.0, arrowsize=1.2)
    plt.quiver(Xc, Yc, Uc, Vc, color='navy',
               pivot='mid', scale=50)
    plt.plot(d / c, a / b, 'ro', markersize=9,
             label=f'fixed point ({d/c:.2f}, {a/b:.2f})')
    plt.xlabel('rabbits  x')
    plt.ylabel('foxes  y')
    plt.title(f'(a) LV vector field  a={a}, b={b}, c={c:.3f}, d={d}')
    plt.legend()
    plt.tight_layout()
    plt.show()


    t_span = (0, 30)
    t_eval = np.linspace(t_span[0], t_span[1], 1500)

    ic_lis = [
        (4.0, 2.0),
        (2.0, 2.0),
        (1.0, 1.5),
        (3.5, 2.5),
        (0.5, 2.5),]
    colors = plt.cm.viridis(np.linspace(0, 1, len(ic_lis)))

    plt.figure(figsize=(9, 5))
    for i in range(len(ic_lis)):
        x0, y0 = ic_lis[i]
        c_col = colors[i]
        sol = solve_ivp(lv, t_span, [x0, y0], t_eval=t_eval,
                        method='RK45', rtol=1e-8, atol=1e-10)
        plt.plot(sol.t, sol.y[0], color=c_col, linewidth=1.8,
                 label=f'rabbits x0={x0}')
        plt.plot(sol.t, sol.y[1], color=c_col, linewidth=1.8,
                 linestyle='--', label=f'foxes y0={y0}')

    plt.xlabel('time  t')
    plt.ylabel('population')
    plt.title('(b) LV time series for different (x0, y0)')
    plt.legend(loc='upper right', fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 6))
    for i in range(len(ic_lis)):
        x0, y0 = ic_lis[i]
        sol = solve_ivp(lv, t_span, [x0, y0], t_eval=t_eval,
                        method='RK45', rtol=1e-8, atol=1e-10)
        plt.plot(sol.y[0], sol.y[1], color=c_col, linewidth=1.8,
                 label=f'x0={x0}, y0={y0}')
        plt.plot(x0, y0, 'o', color=c_col, markersize=6)

    plt.plot(d / c, a / b, 'r*', markersize=14,
             label=f'fixed point ({d/c:.2f}, {a/b:.2f})')
    plt.xlabel('rabbits  x')
    plt.ylabel('foxes  y')
    plt.title('(c) LV phase-space plot  y vs x')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    a_values = [2.0, 4.0, 6.0, 8.0]
    colors = plt.cm.plasma(np.linspace(0, 1, len(a_values)))

    plt.figure(figsize=(9, 5))
    for i in range(len(a_values)):
        av = a_values[i]

        def lv_a(t, st, av=av):
            x, y = st
            return [av * x - b * x * y, c * x * y - d * y]

        sol = solve_ivp(lv_a, t_span, [4.0, 2.0], t_eval=t_eval,
                        method='RK45', rtol=1e-8, atol=1e-10)
        plt.plot(sol.t, sol.y[0], color=c_col, linewidth=1.8,
                 label=f'rabbits a={av}')
        plt.plot(sol.t, sol.y[1], color=c_col, linewidth=1.8,
                 linestyle='--', label=f'foxes a={av}')

    plt.xlabel('time  t')
    plt.ylabel('population')
    plt.title(f'(d) Effect of prey growth rate a  (b={b}, c={c:.3f}, d={d})')
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()

    d_values = [0.5, 1.0, 1.5, 2.0]
    colors = plt.cm.cividis(np.linspace(0, 1, len(d_values)))

    plt.figure(figsize=(7, 6))
    for i in range(len(d_values)):
        dv = d_values[i]
        c_col = colors[i]

        def lv_d(t, state, dv=dv):
            x, y = state
            return [a * x - b * x * y, c * x * y - dv * y]

        sol = solve_ivp(lv_d, t_span, [4.0, 2.0], t_eval=t_eval,
                        method='RK45', rtol=1e-8, atol=1e-10)
        plt.plot(sol.y[0], sol.y[1], color=c_col, linewidth=1.8,
                 label=f'd={dv}')
        plt.plot(dv / c, a / b, 'o', color=c_col, markersize=6)

    plt.xlabel('rabbits  x')
    plt.ylabel('foxes  y')
    plt.title(f'(d) Phase plots for varying d  (a={a}, b={b}, c={c:.3f})')
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
