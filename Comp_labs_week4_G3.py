import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
def main():
    plt.close()
    coords = np.linspace(-3, 3, 101)
    x, v = np.meshgrid(coords, coords)
    omega = 1.0
    b = 0.0
    dxdt = v
    dvdt = -b * v - omega**2 * x
    E = 0.5 * v**2 + 0.5 * omega**2 * x**2
    #fig1
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    ax = axes[0]
    ax.set_aspect('equal', adjustable='box')
    cf = ax.contourf(x, v, E, levels=20, cmap='plasma', alpha=0.7)
    ax.contour(x, v, E, levels=10, colors='white', linewidths=0.5)
    ax.streamplot(x, v, dxdt, dvdt, density=1,
                  color='black', linewidth=0.8, arrowsize=1.2)

    skip = 5
    ax.quiver(x[::skip, ::skip], v[::skip, ::skip],
              dxdt[::skip, ::skip], dvdt[::skip, ::skip],
              color='white', alpha=0.5, scale=30)

    ax.set_xlabel('position  x')
    ax.set_ylabel('velocity  v')
    ax.set_title(f'SHO phase space ($\\omega$ = {omega}, b = {b})')
    plt.colorbar(cf, ax=ax, label='Energy E')

    ax = axes[1]
    ax.set_aspect('equal', adjustable='box')

    t_span = (0, 20)
    t_eval = np.linspace(0, 20, 1000)

    def sho(t, state):
        x_, v_ = state
        return [v_, -b * v_ - omega**2 * x_]

    initial_conditions = [(1, 0), (2, 2), (1, 1), (0, 2), (1.5, 1.5)]
    for ic in initial_conditions:
        x0 = ic[0]
        v0 = ic[1]
        sol = solve_ivp(sho, t_span, [x0, v0], t_eval=t_eval, method='RK45')
        ax.plot(sol.y[0], sol.y[1], linewidth=1.5,
                label=f'($x_0$, $v_0$) = ({x0}, {v0})')

    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('position  x')
    ax.set_ylabel('velocity  v')
    ax.set_title('SHO trajectories (closed orbits)')
    ax.legend(fontsize=8,loc="lower left")

    plt.tight_layout()
    plt.show()
    #fig2
    b_values = [0.0, 0.3, 1.0, 2.0, 4.0]
    omega = 1.0
    
    fig, axes = plt.subplots(1, len(b_values),
                             figsize=(4 * len(b_values), 5),
                             sharey=True)
    
    for i in range(len(b_values)):
        b = b_values[i]
        ax = axes[i]
        ax.set_aspect('equal', adjustable='box')
    
        dxdt = v
        dvdt = -b * v - omega**2 * x
    
        ax.streamplot(x, v, dxdt, dvdt, density=1.0,
                      color='black', linewidth=0.6, arrowsize=1.0)
    
        def damped(t, state, b=b):
            x_, v_ = state
            return [v_, -b * v_ - omega**2 * x_]
    
        sol = solve_ivp(damped, (0, 20), [2, 0],
                        t_eval=np.linspace(0, 20, 1000), method='RK45')
        ax.plot(sol.y[0], sol.y[1], color='red', linewidth=1.8)
    
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_xlabel('x')
        ax.set_title(f'b = {b}')
        
    
    axes[0].set_ylabel('velocity  v')
    plt.suptitle(f'Damped SHO: effect of damping b ($\\omega$ = {omega})',
                 fontsize=13)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    name_reg = ['Underdamped', 'Critically damped', 'Overdamped']
    b_reg = [0.5, 2.0, 5.0]
    omega_reg = [1.0, 1.0, 1.0]

    for i in range(len(name_reg)):
        name_r = name_reg[i]
        b = b_reg[i]
        omega = omega_reg[i]
        ax = axes[i]
        ax.set_aspect('equal', adjustable='box')

        dxdt = v
        dvdt = -b * v - omega**2 * x

        def f(t, state, b=b, omega=omega):
            x_, v_ = state
            return [v_, -b * v_ - omega**2 * x_]

        sol = solve_ivp(f, (0, 20), [2, 0],
                        t_eval=np.linspace(0, 20, 1000), method='RK45')
        ax.plot(sol.y[0], sol.y[1], color='darkred', linewidth=1.8)

        ax.streamplot(x, v, dxdt, dvdt, density=1.0,
                      color='grey', linewidth=0.5, arrowsize=1.0)

        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('v')
        ax.set_title(f'{name_r}\nb = {b}, $\\omega$ = {omega}')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()