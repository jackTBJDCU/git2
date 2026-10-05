import matplotlib.pyplot as plt
import numpy as np


def main():
    plt.close()

    coords = np.linspace(0, 2 * np.pi, 101)
    x, y = np.meshgrid(coords, coords)

    vx = np.cos(x) * y
    vy = np.sin(x) * x

    # Create figure
    plt.figure(figsize=(6, 6))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Quiver plot of $V(x,y) = (\\cos(x)\\,y,\\; \\sin(x)\\,x)$')

    skip = 5
    x_s = x[::skip, ::skip]
    y_s = y[::skip, ::skip]
    vx_s = vx[::skip, ::skip]
    vy_s = vy[::skip, ::skip]

    plt.quiver(x_s, y_s, vx_s, vy_s,
               pivot='mid',
               scale=100,
               label='$V_x = \\cos(x)\\,y$,  $V_y = \\sin(x)\\,x$')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()