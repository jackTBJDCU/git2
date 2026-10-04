import matplotlib.pyplot as plt
import numpy as np


def main():
    plt.close()

    coords = np.linspace(-2, 2, 101)
    x, y = np.meshgrid(coords, coords)

    Z = np.sqrt(x**2 + y**2)

    r_safe = np.where(Z > 0, Z, 1)     # avoid divide-by-zero
    dZdx = np.where(Z > 0, x / r_safe, 0)
    dZdy = np.where(Z > 0, y / r_safe, 0)

    # Figure
    plt.figure(figsize=(7, 6))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Contour + gradient of $Z = \\sqrt{x^2 + y^2}$')

    cf = plt.contourf(x, y, Z, 100, cmap='inferno')
    plt.colorbar(cf, label='Z')

    plt.contour(x, y, Z, 10, colors='white', linewidths=0.5)

    # Coarse grid for quiver (every 5th point)
    skip = 5
    xs = x[::skip, ::skip]
    ys = y[::skip, ::skip]
    us = dZdx[::skip, ::skip]
    vs = dZdy[::skip, ::skip]

    plt.quiver(xs, ys, us, vs, color='black', scale=20)
    plt.show()


if __name__ == '__main__':
    main()
