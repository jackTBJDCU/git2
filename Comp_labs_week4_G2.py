# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 11:16:47 2026

@author: boris
"""

# code courtesy of Adam Dempsey
# modified for PHY1055 by Oisín Creaner
import matplotlib.pyplot as plt
import numpy as np


def main():
    plt.close()

    # Create Grid
    coords = np.linspace(-2, 2, 101)
    x, y = np.meshgrid(coords, coords)

    z = np.sqrt((x**2 + y**2))
    dx, dy = np.gradient(z)  # Calculate the gradient

    # Create Figure
    plt.figure(figsize=(6, 6))
    plt.gca().set_aspect('equal', adjustable='box')  # Make plot box square
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('2d $\sqrt{(x^2+y^2)}$ function')

    # Plot scalar function as color plot (contour map)
    plt.contourf(x, y, z, 100)  # plot a contour map using N=20 levels
    plt.set_cmap('inferno')  # change color of map
    plt.contour(x, y, z, 10)  # plot a contour map using N=20 levels
    plt.set_cmap("grey")
    # Plot Gradient as quiver plot
    skip = 5  # Number of points to skip

    # create coarse grid
    x_skipped, y_skipped = x[::skip, ::skip], y[::skip, ::skip]  # note the indexing [start:end:skip]
    dx_skipped, dy_skipped = dx.T[::skip, ::skip], dy.T[::skip, ::skip]  # note the .T transpose method

    plt.quiver(x_skipped, y_skipped, dx_skipped, dy_skipped, scale=0.8)
    plt.show()


# if this is the module called directly, then execute the main function, otherwise only define it
if __name__ == '__main__':
    main()
