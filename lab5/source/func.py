import numpy as np

from interval import Interval


def f_point(x, xc, yc=0.0):
    x1, x2 = float(x[0]), float(x[1])
    return np.array([x1 - x2 * x2, (x1 - xc) ** 2 + (x2 - yc) ** 2 - 1.0], dtype=float)


def J_point(x, xc, yc=0.0):
    x1, x2 = float(x[0]), float(x[1])
    return np.array([[1.0, -2.0 * x2], [2.0 * (x1 - xc), 2.0 * (x2 - yc)]], dtype=float)


def J_interval(X, xc, yc=0.0):
    x1, x2 = X[0], X[1]
    # df1/dx1 = 1; df1/dx2 = -2*x2
    j11 = Interval(1.0)
    j12 = x2 * (-2.0)
    # df2/dx1 = 2*(x1-xc); df2/dx2 = 2*(x2-yc)
    j21 = (x1 - xc) * 2.0
    j22 = (x2 - yc) * 2.0
    out = np.empty((2, 2), dtype=object)
    out[0, 0], out[0, 1] = j11, j12
    out[1, 0], out[1, 1] = j21, j22
    return out
