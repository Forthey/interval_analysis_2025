import numpy as np
import matplotlib.pyplot as plt

from interval import Interval


def interval_vec_mid(X):
    return np.array([X[0].mid(), X[1].mid()], dtype=float)


def interval_vec_sub(X, x0):
    # X - x0 (x0 point vector)
    return np.array([X[0] - x0[0], X[1] - x0[1]], dtype=object)


def interval_vec_add(Y, Z):
    return np.array([Y[0] + Z[0], Y[1] + Z[1]], dtype=object)


def interval_vec_intersect(X, Y):
    out = []
    for i in range(2):
        it = X[i].intersect(Y[i])
        if it is None:
            return None
        out.append(it)
    return np.array(out, dtype=object)


def mat_point_times_interval_vec(A, V):
    # A (2x2 float) * V (2 interval)
    res = []
    for i in range(2):
        s = Interval(0.0)
        for j in range(2):
            s = s + (V[j] * A[i, j])
        res.append(s)
    return np.array(res, dtype=object)


def mat_point_times_point_vec(A, v):
    return A @ v


def mat_point_minus_interval_mat(Iminus, AJX):
    # Iminus (2x2 float) - AJX (2x2 interval)
    out = np.empty((2, 2), dtype=object)
    for i in range(2):
        for j in range(2):
            out[i, j] = Interval(Iminus[i, j]) - AJX[i, j]
    return out


def mat_interval_times_interval_vec(M, V):
    # M (2x2 interval) * V (2 interval)
    res = []
    for i in range(2):
        s = Interval(0.0)
        for j in range(2):
            s = s + (M[i, j] * V[j])
        res.append(s)
    return np.array(res, dtype=object)


def mat_point_times_interval_mat(A, JX):
    # A (2x2 float) * JX (2x2 interval)
    out = np.empty((2, 2), dtype=object)
    for i in range(2):
        for j in range(2):
            s = Interval(0.0)
            for k in range(2):
                s = s + (JX[k, j] * A[i, k])  # NOTE: A[i,k] is float
            out[i, j] = s
    return out


def plot_case(xc, boxes, yc=0.0):
    # Determine plot ranges from boxes and geometry
    x1_vals = [b[0].lo for b in boxes] + [b[0].hi for b in boxes]
    x2_vals = [b[1].lo for b in boxes] + [b[1].hi for b in boxes]
    x1_min, x1_max = min(x1_vals), max(x1_vals)
    x2_min, x2_max = min(x2_vals), max(x2_vals)

    pad1 = 0.3 + 0.2 * (x1_max - x1_min)
    pad2 = 0.3 + 0.2 * (x2_max - x2_min)

    x1_min -= pad1
    x1_max += pad1
    x2_min -= pad2
    x2_max += pad2

    # Curves
    # parabola: x1=x2^2
    x2_grid = np.linspace(x2_min, x2_max, 800)
    x1_par = x2_grid**2

    # circle: (x1-xc)^2 + x2^2 = 1
    theta = np.linspace(0, 2 * np.pi, 800)
    x1_c = xc + np.cos(theta)
    x2_c = yc + np.sin(theta)

    plt.figure(figsize=(7, 6))
    plt.plot(x1_par, x2_grid, label=r"$f_1(x)=0: x_1=x_2^2$")
    plt.plot(x1_c, x2_c, label=rf"$f_2(x)=0: (x_1-{xc})^2+x_2^2=1$")

    # Boxes
    for k, X in enumerate(boxes):
        w = X[0].hi - X[0].lo
        h = X[1].hi - X[1].lo
        rect = plt.Rectangle(
            (X[0].lo, X[1].lo),
            w,
            h,
            fill=False,
            linewidth=2,
            label="boxes" if k == 0 else None,
        )
        plt.gca().add_patch(rect)
        plt.text(X[0].lo, X[1].hi, f"k={k}", fontsize=10, verticalalignment="bottom")

    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title(f"Krawczyk iterations, xc={xc}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig(f"iterations_{xc}.png")
