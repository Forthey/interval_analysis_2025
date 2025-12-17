import numpy as np

from utility import (
    interval_vec_mid,
    interval_vec_intersect,
    mat_point_times_point_vec,
    mat_point_times_interval_mat,
    mat_point_minus_interval_mat,
    interval_vec_sub,
    mat_interval_times_interval_vec,
)
from func import f_point, J_point, J_interval
from interval import Interval


def krawczyk_step(X, xc, yc=0.0):
    x0 = interval_vec_mid(X)  # точка (середина бокса)
    fx0 = f_point(x0, xc, yc)  # f(x0) (point)
    Jx0 = J_point(x0, xc, yc)  # J(x0) (point)

    # A = inv(J(x0))
    try:
        A = np.linalg.inv(Jx0)
    except np.linalg.LinAlgError:
        return None, {"reason": "Jacobian singular at midpoint", "x0": x0, "Jx0": Jx0}

    # Interval Jacobian J(X)
    JX = J_interval(X, xc, yc)

    # Compute term1 = x0 - A f(x0) (point vector)
    term1 = x0 - mat_point_times_point_vec(A, fx0)

    # Compute (I - A J(X))
    AJX = mat_point_times_interval_mat(A, JX)  # interval 2x2
    I = np.eye(2, dtype=float)
    I_minus_AJX = mat_point_minus_interval_mat(I, AJX)

    # Compute (X - x0)
    Xm = interval_vec_sub(X, x0)  # interval vector

    # term2 = (I - A J(X)) (X - x0)
    term2 = mat_interval_times_interval_vec(I_minus_AJX, Xm)

    # Krawczyk image
    K = np.array(
        [Interval(term1[0]) + term2[0], Interval(term1[1]) + term2[1]], dtype=object
    )

    # Next box: intersection
    Xnext = interval_vec_intersect(X, K)
    meta = {"x0": x0, "fx0": fx0, "Jx0": Jx0, "A": A, "K": K}
    return Xnext, meta


# ----------------------------
# Вспомогательное: найти верхнюю точку пересечения
# Подстановка x1=x2^2 в уравнение окружности даёт:
# (x2^2-xc)^2 + x2^2 - 1 = 0  (quartic in x2)
# ----------------------------


def approximate_upper_solution(xc):
    # y = x2. Polynomial in y:
    # (y^2 - xc)^2 + y^2 - 1 = 0
    # y^4 + (1 - 2xc)*y^2 + (xc^2 - 1) = 0
    # coefficients for y^4 + 0*y^3 + a*y^2 + 0*y + b
    a = 1.0 - 2.0 * xc
    b = xc * xc - 1.0
    coeffs = [1.0, 0.0, a, 0.0, b]
    roots = np.roots(coeffs)

    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10:
            real_roots.append(r.real)

    if not real_roots:
        return None

    # upper intersection (x2>=0) if exists, else max real
    nonneg = [y for y in real_roots if y >= 0]
    y = max(nonneg) if nonneg else max(real_roots)
    x1 = y * y
    return np.array([x1, y], dtype=float)
