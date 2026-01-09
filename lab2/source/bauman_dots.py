import math
import numpy as np

from base import Interval
from diff import interval_differential_estimate

def cut(v, lo=-1.0, hi=1.0):
    return max(lo, min(hi, v))

def derivative_interval_by_sampling(df, X, n=2000):
    xs = np.linspace(X.a, X.b, n)
    vals = np.array([df(float(t)) for t in xs], dtype=float)
    return (float(vals.min()), float(vals.max()))

def bcf_centers_from_derivative_interval(df, X, n=2000):
    dmin, dmax = derivative_interval_by_sampling(df, X, n=n)
    mid = 0.5 * (dmin + dmax)
    rad = 0.5 * (dmax - dmin)

    # если производная практически константа, p можно взять 0 (оба центра = midX)
    if rad == 0.0:
        p = 0.0
    else:
        p = cut(mid / rad, -1.0, 1.0)

    c_mid = X.mid()
    xrad = X.rad()
    c_star = c_mid - p * xrad
    c_hat  = c_mid + p * xrad
    return p, c_star, c_hat, (dmin, dmax)

def bauman_bicentered_interval(f, x, interval):
    a, b = interval
    c = (a + b) / 2  # центр интервала
    h = (b - a) / 2  # полуразмах

    # Узлы Баумана
    x1 = c - h / np.sqrt(3)
    x2 = c + h / np.sqrt(3)

    # значения в узлах
    f_x1 = float(f.subs(x, x1))
    f_x2 = float(f.subs(x, x2))

    # первая и вторая производные
    f1 = sp.diff(f, x)
    f2 = sp.diff(f1, x)

    # значения в центре
    f_c = float(f.subs(x, c))
    f1_c = float(f1.subs(x, c))

    # интервальная оценка второй производной на [a,b]
    f2_func = sp.lambdify(x, f2, "numpy")
    xs = np.linspace(a, b, 400)
    f2_vals = f2_func(xs)
    f2_interval = (float(min(f2_vals)), float(max(f2_vals)))

    # интервал X=[a,b] => X-c = [-h, +h]
    X_minus_c = (-h, h)

    # линейная часть
    lin = (f1_c * X_minus_c[0], f1_c * X_minus_c[1])
    lin = (min(lin), max(lin))

    # квадратичная часть
    quad = (
        0.5 * f2_interval[0] * (min(X_minus_c)**2),
        0.5 * f2_interval[1] * (max(X_minus_c)**2)
    )
    quad = (min(quad), max(quad))

    # итоговая интервальная оценка
    interval_estimate = (
        f_c + lin[0] + quad[0],
        f_c + lin[1] + quad[1]
    )

    return {
        "interval": interval,
        "center c": c,
        "Bauman nodes": (x1, x2),
        "f(x1), f(x2)": (f_x1, f_x2),
        "f(c)": f_c,
        "f'(c)": f1_c,
        "f'' interval": f2_interval,
        "bicentered_interval": interval_estimate
    }

def f1(x) -> float:
    return x**3 - 3*x**2 + 2

def f2(x) -> float:
    return x**2 * math.exp(-x)

def df1(x) -> float: 
    return 3*x**2 - 6*x

def df2(x) -> float:
    return x*(2-x)*math.exp(-x)

X1 = Interval(0, 3)
X2 = Interval(-2, 4)

p1, c1_star, c1_hat, d1_interval = bcf_centers_from_derivative_interval(df1, X1)
print(f"f1: f'(X)={d1_interval}, p={p1}, c*={c1_star}, c^={c1_hat}")
estimate = interval_differential_estimate(f1, df1, c1_star, X1)
print(f"Интервальная оценка f1 для точки c*={c1_star}:", estimate)
estimate = interval_differential_estimate(f1, df1, c1_hat, X1)
print(f"Интервальная оценка f1 для точки c^={c1_hat}:", estimate)

p2, c2_star, c2_hat, d2_interval = bcf_centers_from_derivative_interval(df2, X2)
print(f"f2: f'(X)={d2_interval}, p={p2}, c*={c2_star}, c^={c2_hat}")
estimate = interval_differential_estimate(f2, df2, c2_star, X2)
print(f"Интервальная оценка f2 для точки c*={c2_star}:", estimate)
estimate = interval_differential_estimate(f2, df2, c2_hat, X2)
print(f"Интервальная оценка f2 для точки c^={c2_hat}:", estimate)
