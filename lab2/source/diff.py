from base import Interval

def interval_differential_estimate(f, df, x, interval):
    # f(x_c)
    fx = f(x)

    # интервал f'(X)
    dfX = Interval(df(interval.a), df(interval.b))

    # интервал (X - x_c)
    X_minus_x = Interval(interval.a - x, interval.b - x)

    # f(x_c) как интервал
    fx_interval = Interval(fx, fx)

    # f(x_c) + f'(X)*(X - x_c)
    return fx_interval + dfX * X_minus_x
