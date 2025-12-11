import intvalpy as ip
import numpy as np


# --- Medians ---
def Mode(X, n_bins=200):
    lowers = np.array([float(el.a) for el in X])
    uppers = np.array([float(el.b) for el in X])
    grid = np.linspace(lowers.min(), uppers.max(), n_bins)
    counts = np.zeros(len(grid) - 1)
    for i in range(len(grid) - 1):
        counts[i] = np.sum((lowers <= grid[i + 1]) & (uppers >= grid[i]))
    max_count = counts.max()
    bins = [
        (grid[i], grid[i + 1]) for i in range(len(grid) - 1) if counts[i] == max_count
    ]
    merged = []
    curr = bins[0]
    for b in bins[1:]:
        if b[0] <= curr[1]:
            curr = (curr[0], b[1])
        else:
            merged.append(curr)
            curr = b
    merged.append(curr)
    return [ip.Interval(b[0], b[1]) for b in merged]


def MedK(x):
    lowers = [float(el.a) for el in x]
    uppers = [float(el.b) for el in x]
    med_lower = float(np.median(lowers))
    med_upper = float(np.median(uppers))
    return ip.Interval([med_lower, med_upper])


def MedP(x):
    X = sorted(x, key=lambda t: (float(t.a) + float(t.b)) / 2)
    index_med = len(X) // 2
    if len(X) % 2 == 0:
        return (X[index_med - 1] + X[index_med]) / 2
    return X[index_med]


def scalar_to_interval(x, rad):
    return ip.Interval(x - rad, x + rad)


scalar_to_interval_vec = np.vectorize(scalar_to_interval)


def get_avg(data):
    avg = np.zeros((1024, 8))
    for i in range(len(data)):
        avg = np.add(avg, data[i])
    return np.divide(avg, len(data))
