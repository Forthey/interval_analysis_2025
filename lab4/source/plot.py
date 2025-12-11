import numpy as np
import matplotlib.pyplot as plt

S_GRID_POINTS = 1000

model_to_name = {
    "add": "аддитивная",
    "mul": "мультипликативная"
}

# --- Plot optimization ---
def plot_opt(name, func, X, Y, lb, ub, model, n_points=S_GRID_POINTS):
    vals = np.linspace(lb, ub, n_points)
    Ji = [func(v, X, Y) for v in vals]
    idx = np.argmax(Ji)
    s_hat, Ji_max = vals[idx], Ji[idx]

    plt.figure(figsize=(8, 5))
    plt.plot(vals, Ji, "b-", color="black", linewidth=2, label=f"{name}")
    plt.axvline(
        s_hat, color="gray", linestyle="--", linewidth=2, label=f"smax = {s_hat:.4f}"
    )
    plt.scatter([s_hat], [Ji_max], color="red", s=100, zorder=5)

    plt.xlabel("s(a)" if model == "add" else "s(t)")
    plt.ylabel("F(s)")
    title_str = f"Метод {name}"
    plt.title(
        f"{title_str} {model_to_name[model]} Модель\ns = {s_hat:.4f}, Ji = {Ji_max:.4f}"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/{name}_{model}.png", dpi=150, bbox_inches="tight")

    return s_hat, Ji_max
