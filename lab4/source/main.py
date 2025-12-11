import numpy as np
import intvalpy as ip
import matplotlib.pyplot as plt

ip.precision.extendedPrecisionQ = False

from read import read_bin_file
from plot import plot_opt
import utility as ul

# --- Jaccard ---
def jaccard_interval(x, y):
    a1, b1 = float(x.a), float(x.b)
    a2, b2 = float(y.a), float(y.b)
    numerator = min(b1, b2) - max(a1, a2)
    denominator = max(b1, b2) - min(a1, a2)
    return numerator / denominator


def coefficient_Jaccard(X, Y=None):
    if Y is None:
        lowers = [float(x.a) for x in X]
        uppers = [float(x.b) for x in X]
        return (min(uppers) - max(lowers)) / (max(uppers) - min(lowers))
    if isinstance(X, ip.ClassicalArithmetic) and isinstance(Y, ip.ClassicalArithmetic):
        return jaccard_interval(X, Y)
    return np.mean([jaccard_interval(x, y) for x, y in zip(X, Y)])


def main():
    filename_x = "-0.205_lvl_side_a_fast_data.bin"
    filename_y = "0.225_lvl_side_a_fast_data.bin"
    rad_x = rad_y = 1.0 / (2**14)

    # ----- A -----
    x_data = ul.get_avg(read_bin_file(filename_x))
    y_data = ul.get_avg(read_bin_file(filename_y))
    X = ul.scalar_to_interval_vec(x_data, rad_x).flatten()
    Y = ul.scalar_to_interval_vec(y_data, rad_y).flatten()
    print(f"Final dataset: {len(X)} samples")

    bound_a_l = float(np.min(Y).a) - float(np.max(X).b)
    bound_a_r = float(np.max(Y).b) - float(np.min(X).a)

    bound_t_l = float(np.min(Y).a) / float(np.max(X).b)
    bound_t_r = float(np.max(Y).b) / float(np.min(X).a)
    # ----- B -----
    models = ["add", "mul"]
    methods = ["B1", "B2", "B3", "B4"]
    method_names = {"B1": "full", "B2": "mode", "B3": "medK", "B4": "medP"}
    # Define function dictionaries
    func_a_dict = {
        "B1": lambda a, X, Y: np.mean(coefficient_Jaccard(X + a, Y)),
        "B2": lambda a, X, Y: np.mean(coefficient_Jaccard(ul.Mode(X + a), ul.Mode(Y))),
        "B3": lambda a, X, Y: np.mean(coefficient_Jaccard(ul.MedK(X + a), ul.MedK(Y))),
        "B4": lambda a, X, Y: np.mean(coefficient_Jaccard(ul.MedP(X + a), ul.MedP(Y))),
    }

    func_t_dict = {
        "B1": lambda t, X, Y: np.mean(coefficient_Jaccard(X * t, Y)),
        "B2": lambda t, X, Y: np.mean(coefficient_Jaccard(ul.Mode(X * t), ul.Mode(Y))),
        "B3": lambda t, X, Y: np.mean(coefficient_Jaccard(ul.MedK(X * t), ul.MedK(Y))),
        "B4": lambda t, X, Y: np.mean(coefficient_Jaccard(ul.MedP(X * t), ul.MedP(Y))),
    }

    all_results = {}

    for model in models:
        print(f"\nOptimizing for \"{model}\" model")
        model_results = {}

        for method in methods:
            print(f"\tMethod {method} ({method_names[method]})")
            if model == "add":
                func = func_a_dict[method]
                lb = bound_a_l
                ub = bound_a_r
            else:
                func = func_t_dict[method]
                lb = -1.2
                ub = bound_t_r
            s_hat, Ji_max = plot_opt(method, func, X, Y, lb, ub, model)

            model_results[method] = {"s_hat": s_hat, "Ji_max": Ji_max}
            print(f"\t\ts_hat = {s_hat:.6f}\n\t\tJi_max = {Ji_max:.6f}")

        all_results[model] = model_results

    # --- D ---
    print("\nSummary Table")
    print("\tMethod | Add Model (a) | Ji_add | Mul Model (t) | Ji_mul")
    print("\t-------|---------------|--------|---------------|--------")
    for method in methods:
        add_result = all_results["add"][method]
        mul_result = all_results["mul"][method]
        print(
            f"\t{method:6} | {add_result['s_hat']:13.6f} | {add_result['Ji_max']:6.4f} | "
            f"{mul_result['s_hat']:13.6f} | {mul_result['Ji_max']:6.4f}"
        )

    best_add = max(methods, key=lambda m: all_results["add"][m]["Ji_max"])
    best_mul = max(methods, key=lambda m: all_results["mul"][m]["Ji_max"])

    print(
        f"\nBest method for \"add\" model: {best_add} (Ji = {all_results['add'][best_add]['Ji_max']:.6f})"
    )
    print(
        f"Best method for \"mul\" model: {best_mul} (Ji = {all_results['mul'][best_mul]['Ji_max']:.6f})"
    )

    add_values = [all_results["add"][m]["s_hat"] for m in methods]
    mul_values = [all_results["mul"][m]["s_hat"] for m in methods]

    print(f"\nVariation in \"add\" model estimates: {np.std(add_values):.6f}")
    print(f"Variation in \"mul\" model estimates: {np.std(mul_values):.6f}")


if __name__ == "__main__":
    main()
