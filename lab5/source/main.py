import numpy as np

from interval import Interval
from method import approximate_upper_solution, krawczyk_step
from utility import plot_case


def run_all(xc_list=(0.0, 0.5, 1.0, 1.2), yc=0.0, iters=3, init_halfwidth=(0.15, 0.15), x_0_shift=(0, 0)):
    results = {}

    for xc in xc_list:
        x_approx = approximate_upper_solution(xc)
        print("\n" + "=" * 80)
        print(f"xc = {xc}, yc = {yc}")
        if x_approx is None:
            print("Не найдено действительных пересечений (по приближённому анализу).")
            results[xc] = None
            continue

        x_approx[0] += x_0_shift[0]
        x_approx[1] += x_0_shift[1]

        # A.1: выбрать стартовый брус X(0)
        hw1, hw2 = init_halfwidth
        X0 = np.array(
            [
                Interval(x_approx[0] - hw1, x_approx[0] + hw1),
                Interval(x_approx[1] - hw2, x_approx[1] + hw2),
            ],
            dtype=object,
        )

        print(f"approx solution (upper): x ~= [{x_approx[0]:.8f}, {x_approx[1]:.8f}]")
        print(f"X^(0) = {X0[0]} x {X0[1]}")

        boxes = [X0]
        X = X0

        # A.2/A.3: ≥ 3 итераций Кравчика
        for k in range(1, iters + 1):
            Xnext, meta = krawczyk_step(X, xc, yc)
            if Xnext is None:
                print(f"Iteration {k}: STOP ({meta.get('reason','unknown')})")
                break
            boxes.append(Xnext)
            print(
                f"X^({k}) = {Xnext[0]} x {Xnext[1]}   (widths: {Xnext[0].width():.3e}, {Xnext[1].width():.3e})"
            )
            X = Xnext

        results[xc] = boxes

        # B.1: график
        plot_case(xc, boxes, yc=yc)

    return results


if __name__ == "__main__":
    # Можно менять iters (>=3) и размеры стартового бокса.
    run_all(
        xc_list=(0.0, 0.5, 1.0, 1.2),
        yc=0.0,
        iters=4,
        init_halfwidth=(
            0.10,
            0.10,
        ),  # попробуйте (0.10,0.10) или (0.30,0.30) для анализа влияния X(0)
    )
