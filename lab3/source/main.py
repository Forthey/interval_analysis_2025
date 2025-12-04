"""
Решение задачи с интервальными системами линейных уравнений (2 переменные x=(x1,x2)).

Подход и допущения (важно, читаем внимательно):
- Интервальная матрица A~ и вектор b~ задаются через средние значения (mid) и
  радиусы (rad): A~ = [A_mid - A_rad, A_mid + A_rad], b~ = [b_mid - b_rad, b_mid + b_rad].
- Используется стандартный *tolerance functional* (см. литературу по интервальной
  алгебре): для каждой строки i вводим

    Tol_i(x) = b_rad[i] - |A_mid[i] @ x - b_mid[i]| - sum_j A_rad[i,j] * |x_j|

  и

    Tol(x) = min_i Tol_i(x).

  Тогда условие Tol(x) >= 0 является достаточным и необходимым условием
  того, что 0 лежит в пересечении интервального остатка для соответствующей строки
  (соответствует условию пересечения интервалов левой и правой частей).

- Мы считаем допусковое множество непустым, если существует x такой, что Tol(x) >= 0.

- Коррекции (B.1, B.2, B.3) реализованы как минимальные изменения средних значений
  (mid) правой части и/или средних значений A (в смысле L2-нормы), при которых
  система становится разрешимой (т.е. существует x: Tol(x) >= 0). Радиусы интервалов
  (A_rad, b_rad) при коррекции остаются неизменными.

  Эта постановка — практическое приближение: другие варианты (изменение радиусов,
  L_inf-минимизация и т.д.) возможны и иногда более естественны для конкретных задач.

Функциональность кода:
- Функции для вычисления Tol(x) и Tol_i(x)
- Поиск argmax Tol(x) (глобальное приближение методом перебора на сетке + локальная оптимизация)
- Построение графика Tol(x) (контурный график) и отметка максимума
- Построение допускового множества (множество точек на сетке, где Tol(x) >= 0)
- Коррекции b, A, Ab: оптимизация с ограничением Tol(x) >= 0
- Сравнение коррекций и визуализация результатов для каждого варианта

Пример: в конце задаются три системы (A1,b1), (A2,b2), (A3,b3) — пользователь может
заменить их своими данными.

Зависимости: numpy, scipy, matplotlib

"""

import numpy as np
import re
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from functools import partial

# ----------------- Базовые функции -----------------

def tol_components(A_mid, A_rad, b_mid, b_rad, x):
    """Возвращает вектор Tol_i(x) для каждой строки i.

    A_mid: (m,2), A_rad: (m,2), b_mid: (m,), b_rad: (m,), x: (2,)
    """
    Ax_mid = A_mid @ x  # (m,)
    abs_diff = np.abs(Ax_mid - b_mid)
    sum_arad_absx = np.sum(A_rad * np.abs(x), axis=1)
    return b_rad - abs_diff - sum_arad_absx


def tol(A_mid, A_rad, b_mid, b_rad, x):
    """Tolerance functional: min_i Tol_i(x)."""
    return np.min(tol_components(A_mid, A_rad, b_mid, b_rad, x))


# ----------------- Поиск максимума Tol(x) -----------------

def find_argmax_tol(A_mid, A_rad, b_mid, b_rad, x0=None, grid_bounds=None, grid_n=161):
    """Ищет аргмакс Tol(x) в плоскости.

    Алгоритм: грубый перебор по сетке (grid_n x grid_n) в заданных границах,
    затем локальная оптимизация (SLSQP) от лучших нескольких начальных точек.

    Возвращает (x_max, tol_max, info) где x_max -- найденная точка, tol_max = Tol(x_max).
    """
    # default bounds: вокруг 0
    if grid_bounds is None:
        grid_bounds = (-5, 5, -5, 5)  # xmin,xmax,ymin,ymax
    xmin, xmax, ymin, ymax = grid_bounds

    xs = np.linspace(xmin, xmax, grid_n)
    ys = np.linspace(ymin, ymax, grid_n)

    Xg, Yg = np.meshgrid(xs, ys)
    pts = np.vstack([Xg.ravel(), Yg.ravel()]).T

    # evaluate on grid
    vals = np.array([tol(A_mid, A_rad, b_mid, b_rad, p) for p in pts])
    idx_best = np.argmax(vals)
    x_best = pts[idx_best]
    val_best = vals[idx_best]

    # локальная оптимизация (maximization -> minimize negative)
    def obj(x):
        return -tol(A_mid, A_rad, b_mid, b_rad, x)

    # попробуем несколько начальных точек: лучшие 5 с сетки
    top_k = np.argsort(-vals)[:5]
    best_candidate = x_best.copy()
    best_val = val_best
    for idx in top_k:
        x0 = pts[idx]
        res = minimize(obj, x0, method='SLSQP', bounds=None, options={'ftol':1e-9, 'maxiter':500})
        if not res.success:
            continue
        v = -res.fun
        if v > best_val:
            best_val = v
            best_candidate = res.x

    info = {'grid_best': x_best, 'grid_val': val_best}
    return best_candidate, best_val, (Xg, Yg, vals.reshape(Xg.shape))


# ----------------- Визуализация -----------------

def plot_tol_and_feasible(A_mid, A_rad, b_mid, b_rad, x_max, tol_max, grid_data, title_suffix=''):
    Xg, Yg, Z = grid_data
    plt.figure(figsize=(10,4))

    # контур Tol
    plt.subplot(1,2,1)
    cs = plt.contourf(Xg, Yg, Z, levels=40)
    plt.colorbar(cs)
    plt.scatter([x_max[0]], [x_max[1]], marker='*', s=100, color='white')
    plt.title('Tol(x) (максимум помечен) '+title_suffix)
    plt.xlabel('x1'); plt.ylabel('x2')

    # допусковое множество (Tol >= 0)
    plt.subplot(1,2,2)
    feasible = (Z >= 0)
    plt.contourf(Xg, Yg, feasible.astype(float), levels=[-0.1,0.5,1.1])
    plt.scatter([x_max[0]], [x_max[1]], marker='*', s=100, color='red')
    plt.title('Допусковое множество (Tol >= 0) '+title_suffix)
    plt.xlabel('x1'); plt.ylabel('x2')

    plt.tight_layout()
    plt.show()


# ----------------- Коррекции -----------------

def b_correction_min_mid(A_mid, A_rad, b_mid, b_rad, x0=None):
    """Минимальная по L2 корректировка средних b_mid (delta_b) такая, что
    существует x с Tol(x; b_mid+delta_b) >= 0.

    Переменные оптимизации: [delta_b_0..m-1, x1, x2]
    Минимизируем ||delta_b||_2^2
    Ограничение: Tol(x; b_mid+delta_b) >= 0
    """
    m = len(b_mid)

    if x0 is None:
        x0 = np.zeros(2)
    z0 = np.concatenate([np.zeros(m), x0])

    def obj(z):
        delta_b = z[:m]
        return np.sum(delta_b**2)

    def cons(z):
        delta_b = z[:m]
        x = z[m:]
        return tol(A_mid, A_rad, b_mid + delta_b, b_rad, x)

    cons_dict = ({'type':'ineq', 'fun': cons},)

    res = minimize(obj, z0, constraints=cons_dict, method='SLSQP', options={'ftol':1e-9, 'maxiter':1000})
    if not res.success:
        print('b-correction: оптимизатор не сошелся:', res.message)
    delta_b = res.x[:m]
    x = res.x[m:]
    return delta_b, x, res


def A_correction_min_mid(A_mid, A_rad, b_mid, b_rad, x0=None):
    """Минимальная по L2 корректировка средних элементов A_mid (векторизованно),
    такая, что существует x с Tol(...) >= 0.

    Переменные: [delta_A_flat (m*2), x1, x2]
    Минимизируем ||delta_A||_2^2
    Ограничение: Tol(x; A_mid+delta_A, A_rad, b_mid, b_rad) >= 0
    """
    m = A_mid.shape[0]
    if x0 is None:
        x0 = np.zeros(2)
    z0 = np.concatenate([np.zeros(m*2), x0])

    def obj(z):
        deltaA = z[:m*2]
        return np.sum(deltaA**2)

    def cons(z):
        deltaA = z[:m*2].reshape((m,2))
        x = z[m*2:]
        return tol(A_mid + deltaA, A_rad, b_mid, b_rad, x)

    cons_dict = ({'type':'ineq', 'fun': cons},)
    res = minimize(obj, z0, constraints=cons_dict, method='SLSQP', options={'ftol':1e-9, 'maxiter':2000})
    if not res.success:
        print('A-correction: оптимизатор не сошелся:', res.message)
    deltaA = res.x[:m*2].reshape((m,2))
    x = res.x[m*2:]
    return deltaA, x, res


def Ab_correction_min_mid(A_mid, A_rad, b_mid, b_rad, x0=None):
    """Одновременная коррекция A_mid и b_mid минимальная по сумме квадратов.
    Переменные: [deltaA_flat, delta_b, x1, x2]
    """
    m = A_mid.shape[0]
    if x0 is None:
        x0 = np.zeros(2)
    z0 = np.concatenate([np.zeros(m*2), np.zeros(m), x0])

    def obj(z):
        deltaA = z[:m*2]
        deltaB = z[m*2:m*2+m]
        return np.sum(deltaA**2) + np.sum(deltaB**2)

    def cons(z):
        deltaA = z[:m*2].reshape((m,2))
        deltaB = z[m*2:m*2+m]
        x = z[m*2+m:]
        return tol(A_mid + deltaA, A_rad, b_mid + deltaB, b_rad, x)

    cons_dict = ({'type':'ineq', 'fun': cons},)
    res = minimize(obj, z0, constraints=cons_dict, method='SLSQP', options={'ftol':1e-9, 'maxiter':3000})
    if not res.success:
        print('Ab-correction: оптимизатор не сошелся:', res.message)
    deltaA = res.x[:m*2].reshape((m,2))
    deltaB = res.x[m*2:m*2+m]
    x = res.x[m*2+m:]
    return deltaA, deltaB, x, res


# ----------------- Утилиты для сравнения -----------------

def analyze_system(A_mid, A_rad, b_mid, b_rad, system_name='system', grid_bounds=None):
    """Выполняет все шаги A, B, C для одной интервальной системы.

    Возвращает словарь с результатами и рисует графики.
    """
    print('\n=== Анализ системы:', system_name, '===')

    # A. Проверка непустоты допускового множества
    x_max, tol_max, grid_data = find_argmax_tol(A_mid, A_rad, b_mid, b_rad, grid_bounds=grid_bounds)
    print('Максимум Tol:', tol_max, 'в точке', x_max)

    feasible = (tol_max >= 0)
    if feasible:
        print('A: Допусковое множество НЕпусто.')
        plot_tol_and_feasible(A_mid, A_rad, b_mid, b_rad, x_max, tol_max, grid_data, title_suffix=f'({system_name})')
        results = {'feasible': True, 'x_max': x_max, 'tol_max': tol_max, 'grid_data': grid_data}
        # также возвращаем пустые поля для коррекций
        results.update({'b_corr': None, 'A_corr': None, 'Ab_corr': None})
        return results

    print('A: Допусковое множество ПУСТО. Переходим к корректировкам...')

    # B.1 b-correction
    delta_b, x_b, res_b = b_correction_min_mid(A_mid, A_rad, b_mid, b_rad, x0=x_max)
    b_mid_b = b_mid + delta_b
    print('\nB.1 b-correction: ||delta_b||_2 =', np.linalg.norm(delta_b))
    xb_max, tolb_max, grid_b = find_argmax_tol(A_mid, A_rad, b_mid_b, b_rad, grid_bounds=grid_bounds)
    print('  Tol после b-коррекции (максимум):', tolb_max, 'в точке', xb_max)
    plot_tol_and_feasible(A_mid, A_rad, b_mid_b, b_rad, xb_max, tolb_max, grid_b, title_suffix=f'({system_name} - b-corr)')

    # B.2 A-correction
    deltaA, x_A, res_A = A_correction_min_mid(A_mid, A_rad, b_mid, b_rad, x0=x_max)
    A_mid_A = A_mid + deltaA
    print('\nB.2 A-correction: ||delta_A||_F =', np.linalg.norm(deltaA))
    xa_max, tolA_max, grid_A = find_argmax_tol(A_mid_A, A_rad, b_mid, b_rad, grid_bounds=grid_bounds)
    print('  Tol после A-коррекции (максимум):', tolA_max, 'в точке', xa_max)
    plot_tol_and_feasible(A_mid_A, A_rad, b_mid, b_rad, xa_max, tolA_max, grid_A, title_suffix=f'({system_name} - A-corr)')

    # B.3 Ab-correction
    deltaA_ab, deltaB_ab, x_ab, res_ab = Ab_correction_min_mid(A_mid, A_rad, b_mid, b_rad, x0=x_max)
    A_mid_ab = A_mid + deltaA_ab
    b_mid_ab = b_mid + deltaB_ab
    print('\nB.3 Ab-correction: ||delta_A||_F =', np.linalg.norm(deltaA_ab), ', ||delta_b||_2 =', np.linalg.norm(deltaB_ab))
    xab_max, tolab_max, grid_ab = find_argmax_tol(A_mid_ab, A_rad, b_mid_ab, b_rad, grid_bounds=grid_bounds)
    print('  Tol после Ab-коррекции (максимум):', tolab_max, 'в точке', xab_max)
    plot_tol_and_feasible(A_mid_ab, A_rad, b_mid_ab, b_rad, xab_max, tolab_max, grid_ab, title_suffix=f'({system_name} - Ab-corr)')

    # C. Сравнение влияния коррекций
    print('\n--- Сравнение коррекций ---')
    print('Исходная Tol_max:', tol_max)
    print('b-corr: ||delta_b||_2 =', np.linalg.norm(delta_b), ', Tol_max =', tolb_max)
    print('A-corr: ||delta_A||_F =', np.linalg.norm(deltaA), ', Tol_max =', tolA_max)
    print('Ab-corr: ||delta_A||_F =', np.linalg.norm(deltaA_ab), ', ||delta_b||_2 =', np.linalg.norm(deltaB_ab), ', Tol_max =', tolab_max)

    # Оцениваем какое искажение меньше: используем норму отклонений в сравнение
    distortions = {
        'b': np.linalg.norm(delta_b),
        'A': np.linalg.norm(deltaA),
        'Ab': np.sqrt(np.linalg.norm(deltaA_ab)**2 + np.linalg.norm(deltaB_ab)**2)
    }
    sorted_dist = sorted(distortions.items(), key=lambda kv: kv[1])
    print('Наименее искажающая коррекция (по L2/LF):', sorted_dist[0])

    results = {
        'feasible': False,
        'x_max': x_max, 'tol_max': tol_max,
        'b_corr': {'delta_b': delta_b, 'x': x_b, 'tol_max': tolb_max, 'b_mid_new': b_mid_b},
        'A_corr': {'delta_A': deltaA, 'x': x_A, 'tol_max': tolA_max, 'A_mid_new': A_mid_A},
        'Ab_corr': {'delta_A': deltaA_ab, 'delta_b': deltaB_ab, 'x': x_ab, 'tol_max': tolab_max, 'A_mid_new': A_mid_ab, 'b_mid_new': b_mid_ab},
        'distortions': distortions,
    }
    return results


# ----------------- Парсинг матриц из файла -----------------

interval_re = re.compile(r'\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]')

def parse_interval_row(line):
    """Парсит строку вида: [0.65,1.25] [0.70,1.30]  → список (mid, rad)."""
    matches = interval_re.findall(line)
    mids = []
    rads = []
    for a, b in matches:
        a = float(a)
        b = float(b)
        mid = (a + b) / 2
        rad = abs(b - a) / 2
        mids.append(mid)
        rads.append(rad)
    return mids, rads


def parse_interval_systems_intervals(filename):
    """
    Парсинг систем, заданных ИНТЕРВАЛАМИ, например:
    
    SYSTEM A1
    A:
    [0.65,1.25]  [0.70,1.30]
    [0.75,1.35]  [0.70,1.30]
    
    b:
    [2.75,3.15]
    [2.85,3.25]
    END

    Возвращает список систем в форме:
        (name, A_mid, A_rad, b_mid, b_rad)
    """
    systems = []
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    i = 0
    while i < len(lines):
        if not lines[i].startswith("SYSTEM"):
            i += 1
            continue

        name = lines[i].split(maxsplit=1)[1] if " " in lines[i] else f"System_{len(systems)+1}"
        i += 1

        # --- A matrix ---
        if lines[i].rstrip(':').upper() != "A":
            print(lines[i].rstrip(':').upper())
            raise ValueError(f"Ожидалось 'A:' в строке {lines[i]}")
        i += 1

        A_mid_rows = []
        A_rad_rows = []
        while i < len(lines) and lines[i][0] == '[':
            mids, rads = parse_interval_row(lines[i])
            A_mid_rows.append(mids)
            A_rad_rows.append(rads)
            i += 1

        A_mid = np.array(A_mid_rows, float)
        A_rad = np.array(A_rad_rows, float)

        # --- b vector ---
        if lines[i].rstrip(':').lower() != "b":
            raise ValueError(f"Ожидалось 'b:' в строке {lines[i]}")
        i += 1

        b_mid = []
        b_rad = []
        while i < len(lines) and lines[i][0] == '[':
            mids, rads = parse_interval_row(lines[i])
            if len(mids) != 1:
                raise ValueError("b-vector должен быть одномерным.")
            b_mid.append(mids[0])
            b_rad.append(rads[0])
            i += 1

        b_mid = np.array(b_mid)
        b_rad = np.array(b_rad)

        if lines[i] != "END":
            raise ValueError(f"Ожидалось END, получено: {lines[i]}")
        i += 1

        systems.append((name, A_mid, A_rad, b_mid, b_rad))

    return systems

# ----------------- Пример использования -----------------
if __name__ == '__main__':
    systems = parse_interval_systems_intervals("var1.txt")


    # Анализ всех систем
    for name, A_mid, A_rad, b_mid, b_rad in systems:
        # диапазоны сетки для визуализации стоит подбирать под задачу;
        # здесь простой общий выбор
        grid_bounds = (-3, 3, -3, 3)
        res = analyze_system(A_mid, A_rad, b_mid, b_rad, system_name=name, grid_bounds=grid_bounds)
