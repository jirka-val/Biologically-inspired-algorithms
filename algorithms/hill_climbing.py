import numpy as np


def hill_climbing(func, iterations=500, neighbors=8, step_size=0.1):
    """
    Hill Climbing algoritmus.

    Args:
        func: instance testovací funkce (musí mít .evaluate(), .lower_bound, .upper_bound, .dimension)
        iterations: maximální počet iterací
        neighbors: kolik sousedů generovat v každém kroku
        step_size: směrodatná odchylka pro generování sousedů (normal distribution)

    Returns:
        best_x: nejlepší nalezený vektor parametrů
        best_f: hodnota funkce v best_x
        history: seznam všech navštívených bodů [(x, f), ...]
    """
    # start: náhodné řešení
    x_current = np.random.uniform(func.lower_bound, func.upper_bound, func.dimension)
    f_current = func.evaluate(x_current)

    best_x, best_f = x_current, f_current
    history = [(x_current, f_current)]

    for _ in range(iterations):
        # generování sousedů (NP sousedů kolem x_current)
        neighbors_x = np.random.normal(loc=x_current, scale=step_size,
                                       size=(neighbors, func.dimension))

        # udržet v doméně funkce
        neighbors_x = np.clip(neighbors_x, func.lower_bound, func.upper_bound)

        # vyhodnocení sousedů
        values = [func.evaluate(x) for x in neighbors_x]

        # najít nejlepšího souseda
        idx_best = np.argmin(values)
        x_best_neighbor = neighbors_x[idx_best]
        f_best_neighbor = values[idx_best]

        # aktualizace
        if f_best_neighbor < f_current:
            x_current, f_current = x_best_neighbor, f_best_neighbor

            if f_best_neighbor < best_f:
                best_x, best_f = x_best_neighbor, f_best_neighbor

        history.append((x_current, f_current))

    return best_x, best_f, history
