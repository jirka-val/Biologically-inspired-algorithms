import numpy as np


def blind_search(func, iterations=500):
    """
    Blind Search (náhodné hledání) pro libovolnou funkci.

    Args:
        func: instance testovací funkce (musí mít .evaluate(), .lower_bound, .upper_bound, .dimension)
        iterations: počet náhodných vzorků

    Returns:
        best_x: nejlepší nalezený vektor parametrů
        best_f: hodnota funkce v best_x
        history: seznam všech navštívených bodů [(x, f), ...]
    """
    best_x = None
    best_f = float("inf")
    history = []

    for _ in range(iterations):
        x = np.random.uniform(func.lower_bound, func.upper_bound, func.dimension)
        f = func.evaluate(x)
        history.append((x, f))

        if f < best_f:
            best_x, best_f = x, f

    return best_x, best_f, history
