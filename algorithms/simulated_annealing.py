import numpy as np


def simulated_annealing(func, iterations=500, T0=100, Tmin=0.5, alpha=0.95):
    """
    Simulated Annealing (SA) pro libovolnou funkci.

    Args:
        func: instance testovací funkce (musí mít .evaluate(), .lower_bound, .upper_bound, .dimension)
        T0: počáteční teplota
        Tmin: minimální teplota (stop podmínka)
        alpha: chladicí koeficient (0 < alpha < 1)
        max_iter: maximální počet iterací (bez ohledu na Tmin)

    Returns:
        best_x: nejlepší nalezený vektor parametrů
        best_f: hodnota funkce v best_x
        history: seznam všech navštívených bodů [(x, f), ...]
        :param func:
        :param alpha:
        :param Tmin:
        :param T0:
        :param iterations:
    """
    # inicializace
    T = T0
    x = np.random.uniform(func.lower_bound, func.upper_bound, func.dimension)
    f = func.evaluate(x)

    best_x, best_f = np.copy(x), f
    history = [(np.copy(x), f)]

    iteration = 0
    while T > Tmin and iteration < iterations:
        # vygeneruj souseda
        x_new = x + np.random.normal(0, 1, func.dimension)
        # ořež do domény
        x_new = np.clip(x_new, func.lower_bound, func.upper_bound)

        f_new = func.evaluate(x_new)
        history.append((np.copy(x_new), f_new))

        # akceptační kritérium
        if f_new < f:
            x, f = x_new, f_new
            if f_new < best_f:
                best_x, best_f = np.copy(x_new), f_new
        else:
            r = np.random.rand()
            if r < np.exp(-(f_new - f) / T):
                x, f = x_new, f_new

        # ochlazování
        T *= alpha
        iteration += 1

    return best_x, best_f, history
