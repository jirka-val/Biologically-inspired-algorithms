import numpy as np
from copy import deepcopy


class Solution:
    """Reprezentace jednoho řešení pro Differential Evolution."""
    def __init__(self, dimension, lower_bound, upper_bound):
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.params = np.random.uniform(lower_bound, upper_bound, dimension)
        self.f = None  # fitness hodnota


def differential_evolution(function, NP=30, F=0.8, CR=0.9, G=200):
    """
    Differential Evolution (DE) algoritmus.
    :param function: instance třídy Function (např. Griewank)
    :param NP: velikost populace
    :param F: faktor mutace
    :param CR: crossover rate
    :param G: počet generací
    :return: nejlepší nalezené řešení, jeho fitness a historie
    """
    dimension = function.dimension
    lower = function.lower_bound
    upper = function.upper_bound

    # inicializace populace
    pop = [Solution(dimension, lower, upper) for _ in range(NP)]
    for ind in pop:
        ind.f = function.evaluate(ind.params)

    # uložíme nejlepší
    best = min(pop, key=lambda s: s.f)
    history = [(best.params.copy(), best.f)]

    for g in range(G):
        new_pop = deepcopy(pop)

        for i, x_i in enumerate(pop):
            # random na začatek
            indices = list(range(NP))
            indices.remove(i)
            r1, r2, r3 = np.random.choice(indices, 3, replace=False)

            x_r1, x_r2, x_r3 = pop[r1], pop[r2], pop[r3]

            # mutace
            v = x_r3.params + F * (x_r1.params - x_r2.params)

            # ošetření hranic
            v = np.clip(v, lower, upper)

            # křížení
            u = np.zeros(dimension)
            j_rand = np.random.randint(0, dimension)
            for j in range(dimension):
                if np.random.uniform() < CR or j == j_rand:
                    u[j] = v[j]
                else:
                    u[j] = x_i.params[j]

            # vyhodnocení nové varianty
            f_u = function.evaluate(u)

            # Selecke
            if f_u <= x_i.f:  # minimalizace
                new_pop[i].params = u
                new_pop[i].f = f_u

        pop = new_pop

        # aktuálně nejlepší řešení
        best = min(pop, key=lambda s: s.f)
        history.append((best.params.copy(), best.f))

    return best.params, best.f, history
