import numpy as np
from copy import deepcopy


class Individual:
    """Reprezentace jednoho jedince pro SOMA All-to-One."""
    def __init__(self, dimension, lower_bound, upper_bound):
        self.dimension = dimension
        self.position = np.random.uniform(lower_bound, upper_bound, dimension)
        self.fitness = None


def soma_all_to_one(function, pop_size=20, PRT=0.4, path_length=3.0, step=0.11, M_max=100):
    """
    SOMA All-to-One (Self-Organizing Migrating Algorithm)
    Implementace podle prezentace prof. Zelinky (09c BIA – Algorithms.pptx)
    """
    dim = function.dimension
    lower, upper = function.lower_bound, function.upper_bound

    # inicializace populace
    population = [Individual(dim, lower, upper) for _ in range(pop_size)]
    for ind in population:
        ind.fitness = function.evaluate(ind.position)

    # historie pro vizualizaci
    history = [[(ind.position.copy(), ind.fitness) for ind in population]]

    # pomocné funkce
    def get_leader(pop):
        return min(pop, key=lambda i: i.fitness)

    def get_prt_vector(prt, dim):
        """Náhodný binární vektor s pravděpodobností prt."""
        return np.random.rand(dim) < prt

    def clip_coords(position):
        return np.clip(position, lower, upper)

    # Hlavní cyklus migrací
    for migration in range(M_max):
        leader = get_leader(population)

        for individual in population:
            # Leader se nehýbe
            if np.all(individual.position == leader.position):
                continue

            best_pos = individual.position.copy()
            best_fit = individual.fitness

            prt_vector = get_prt_vector(PRT, dim)

            # Pohyb po krocích směrem k leaderovi
            for t in np.arange(step, path_length + step, step):
                # Výpočet nové pozice
                pos = individual.position + (leader.position - individual.position) * t * prt_vector
                pos = clip_coords(pos)
                fit = function.evaluate(pos)

                # Pokud je lepší – uložit
                if fit < best_fit:
                    best_fit = fit
                    best_pos = pos.copy()

                # Pokud jsme na hraně prostoru – přeruš migraci
                if np.any(np.abs(pos) >= upper):
                    break

            # Přesun na nejlepší nalezenou pozici
            individual.position = best_pos
            individual.fitness = best_fit

        # Uložit po každé migraci
        history.append([(ind.position.copy(), ind.fitness) for ind in population])

    # Výsledek
    leader = get_leader(population)
    return leader.position, leader.fitness, history
