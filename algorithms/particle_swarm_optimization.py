import numpy as np
from copy import deepcopy


class Particle:
    """Reprezentace jedné částice ve swarmu (pro PSO)."""
    def __init__(self, dimension, lower_bound, upper_bound):
        self.dimension = dimension
        self.position = np.random.uniform(lower_bound, upper_bound, dimension)
        self.velocity = np.random.uniform(-abs(upper_bound - lower_bound),
                                          abs(upper_bound - lower_bound), dimension) * 0.1
        self.best_position = self.position.copy()
        self.f = None
        self.best_f = np.inf


def particle_swarm_optimization(function, pop_size=15, c1=2.0, c2=2.0, w=0.7, M_max=50):
    """
    Particle Swarm Optimization (PSO) s inertia weight.
    :param function: instance třídy Function (např. Griewank)
    :param pop_size: počet částic
    :param c1, c2: akcelerační koeficienty (kognitivní a sociální složka)
    :param w: setrvačnost (inertia weight)
    :param M_max: maximální počet iterací
    :return: nejlepší nalezené řešení, jeho fitness, historie (populace v každé iteraci)
    """
    dim = function.dimension
    lower, upper = function.lower_bound, function.upper_bound

    # inicializace swarmu
    swarm = [Particle(dim, lower, upper) for _ in range(pop_size)]
    for p in swarm:
        p.f = function.evaluate(p.position)
        p.best_f = p.f
        p.best_position = p.position.copy()

    # globálně nejlepší částice
    gBest = min(swarm, key=lambda p: p.f)
    gBest_position = gBest.position.copy()
    gBest_value = gBest.f

    history = [[(p.position.copy(), p.f) for p in swarm]]

    # hlavní iterace
    for m in range(M_max - 1):
        for i, p in enumerate(swarm):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)

            # aktualizace rychlosti
            v_new = (
                w * p.velocity
                + c1 * r1 * (p.best_position - p.position)
                + c2 * r2 * (gBest_position - p.position)
            )

            # kontrola hranic rychlosti
            v_min, v_max = -abs(upper - lower) * 0.2, abs(upper - lower) * 0.2
            p.velocity = np.clip(v_new, v_min, v_max)

            # aktualizace pozice
            p.position = np.clip(p.position + p.velocity, lower, upper)

            # vyhodnocení nové pozice
            p.f = function.evaluate(p.position)

            # aktualizace osobního i globálního optima
            if p.f < p.best_f:
                p.best_f = p.f
                p.best_position = p.position.copy()

                if p.f < gBest_value:
                    gBest_value = p.f
                    gBest_position = p.position.copy()

        # uložit celou populaci do historie
        generation_data = [(p.position.copy(), p.f) for p in swarm]
        history.append(generation_data)

    return gBest_position, gBest_value, history
