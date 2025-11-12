import numpy as np
from copy import deepcopy


class Firefly:
    """Reprezentace jedné světlušky."""

    def __init__(self, dimension, lower_bound, upper_bound):
        self.dimension = dimension
        self.position = np.random.uniform(lower_bound, upper_bound, dimension)
        self.fitness = np.inf  # Fitness = Intenzita světla (nižší je lepší)


def firefly_algorithm(function, pop_size=20, alpha=0.3, beta_0=1.0, max_gen=50):
    """
    Implementace Firefly Algorithm (FA).

    Args:
        function: instance třídy Function (např. Ackley)
        pop_size (int): Počet světlušek v populaci.
        alpha (float): Parametr náhodného pohybu[cite: 145].
        beta_0 (float): Základní atraktivita při r=0[cite: 138].
        max_gen (int): Maximální počet generací.

    Returns:
        tuple: (best_pos, best_fit, history)
               best_pos: Pozice nejlepší nalezené světlušky.
               best_fit: Fitness nejlepší nalezené světlušky.
               history: Seznam populací v každé generaci pro vizualizaci.
    """
    dim = function.dimension
    lower = function.lower_bound
    upper = function.upper_bound

    # --- 1. Inicializace ---
    population = [Firefly(dim, lower, upper) for _ in range(pop_size)]
    for ff in population:
        ff.fitness = function.evaluate(ff.position)

    # Najdeme nejlepší
    gBest = min(population, key=lambda f: f.fitness)
    best_pos = gBest.position.copy()
    best_fit = gBest.fitness

    history = [[(ff.position.copy(), ff.fitness) for ff in population]]

    # --- 2. Hlavní cyklus algoritmu ---
    for gen in range(max_gen - 1):
        new_population = deepcopy(population)

        # Projdeme každou světlušku 'i'
        for i in range(pop_size):
            moved = False
            # Porovnáme ji s každou světluškou 'j'
            for j in range(pop_size):

                if population[j].fitness < population[i].fitness:
                    r = np.linalg.norm(population[i].position - population[j].position)

                    # Vypočítáme atraktivitu
                    beta = beta_0 / (1.0 + r)
                    attraction = beta * (population[j].position - population[i].position)

                    # Složka náhodného pohybu
                    random_step = alpha * (np.random.normal(0, 1, size=dim))

                    # Vypočet nové pozice
                    new_pos = population[i].position + attraction + random_step
                    new_pos = np.clip(new_pos, lower, upper)

                    # Uložíme do nové populace
                    new_population[i].position = new_pos
                    new_population[i].fitness = function.evaluate(new_pos)
                    moved = True

                    break

            if not moved:
                random_step = alpha * (np.random.normal(0, 1, size=dim))
                new_pos = population[i].position + random_step
                new_pos = np.clip(new_pos, lower, upper)

                new_population[i].position = new_pos
                new_population[i].fitness = function.evaluate(new_pos)

        # Aktualizujeme populaci
        population = new_population

        # Najdeme a uložíme nejlepší
        current_best_ff = min(population, key=lambda f: f.fitness)
        if current_best_ff.fitness < best_fit:
            best_fit = current_best_ff.fitness
            best_pos = current_best_ff.position.copy()

        # Uložíme historii pro vizualizaci
        history.append([(ff.position.copy(), ff.fitness) for ff in population])

    return best_pos, best_fit, history