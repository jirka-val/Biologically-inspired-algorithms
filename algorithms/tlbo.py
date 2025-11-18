import numpy as np
from copy import deepcopy


def tlbo(function, population_size=30, max_generations=50):
    """
    Teaching-Learning Based Optimization (TLBO)

    Args:
        function: Testovací funkce (musí mít .evaluate(), .lower_bound, .upper_bound, .dimension)
        population_size (int): Velikost třídy (NP).
        max_generations (int): Počet generací (respektive iterací).

    Returns:
        best_pos: Nejlepší nalezené řešení.
        best_val: Hodnota f(best_pos).
        history: Historie pro vizualizaci (volitelné).
    """
    dim = function.dimension
    lower = function.lower_bound
    upper = function.upper_bound

    # 1. Inicializace populace (třídy)
    population = np.random.uniform(lower, upper, (population_size, dim))
    fitness = np.array([function.evaluate(ind) for ind in population])

    # Uložení globálního optima
    best_idx = np.argmin(fitness)
    best_pos = population[best_idx].copy()
    best_val = fitness[best_idx]

    history = []  # Pokud byste chtěl vizualizaci, sem se ukládá

    for g in range(max_generations):
        # --- TEACHER PHASE ---
        # Vypočítat průměr třídy (Mean)
        mean_population = np.mean(population, axis=0)

        # Identifikovat učitele (nejlepší řešení)
        teacher_idx = np.argmin(fitness)
        teacher = population[teacher_idx]

        for i in range(population_size):
            # Teaching factor (T_F) je náhodně 1 nebo 2
            tf = np.random.randint(1, 3)
            r = np.random.rand(dim)

            # Rozdíl mezi učitelem a průměrem
            difference_mean = r * (teacher - tf * mean_population)

            # Nové řešení žáka na základě učení od učitele
            new_solution = population[i] + difference_mean
            new_solution = np.clip(new_solution, lower, upper)
            new_fitness = function.evaluate(new_solution)

            # Akceptace (Greedy selection)
            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness

        # --- LEARNER PHASE ---
        for i in range(population_size):
            # Vyber náhodného spolužáka j (různého od i)
            candidates = list(range(population_size))
            candidates.remove(i)
            j = np.random.choice(candidates)

            xi = population[i]
            xj = population[j]
            fi = fitness[i]
            fj = fitness[j]

            r = np.random.rand(dim)

            # Interakce mezi žáky
            if fi < fj:
                new_solution = xi + r * (xi - xj)
            else:
                new_solution = xi + r * (xj - xi)

            new_solution = np.clip(new_solution, lower, upper)
            new_fitness = function.evaluate(new_solution)

            # Akceptace
            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness

        # Aktualizace best known
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_val:
            best_val = fitness[current_best_idx]
            best_pos = population[current_best_idx].copy()

        # (Volitelné) Uložení historie
        # history.append([(ind.copy(), f) for ind, f in zip(population, fitness)])

    return best_pos, best_val, history