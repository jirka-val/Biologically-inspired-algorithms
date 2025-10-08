import numpy as np


def calculate_distance(route, cities):
    dist = 0.0
    for i in range(len(route)):
        city_a = cities[route[i]]
        city_b = cities[route[(i + 1) % len(route)]]
        dist += np.linalg.norm(city_a - city_b)
    return dist


def ordered_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(np.random.choice(range(size), 2, replace=False))
    child = [-1] * size
    child[start:end] = parent1[start:end]

    fill_values = [c for c in parent2 if c not in child]
    j = 0
    for i in range(size):
        if child[i] == -1:
            child[i] = fill_values[j]
            j += 1
    return child


def mutate(route):
    if np.random.uniform() < 0.5:
        i, j = np.random.choice(len(route), 2, replace=False)
        route[i], route[j] = route[j], route[i]
    return route


def genetic_tsp(cities, NP=20, G=200):
    """
    Genetický algoritmus pro TSP.
    """
    D = len(cities)
    population = [np.random.permutation(D).tolist() for _ in range(NP)]
    distances = np.array([calculate_distance(ind, cities) for ind in population])

    history = []
    best_idx = np.argmin(distances)
    best_route = population[best_idx]
    best_distance = distances[best_idx]
    history.append((best_route.copy(), best_distance))

    for _ in range(G):
        new_population = population.copy()

        for j in range(NP):
            parent_A = population[j]
            idx_B = np.random.choice([k for k in range(NP) if k != j])
            parent_B = population[idx_B]

            offspring_AB = ordered_crossover(parent_A, parent_B)
            offspring_AB = mutate(offspring_AB)

            f_offspring = calculate_distance(offspring_AB, cities)
            f_parentA = calculate_distance(parent_A, cities)

            if f_offspring < f_parentA:
                new_population[j] = offspring_AB

        population = new_population
        distances = np.array([calculate_distance(ind, cities) for ind in population])

        best_idx = np.argmin(distances)
        best_route = population[best_idx]
        best_distance = distances[best_idx]

        # uložíme nejlepší
        history.append((best_route.copy(), best_distance))

    return best_route, best_distance, history
