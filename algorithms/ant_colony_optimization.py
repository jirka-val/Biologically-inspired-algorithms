import numpy as np


def calculate_tour_distance(route, dist_matrix):
    """Pomocná funkce pro výpočet celkové délky cesty."""
    distance = 0.0
    for i in range(len(route)):
        city_a = route[i]
        city_b = route[(i + 1) % len(route)]  # % pro uzavření cesty
        distance += dist_matrix[city_a, city_b]
    return distance


def ant_colony_optimization(cities, n_ants=20, n_iterations=200,
                            alpha=1.0, beta=2.0, rho=0.5, Q=1.0):
    """
    Implementace Ant Colony Optimization (ACO) pro problém TSP.

    Args:
        cities (np.ndarray): Matice (n_cities, 2) se souřadnicemi měst.
        n_ants (int): Počet mravenců v kolonii. Dle zadání by měl být
                      založen na počtu měst (např. n_ants = n_cities).
        n_iterations (int): Počet generací/iterací algoritmu.
        alpha (float): Váha feromonu (α). V příkladu = 1.0.
        beta (float): Váha viditelnosti (η). V příkladu = 2.0.
        rho (float): Míra vypařování feromonu (ρ). V příkladu = 0.5.
        Q (float): Konstanta pro pokládání feromonu. V příkladu = 1.0.

    Returns:
        tuple: (best_route, best_distance, history)
               best_route: Seznam ID měst v nejlepší nalezené trase.
               best_distance: Délka nejlepší nalezené trasy.
               history: Seznam (best_route, best_distance) pro každou generaci.
    """
    n_cities = len(cities)

    # --- 1. Inicializace ---

    # Výpočet matice vzdáleností (d)
    dist_matrix = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(i + 1, n_cities):
            dist = np.linalg.norm(cities[i] - cities[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    # Výpočet matice viditelnosti (η = 1/d)
    # Přidáme malou epsilon, abychom se vyhnuli dělení nulou (pro i == j)
    eta = 1.0 / (dist_matrix + 1e-10)
    np.fill_diagonal(eta, 0)  # Mravenec nemůže přejít do stejného města

    # Inicializace matice feromonů (τ)
    tau = np.ones((n_cities, n_cities))

    # Proměnné pro sledování nejlepší cesty
    best_global_route = None
    best_global_distance = float('inf')
    history = []

    # --- 2. Hlavní cyklus algoritmu ---
    for iteration in range(n_iterations):

        all_ant_routes = []
        all_ant_distances = []

        # Pro každého mravence...
        for ant in range(n_ants):

            # Každý mravenec začíná v jiném městě.
            current_city = ant % n_cities
            route = [current_city]
            visited = {current_city}

            # Konstrukce cesty (navštívení n_cities měst)
            while len(route) < n_cities:
                unvisited_cities = [city for city in range(n_cities) if city not in visited]

                # --- Výpočet pravděpodobností přechodu (podle PDF) ---

                # τ(r,s)^α * η(r,s)^β
                numerators = np.zeros(len(unvisited_cities))
                for i, next_city in enumerate(unvisited_cities):
                    tau_rs = tau[current_city, next_city]
                    eta_rs = eta[current_city, next_city]
                    numerators[i] = (tau_rs ** alpha) * (eta_rs ** beta)

                # Σ [τ(r,u)^α * η(r,u)^β]
                denominator = np.sum(numerators)

                if denominator == 0:
                    probs = np.ones(len(unvisited_cities)) / len(unvisited_cities)
                else:
                    probs = numerators / denominator

                # --- Výběr dalšího města ---
                next_city = np.random.choice(unvisited_cities, p=probs)

                route.append(next_city)
                visited.add(next_city)
                current_city = next_city

            # Cesta je hotová, uložíme ji
            all_ant_routes.append(route)
            distance = calculate_tour_distance(route, dist_matrix)
            all_ant_distances.append(distance)

            # Aktualizace globálního optima
            if distance < best_global_distance:
                best_global_distance = distance
                best_global_route = route

        # Uložení nejlepšího výsledku této iterace pro vizualizaci
        history.append((best_global_route.copy(), best_global_distance))

        # --- 3. Aktualizace feromonů (na konci "migration loop") ---

        # 3.1 Vypařování (Evaporation)
        tau = (1.0 - rho) * tau

        # 3.2 Pokládání (Deposition)
        for route, distance in zip(all_ant_routes, all_ant_distances):
            delta_tau = Q / distance
            for i in range(n_cities):
                city_a = route[i]
                city_b = route[(i + 1) % n_cities]

                tau[city_a, city_b] += delta_tau
                tau[city_b, city_a] += delta_tau

    return best_global_route, best_global_distance, history