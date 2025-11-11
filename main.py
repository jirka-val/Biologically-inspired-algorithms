import os
import numpy as np

from algorithms.ant_colony_optimization import ant_colony_optimization
# --- Differential Evolution ---
from algorithms.differential_evolution import differential_evolution
from algorithms.firefly_algorithm import firefly_algorithm
from algorithms.particle_swarm_optimization import particle_swarm_optimization
from algorithms.soma import soma_all_to_one
from core.visualization_tsp import visualize_tsp

# --- testovací funkce (spojité) ---
from functions.ackley import Ackley
from functions.griewank import Griewank
from functions.levy import Levy
from functions.michalewicz import Michalewicz
from functions.rastrigin import Rastrigin
from functions.rosenbrock import Rosenbrock
from functions.schwefel import Schwefel
from functions.sphere import Sphere
from functions.zakharov import Zakharov

# --- vizualizace ---
from core.visualization import visualize_function, visualize_search_gif
from core.visualization_de import visualize_population_evolution

# --- pomocná funkce ---
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":

    # ==========================================================
    # SEZNAM FUNKCÍ PRO TESTOVÁNÍ
    # ==========================================================
    FUNCTIONS = [
        Sphere, Ackley, Schwefel, Rosenbrock,
        Rastrigin, Griewank, Levy, Michalewicz, Zakharov
    ]

    # ==========================================================
    # FIREFLY ALGORITHM (FA)
    # ==========================================================

    for func_class in FUNCTIONS:
        func = func_class(dimension=2)
        print(f"\n=== FIREFLY ALGORITHM on {func.name} ===")

        best_x, best_f, history = firefly_algorithm(
            function=func,
            pop_size=20,
            alpha=0.3,  # Podle zadání [cite: 145]
            beta_0=1.0,  # Podle zadání [cite: 138]
            max_gen=50  # 50 generací stačí na vizualizaci
        )

        print(f"Best solution found: {best_x}")
        print(f"Best fitness: {best_f:.6f}")

        save_dir = os.path.join("results", "firefly_algorithm")
        ensure_dir(save_dir)
        filename = os.path.join(save_dir, f"{func.name}.gif")

        # Použijeme vizualizaci pro spojité funkce (jako u SOMA, PSO...)
        visualize_population_evolution(
            func,
            history,
            filename=filename,
            algorithm_name="Firefly Algorithm (FA)"
        )
        print(f"Animation saved to {filename}")


    # =======================
    # ANT COLONY OPTIMIZATION (ACO) PRO TSP
    # =======================

    """
    # Odkomentuj pro spuštění

    # Podle zadání: 20-40 měst
    num_cities = 25 
    cities = np.random.rand(num_cities, 2) * 200 # Mapa 200x200

    # Podle zadání: počet mravenců založený na počtu měst
    num_ants = num_cities 

    print(f"=== ANT COLONY OPTIMIZATION – TSP ({num_cities} cities, {num_ants} ants) ===")

    best_route, best_distance, history = ant_colony_optimization(
        cities,
        n_ants=num_ants,
        n_iterations=300, 
        alpha=1.0,       # Váha feromonu
        beta=2.0,        # Váha vzdálenosti
        rho=0.5,         # Míra vypařování
        Q=10.0           # Množství pokládaného feromonu
    )

    save_dir = os.path.join("results", "aco_tsp")
    ensure_dir(save_dir)
    filename = os.path.join(save_dir, "tsp_aco.gif")

    # Použijeme stejnou vizualizaci jako pro GA (podle zadání)
    visualize_tsp(history, cities, filename=filename)

    print("Best route found (ACO):", best_route)
    print("Total distance (ACO):", best_distance)
    """

    # ==========================================================
    # SOMA – Self-Organizing Migrating Algorithm (All-to-One)
    # ==========================================================

    """
    FUNCTIONS = [
        Sphere, Ackley, Schwefel, Rosenbrock,
        Rastrigin, Griewank, Levy, Michalewicz, Zakharov
    ]

    for func_class in FUNCTIONS:
        func = func_class(dimension=2)
        print(f"\n=== SOMA All-to-One on {func.name} ===")

        best_x, best_f, history = soma_all_to_one(
            function=func,
            pop_size=20,
            PRT=0.4,
            path_length=3.0,
            step=0.11,
            M_max=100
        )

        print(f"Best solution found: {best_x}")
        print(f"Best fitness: {best_f:.6f}")

        save_dir = os.path.join("results", "soma")
        ensure_dir(save_dir)
        filename = os.path.join(save_dir, f"{func.name}.gif")

        visualize_population_evolution(func, history, filename=filename, algorithm_name="SOMA – All-to-One")

        print(f"Animation saved to {filename}")

    """

    # ==========================================================
    # PARTICLE SWARM OPTIMIZATION (PSO)
    # ==========================================================

    """
    FUNCTIONS = [
        Sphere, Ackley, Schwefel, Rosenbrock,
        Rastrigin, Griewank, Levy, Michalewicz, Zakharov
    ]
    
    for func_class in FUNCTIONS:
        func = func_class(dimension=2)
        print(f"\n=== PARTICLE SWARM OPTIMIZATION on {func.name} ===")

        # spuštění algoritmu
        best_x, best_f, history = particle_swarm_optimization(
            function=func,
            pop_size=15,
            c1=2.0,
            c2=2.0,
            M_max=50
        )

        print(f"Best solution found: {best_x}")
        print(f"Best fitness: {best_f:.6f}")

        save_dir = os.path.join("results", "pso")
        ensure_dir(save_dir)
        filename = os.path.join(save_dir, f"{func.name}.gif")

        visualize_population_evolution(
            func,
            history,
            filename=filename,
            algorithm_name="Particle Swarm Optimization (PSO)"
        )

        print(f"Animation saved to {filename}")
    #"""
    """
    # ==========================================================
    # DIFFERENTIAL EVOLUTION NA VŠECH SPOJITÝCH FUNKCÍCH
    # ==========================================================
    
    FUNCTIONS = [
        Sphere, Ackley, Schwefel, Rosenbrock,
        Rastrigin, Griewank, Levy, Michalewicz, Zakharov
    ]

    for func_class in FUNCTIONS:
        func = func_class(dimension=2)
        print(f"\n=== DIFFERENTIAL EVOLUTION on {func.name} ===")

        # --- spuštění algoritmu ---
        best_x, best_f, history = differential_evolution(
            function=func,
            NP=20,
            F=0.5,
            CR=0.5,
            G=50
        )

        print(f"Best solution found: {best_x}")
        print(f"Best fitness: {best_f:.6f}")

        # --- vizualizace (uloží GIF do results/differential_evolution) ---
        save_dir = os.path.join("results", "differential_evolution")
        ensure_dir(save_dir)
        filename = os.path.join(save_dir, f"{func.name}.gif")

        visualize_population_evolution(func, history, filename=filename)

        print(f"Animation saved to {filename}")
    """

    # =======================
    # GENETICKÝ ALGORITMUS PRO TSP
    # =======================
    """
        num_cities = 20
        cities = np.random.rand(num_cities, 2) * 200
    
        print(f"=== GENETIC ALGORITHM – Travelling Salesman Problem ({num_cities} cities) ===")
    
        best_route, best_distance, history = genetic_tsp(cities, NP=20, G=300)
    
        save_dir = os.path.join("results", "genetic_tsp")
        ensure_dir(save_dir)
        filename = os.path.join(save_dir, "tsp.gif")
    
        visualize_tsp(history, cities, filename=filename)
    
        print("Best route found:", best_route)
        print("Total distance:", best_distance)
    """

    # =======================
    # SPOJITÉ FUNKCE
    # =======================
    """
    iterations = 30  # počet kroků

    FUNCTIONS = [
        Sphere, Ackley, Schwefel, Rosenbrock,
        Rastrigin, Griewank, Levy, Michalewicz, Zakharov
    ]

    ALGORITHMS = {
        "blind_search": blind_search,
        "hill_climbing": hill_climbing,
        "simulated_annealing": simulated_annealing
    }

    for func_class in FUNCTIONS:
        func = func_class()
        # visualize_function(func)

        for algo_name, algo in ALGORITHMS.items():
            print(f"=== {algo_name.upper()} on {func.name} ===")

            best_x, best_f, history = algo(func, iterations=iterations)

            save_dir = os.path.join("results", algo_name)
            ensure_dir(save_dir)
            filename = os.path.join(save_dir, f"{func.name}.gif")

            visualize_search_gif(func, history, filename=filename)

            print(f"{algo_name} on {func.name} → Best solution: {best_x}, value: {best_f}\n")
    """

