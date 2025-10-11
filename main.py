import os
import numpy as np

# --- Differential Evolution ---
from algorithms.differential_evolution import differential_evolution
from algorithms.particle_swarm_optimization import particle_swarm_optimization

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
    # PARTICLE SWARM OPTIMIZATION (PSO)
    # ==========================================================

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
            pop_size=15,  # počet částic
            c1=2.0,  # kognitivní složka
            c2=2.0,  # sociální složka
            w=0.7,  # setrvačnost
            M_max=50  # počet iterací
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

    # ==========================================================
    # DIFFERENTIAL EVOLUTION NA VŠECH SPOJITÝCH FUNKCÍCH
    # ==========================================================
    """
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
            G=10
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

