import os
import numpy as np

# --- spojité algoritmy ---
from algorithms.blind_search import blind_search
from algorithms.hill_climbing import hill_climbing
from algorithms.simulated_annealing import simulated_annealing

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
from core.visualization_tsp import visualize_tsp

# --- genetický algoritmus pro TSP ---
from algorithms.genetic_tsp import genetic_tsp


# --- pomocná funkce ---
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":

    # =======================
    # GENETICKÝ ALGORITMUS PRO TSP
    # =======================

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

