import os

from algorithms.blind_search import blind_search
from algorithms.hill_climbing import hill_climbing
from algorithms.simulated_annealing import simulated_annealing

from functions.ackley import Ackley
from functions.griewank import Griewank
from functions.levy import Levy
from functions.michalewicz import Michalewicz
from functions.rastrigin import Rastrigin
from functions.rosenbrock import Rosenbrock
from functions.schwefel import Schwefel
from functions.sphere import Sphere
from functions.zakharov import Zakharov

from core.visualization import visualize_function, visualize_search_gif

# seznam všech funkcí
FUNCTIONS = [
    Sphere, Ackley, Schwefel, Rosenbrock,
    Rastrigin, Griewank, Levy, Michalewicz, Zakharov
]

# seznam algoritmů (název, funkce)
ALGORITHMS = {
    "blind_search": blind_search,
    "hill_climbing": hill_climbing,
    "simulated_annealing": simulated_annealing
}


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    iterations = 30  # počet kroků

    for func_class in FUNCTIONS:
        func = func_class()

        # vykresli základní 3D graf funkce
        visualize_function(func)

        for algo_name, algo in ALGORITHMS.items():
            print(f"=== {algo_name.upper()} on {func.name} ===")

            # spusť algoritmus
            best_x, best_f, history = algo(func, iterations=iterations)

            # složka pro uložení výsledků
            save_dir = os.path.join("results", algo_name)
            ensure_dir(save_dir)

            filename = os.path.join(save_dir, f"{func.name}.gif")

            # ulož animaci
            visualize_search_gif(func, history, filename=filename)

            print(f"{algo_name} on {func.name} → Best solution: {best_x}, value: {best_f}\n")


#   func = Sphere()
#   func = Ackley()
#   func = Schwefel()
#   func = Rosenbrock()
#   func = Rastrigin()
#   func = Griewank()
#   func = Levy()
#   func = Michalewicz()
#   func = Zakharov()
#   visualize_function(func) #best_x, best_f, history = blind_search(func, iterations=15)
#   best_x, best_f, history = hill_climbing(func, iterations=15)
#   best_x, best_f, history = simulated_annealing(func, iterations=15)
#   visualize_search_gif(func, history, filename="blind_search.gif")
#   print("Best solution found:", best_x, "value:", best_f)