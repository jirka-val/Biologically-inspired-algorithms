import numpy as np
from core.base_function import Function


class Rosenbrock(Function):
    """
    Rosenbrockova funkce (Banánová funkce)
    Globální minimum: f(1,...,1) = 0
    Input domain: xi ∈ [-5, 10] (často se také používá [-2.048, 2.048])
    """

    def __init__(self, dimension=2, lower_bound=-5, upper_bound=10):
        super().__init__("Rosenbrock", dimension, lower_bound, upper_bound)

        # doporučený rozsah pro vizualizaci
        self.viz_bounds = (-10, 10, -6, 6)

    def evaluate(self, x: np.ndarray) -> float:
        d = self.dimension
        sum_val = 0
        for i in range(d - 1):
            xi = x[i]
            xnext = x[i + 1]
            sum_val += 100 * (xnext - xi**2) ** 2 + (xi - 1) ** 2
        return sum_val

    def ideal_grid_points(self, base_density=50):
        return 200
