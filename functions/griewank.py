import numpy as np
from core.base_function import Function


class Griewank(Function):
    """
    Griewankova funkce
    Globální minimum: f(0,...,0) = 0
    Input domain: xi ∈ [-600, 600]
    """

    def __init__(self, dimension=2, lower_bound=-600, upper_bound=600):
        super().__init__("Griewank", dimension, lower_bound, upper_bound)

        # doporučený rozsah pro vizualizaci
        self.viz_bounds = (-10, 10, -10, 10)

    def evaluate(self, x: np.ndarray) -> float:
        d = self.dimension
        sum_val = np.sum(x**2) / 4000
        prod_val = np.prod([np.cos(x[i] / np.sqrt(i + 1)) for i in range(d)])
        return sum_val - prod_val + 1

    def ideal_grid_points(self, base_density=50):
        return 150
