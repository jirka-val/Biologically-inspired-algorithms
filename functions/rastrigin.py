import numpy as np
from core.base_function import Function


class Rastrigin(Function):
    """
    Rastriginova funkce
    Globální minimum: f(0,...,0) = 0
    Input domain: xi ∈ [-5.12, 5.12]
    """

    def __init__(self, dimension=2, lower_bound=-5.12, upper_bound=5.12):
        super().__init__("Rastrigin", dimension, lower_bound, upper_bound)

        # doporučený rozsah pro vizualizaci
        self.viz_bounds = (-5.12, 5.12, -5.12, 5.12)

    def evaluate(self, x: np.ndarray) -> float:
        d = self.dimension
        return 10 * d + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

    def ideal_grid_points(self, base_density=50):
        return 200
