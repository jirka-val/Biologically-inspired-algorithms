import numpy as np
from core.base_function import Function


class Schwefel(Function):
    """
    Schwefelova funkce
    Globální minimum: f(420.9687,...,420.9687) = 0
    Input domain: xi ∈ [-500, 500]
    """

    def __init__(self, dimension=2, lower_bound=-500, upper_bound=500):
        super().__init__("Schwefel", dimension, lower_bound, upper_bound)

        # doporučený rozsah pro vizualizaci
        self.viz_bounds = (-500, 500, -500, 500)

    def evaluate(self, x: np.ndarray) -> float:
        d = self.dimension
        sum_val = np.sum(x * np.sin(np.sqrt(np.abs(x))))
        return 418.9829 * d - sum_val

    def ideal_grid_points(self, base_density=50):
        return 150
