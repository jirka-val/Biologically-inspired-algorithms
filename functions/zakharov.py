import numpy as np
from core.base_function import Function


class Zakharov(Function):
    """
    Zakharovova funkce
    Globální minimum: f(0,...,0) = 0
    Input domain: xi ∈ [-5, 10]
    """

    def __init__(self, dimension=2, lower_bound=-5, upper_bound=10):
        super().__init__("Zakharov", dimension, lower_bound, upper_bound)

        # doporučený rozsah pro vizualizaci
        self.viz_bounds = (-10, 10, -10, 10)

    def evaluate(self, x: np.ndarray) -> float:
        d = self.dimension
        sum1 = np.sum(x**2)
        sum2 = np.sum(0.5 * (np.arange(1, d + 1)) * x)
        return sum1 + sum2**2 + sum2**4

    def ideal_grid_points(self, base_density=50):
        return 100
