import numpy as np
from core.base_function import Function


class Levy(Function):
    """
    Levyho funkce
    Globální minimum: f(1,...,1) = 0
    Input domain: xi ∈ [-10, 10]
    """

    def __init__(self, dimension=2, lower_bound=-10, upper_bound=10):
        super().__init__("Levy", dimension, lower_bound, upper_bound)

        # doporučený rozsah pro vizualizaci
        self.viz_bounds = (-10, 10, -10, 10)

    def evaluate(self, x: np.ndarray) -> float:
        d = self.dimension

        w = 1 + (x - 1) / 4
        term1 = np.sin(np.pi * w[0])**2
        term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)

        sum_val = 0
        for i in range(d - 1):
            wi = w[i]
            sum_val += (wi - 1)**2 * (1 + 10 * np.sin(np.pi * wi + 1)**2)

        return term1 + sum_val + term3

    def ideal_grid_points(self, base_density=50):
        return 300
