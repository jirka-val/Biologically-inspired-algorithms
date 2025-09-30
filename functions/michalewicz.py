import numpy as np
from core.base_function import Function


class Michalewicz(Function):
    """
    Michalewiczova funkce
    - má d! lokálních minim
    - parametr m určuje „strmost“ údolí (doporučeno m = 10)
    Globální minimum: není známé obecně pro d > 2
    Input domain: xi ∈ [0, π]
    """

    def __init__(self, dimension=2, lower_bound=0.0, upper_bound=np.pi, m=10):
        super().__init__("Michalewicz", dimension, lower_bound, upper_bound)
        self.m = m

        # doporučený rozsah pro vizualizaci
        self.viz_bounds = (0, np.pi, 0, np.pi)

    def evaluate(self, x: np.ndarray) -> float:
        d = self.dimension
        m = self.m
        sum_val = 0
        for i in range(d):
            xi = x[i]
            sum_val += np.sin(xi) * (np.sin((i + 1) * xi**2 / np.pi) ** (2 * m))
        return -sum_val

    def ideal_grid_points(self, base_density=50):
        return 250
