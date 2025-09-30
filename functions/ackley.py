import numpy as np
from core.base_function import Function


class Ackley(Function):
    """
    Ackleyho funkce
    Globální minimum: f(0,...,0) = 0
    """

    def __init__(self, dimension=2, lower_bound=-32.768, upper_bound=32.768,
                 a=20, b=0.2, c=2*np.pi):
        super().__init__("Ackley", dimension, lower_bound, upper_bound)
        self.a = a
        self.b = b
        self.c = c

    def evaluate(self, x: np.ndarray) -> float:
        d = self.dimension
        sum1 = np.sum(x ** 2)
        sum2 = np.sum(np.cos(self.c * x))

        term1 = -self.a * np.exp(-self.b * np.sqrt(sum1 / d))
        term2 = -np.exp(sum2 / d)

        return term1 + term2 + self.a + np.exp(1)

    def ideal_grid_points(self, base_density=50):
        return 30
