import numpy as np
from core.base_function import Function

class Sphere(Function):
    """Sphere test function"""
    def __init__(self, dimension=2, lower_bound=-5, upper_bound=5):
        super().__init__("Sphere", dimension, lower_bound, upper_bound)

    def evaluate(self, x: np.ndarray) -> float:
        return np.sum(x**2)

    def ideal_grid_points(self, base_density=50):
        return 80
