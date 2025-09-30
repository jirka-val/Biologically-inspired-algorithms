import numpy as np

class Function:
    """Základní třída pro všechny testovací funkce"""
    def __init__(self, name, dimension=2, lower_bound=-5, upper_bound=5):
        self.name = name
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def evaluate(self, x: np.ndarray) -> float:
        """Potomek musí implementovat vlastní hodnotící funkci"""
        raise NotImplementedError("evaluate() musí být implementováno v potomkovi")

    def ideal_grid_points(self, base_density=50):
        """
        Potomek si to může přepsat, jinak se použije default.
        """
        return base_density
