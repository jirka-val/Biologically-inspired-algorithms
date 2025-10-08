import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def visualize_tsp(history, cities, filename="tsp.gif"):
    """
    Vizualizace průběhu hledání řešení TSP pomocí genetického algoritmu.
    - Zobrazuje města a aktuální nejlepší cestu
    - Druhý graf ukazuje vývoj nejlepší vzdálenosti v čase

    Args:
        history: seznam [(route, distance)] z běhu GA
        cities: np.ndarray tvaru (n, 2) – souřadnice měst
        filename: kam uložit výsledný .gif
    """
    cities = np.array(cities)
    fig, (ax_path, ax_plot) = plt.subplots(1, 2, figsize=(12, 6))

    # --- Nastavení grafu cesty ---
    ax_path.set_title("Travelling Salesman Problem – Best Route")
    ax_path.scatter(cities[:, 0], cities[:, 1], color="red", s=40)
    for i, (x, y) in enumerate(cities):
        ax_path.text(x + 2, y + 2, str(i), fontsize=8, color="black")

    line, = ax_path.plot([], [], "b-", lw=2, label="Best route")
    ax_path.set_xlim(0, np.max(cities[:, 0]) + 20)
    ax_path.set_ylim(0, np.max(cities[:, 1]) + 20)
    ax_path.legend()

    # --- Nastavení grafu konvergence ---
    ax_plot.set_title("Best Distance over Generations")
    ax_plot.set_xlabel("Generation")
    ax_plot.set_ylabel("Distance")
    line_best, = ax_plot.plot([], [], "g-", lw=2, label="Best distance")
    ax_plot.legend()

    best_values = []

    # === update funkce ===
    def update(i):
        route, dist = history[i]
        route_coords = cities[route + [route[0]]]

        # aktualizace cesty
        line.set_data(route_coords[:, 0], route_coords[:, 1])

        # aktualizace konvergence
        best_values.append(dist)
        line_best.set_data(range(len(best_values)), best_values)
        ax_plot.set_xlim(0, len(history))
        ax_plot.set_ylim(0, max(best_values) * 1.1)

        return line, line_best

    ani = animation.FuncAnimation(
        fig, update, frames=len(history), interval=200, blit=False, repeat=False
    )

    ani.save(filename, writer="pillow", fps=15)
    plt.close(fig)
    print(f"TSP animace uložena do {filename}")
