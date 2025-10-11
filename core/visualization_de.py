import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, gridspec
from core.visualization import get_visualization_grid  # import základní funkce


def visualize_population_evolution(func, history, filename="de_population.gif"):
    """
    Vylepšená vizualizace pro Differential Evolution – zobrazuje celou populaci + tabulku hodnot.
    Nejlepší jedinec je v tabulce zvýrazněn.
    """
    if func.dimension != 2:
        print("Vizualizace funguje jen pro 2D funkce.")
        return

    grid_points = func.ideal_grid_points()
    X, Y, Z = get_visualization_grid(func, grid_points)

    fig = plt.figure(figsize=(13, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax = fig.add_subplot(gs[0])
    axtable = fig.add_subplot(gs[1])
    axtable.axis("off")

    fig.suptitle(
        "Differential Evolution ",
        fontsize=16,
        fontweight="bold",
        color="#212121",
        y=0.97
    )

    # konturový graf
    contour = ax.contourf(X, Y, Z, levels=50, cmap="jet")
    fig.colorbar(contour, ax=ax, shrink=0.8)

    if hasattr(func, "viz_bounds"):
        ax.set_xlim(func.viz_bounds[0], func.viz_bounds[1])
        ax.set_ylim(func.viz_bounds[2], func.viz_bounds[3])
    else:
        ax.set_xlim(func.lower_bound, func.upper_bound)
        ax.set_ylim(func.lower_bound, func.upper_bound)

    ax.set_title(f"{func.name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")

    # inicializace bodů
    scat = ax.scatter([], [], c="black", s=25, label="Populace")
    best_point, = ax.plot([], [], "yo", markersize=8, label="Nejlepší jedinec")
    ax.legend()

    # update funkce
    def update(frame):
        gen = history[frame]
        positions = np.array([p for p, _ in gen])
        values = np.array([f for _, f in gen])

        scat.set_offsets(positions)

        best_idx = np.argmin(values)
        best_x, best_y = positions[best_idx]
        best_point.set_data([best_x], [best_y])

        sorted_indices = np.argsort(values)
        sorted_gen = [(positions[i], values[i]) for i in sorted_indices]

        data = []
        for rank, (pos, val) in enumerate(sorted_gen[:20]):
            data.append([
                f"{rank+1}",
                f"{pos[0]:.3f}",
                f"{pos[1]:.3f}",
                f"{val:.5f}"
            ])

        axtable.clear()
        axtable.axis("off")
        table = axtable.table(
            cellText=data,
            colLabels=["#", "x₁", "x₂", "f(x)"],
            loc="center",
            colWidths=[0.1, 0.25, 0.25, 0.4],
            cellLoc="center"
        )

        # stylování tabulky
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        for key, cell in table.get_celld().items():
            row, col = key
            if row == 0:
                cell.set_facecolor("#212121")
                cell.get_text().set_color("white")
                cell.get_text().set_weight("bold")
            else:
                if row == 1:
                    cell.set_facecolor("#FFF176")
                elif row % 2 == 0:
                    cell.set_facecolor("#FAFAFA")
                else:
                    cell.set_facecolor("#ECEFF1")
            cell.set_edgecolor("#BDBDBD")
            cell.set_linewidth(0.4)

        # aktualizace titulku uvnitř grafu
        ax.set_title(
            f"{func.name} – Generace {frame+1}/{len(history)} | Nejlepší f = {values[best_idx]:.5f}",
            fontsize=12,
            fontweight="bold"
        )

        return scat, best_point, table

    ani = animation.FuncAnimation(
        fig, update, frames=len(history), interval=200, blit=False, repeat=False
    )

    ani.save(filename, writer="pillow", fps=4)
    print(f"Animace uložena do {filename}")
    plt.close(fig)
