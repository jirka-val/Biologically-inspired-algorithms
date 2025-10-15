import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, gridspec
from core.visualization import get_visualization_grid


def visualize_population_evolution(func, history, filename="de_population.gif", algorithm_name="Differential Evolution"):
    """
    Rozšířená vizualizace evolučních algoritmů:
    - vlevo: konturový graf s populací
    - vpravo: graf konvergence (nejlepší hodnota v čase)
    - dole: tabulka nejlepších jedinců
    """
    if func.dimension != 2:
        print("Vizualizace funguje jen pro 2D funkce.")
        return

    grid_points = func.ideal_grid_points()
    X, Y, Z = get_visualization_grid(func, grid_points)

    # --- rozložení figure ---
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1.5], width_ratios=[3, 2])
    gs.update(wspace=0.35, hspace=0.6)

    ax_contour = fig.add_subplot(gs[0, 0])
    ax_convergence = fig.add_subplot(gs[0, 1])
    axtable = fig.add_subplot(gs[1, :])
    axtable.axis("off")

    fig.suptitle(
        f"{algorithm_name} – Optimalizace funkce {func.name}",
        fontsize=16,
        fontweight="bold",
        color="#212121",
        y=0.94
    )

    # --- konturový graf ---
    contour = ax_contour.contourf(X, Y, Z, levels=50, cmap="jet")
    fig.colorbar(contour, ax=ax_contour, shrink=0.8)

    if hasattr(func, "viz_bounds"):
        ax_contour.set_xlim(func.viz_bounds[0], func.viz_bounds[1])
        ax_contour.set_ylim(func.viz_bounds[2], func.viz_bounds[3])
    else:
        ax_contour.set_xlim(func.lower_bound, func.upper_bound)
        ax_contour.set_ylim(func.lower_bound, func.upper_bound)

    ax_contour.set_title(f"{func.name}", fontsize=13, fontweight="bold")
    ax_contour.set_xlabel("x₁")
    ax_contour.set_ylabel("x₂")

    scat = ax_contour.scatter([], [], c="black", s=25, label="Populace")
    best_point, = ax_contour.plot([], [], "yo", markersize=8, label="Nejlepší jedinec")
    ax_contour.legend()

    # --- graf konvergence ---
    ax_convergence.set_title("Nejlepší hodnota v čase", fontsize=13, fontweight="bold")
    ax_convergence.set_xlabel("Generace")
    ax_convergence.set_ylabel("f(x)")
    ax_convergence.set_xlim(1, len(history))  # <-- začínáme od 1
    ax_convergence.set_ylim(0, max(f for gen in history for _, f in gen))
    line_best, = ax_convergence.plot([], [], "b-", label="Best so far", linewidth=2)
    ax_convergence.legend()

    best_so_far = float("inf")
    best_values = []

    # === update funkce ===
    def update(frame):
        nonlocal best_so_far

        gen = history[frame]
        positions = np.array([p for p, _ in gen])
        values = np.array([f for _, f in gen])

        # --- aktualizace populace ---
        scat.set_offsets(positions)
        best_idx = np.argmin(values)
        best_x, best_y = positions[best_idx]
        best_point.set_data([best_x], [best_y])

        # --- tabulka nejlepších ---
        sorted_indices = np.argsort(values)
        sorted_gen = [(positions[i], values[i]) for i in sorted_indices]

        data = []
        for rank, (pos, val) in enumerate(sorted_gen[:15]):
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
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.4)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#212121")
                cell.get_text().set_color("white")
                cell.get_text().set_weight("bold")
            elif row == 1:
                cell.set_facecolor("#FFF176")
            elif row % 2 == 0:
                cell.set_facecolor("#FAFAFA")
            else:
                cell.set_facecolor("#ECEFF1")
            cell.set_edgecolor("#BDBDBD")
            cell.set_linewidth(0.4)

        # --- konvergence ---
        if np.min(values) < best_so_far:
            best_so_far = np.min(values)
        best_values.append(best_so_far)
        # X osa začíná na 1 místo 0
        line_best.set_data(range(1, len(best_values) + 1), best_values)

        # --- aktualizace titulu ---
        ax_contour.set_title(
            f"{func.name} – Generace {frame+1}/{len(history)} | Nejlepší f = {best_so_far:.5f}",
            fontsize=12,
            fontweight="bold"
        )

        return scat, best_point, line_best, table

    ani = animation.FuncAnimation(
        fig, update, frames=len(history), interval=250, blit=False, repeat=False
    )

    ani.save(filename, writer="pillow", fps=4)
    print(f"Animace uložena do {filename}")
    plt.close(fig)
