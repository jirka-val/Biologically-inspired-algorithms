import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def get_visualization_grid(func, grid_points):
    """Vrátí X, Y, Z mřížku pro vizualizaci funkce."""
    if hasattr(func, "viz_bounds"):
        x = np.linspace(func.viz_bounds[0], func.viz_bounds[1], grid_points)
        y = np.linspace(func.viz_bounds[2], func.viz_bounds[3], grid_points)
    else:
        x = np.linspace(func.lower_bound, func.upper_bound, grid_points)
        y = np.linspace(func.lower_bound, func.upper_bound, grid_points)

    X, Y = np.meshgrid(x, y)
    Z = np.array([func.evaluate(np.array([xx, yy]))
                  for xx, yy in zip(np.ravel(X), np.ravel(Y))])
    Z = Z.reshape(X.shape)
    return X, Y, Z


def visualize_function(func):
    if func.dimension != 2:
        print("3D vizualizace funguje jen pro 2D funkce!")
        return

    grid_points = func.ideal_grid_points()
    X, Y, Z = get_visualization_grid(func, grid_points)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, Z, cmap="jet", edgecolor="k", linewidth=0.2)
    fig.colorbar(surf, shrink=0.5, aspect=10)

    if hasattr(func, "viz_bounds"):
        ax.set_xlim(func.viz_bounds[0], func.viz_bounds[1])
        ax.set_ylim(func.viz_bounds[2], func.viz_bounds[3])
    else:
        ax.set_xlim(func.lower_bound, func.upper_bound)
        ax.set_ylim(func.lower_bound, func.upper_bound)

    ax.set_title(f"{func.name} Function")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x)")

    #plt.show()


def visualize_search_gif(func, history, filename="search.gif"):
    if func.dimension != 2:
        print("3D vizualizace funguje jen pro 2D funkce!")
        return

    grid_points = func.ideal_grid_points()
    X, Y, Z = get_visualization_grid(func, grid_points)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    ax3d = fig.add_subplot(221, projection="3d")
    ax2d = axs[0, 1]
    axtable = axs[1, 0]
    axplot = axs[1, 1]

    # --- 3D surface ---
    surf = ax3d.plot_surface(X, Y, Z, cmap="jet", edgecolor="k", linewidth=0.3)

    if hasattr(func, "viz_bounds"):
        ax3d.set_xlim(func.viz_bounds[0], func.viz_bounds[1])
        ax3d.set_ylim(func.viz_bounds[2], func.viz_bounds[3])
    else:
        ax3d.set_xlim(func.lower_bound, func.upper_bound)
        ax3d.set_ylim(func.lower_bound, func.upper_bound)

    ax3d.set_zlim(np.min(Z), np.max(Z))

    ax3d.set_title(f"{func.name} – 3D")
    #ax3d.set_xlabel("x1")
    #ax3d.set_ylabel("x2")
    #ax3d.set_zlabel("f(x)")
    point3d, = ax3d.plot([], [], [], "co", markersize=15)  # cyan bod

    # --- 2D contour ---
    contour = ax2d.contourf(X, Y, Z, levels=50, cmap="jet")
    fig.colorbar(contour, ax=ax2d, shrink=0.8)

    if hasattr(func, "viz_bounds"):
        ax2d.set_xlim(func.viz_bounds[0], func.viz_bounds[1])
        ax2d.set_ylim(func.viz_bounds[2], func.viz_bounds[3])
    else:
        ax2d.set_xlim(func.lower_bound, func.upper_bound)
        ax2d.set_ylim(func.lower_bound, func.upper_bound)

    ax2d.set_title(f"{func.name} – 2D Contour")
    trajectory, = ax2d.plot([], [], "c-", alpha=0.8)  # cyan čára
    point2d, = ax2d.plot([], [], "co", markersize=6)  # cyan bod

    # --- tabulka ---
    axtable.axis("off")
    table_data = [["Krok", ""],
                  ["Pozice", ""],
                  ["Hodnota", ""],
                  ["Nejlepší", ""]]
    table = axtable.table(cellText=table_data,
                          colWidths=[0.3, 0.6],
                          loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # --- graf konvergence ---
    axplot.set_xlim(0, len(history))
    all_f = [h[1] for h in history]
    axplot.set_ylim(0, max(all_f))
    axplot.set_title("Nejlepší hodnota v čase")
    axplot.set_xlabel("Krok")
    axplot.set_ylabel("f(x)")
    line_best, = axplot.plot([], [], "b-", label="Best so far")
    axplot.legend()

    best_so_far = float("inf")
    best_pos = None
    best_values = []

    # === update funkce ===
    def update(i):
        nonlocal best_so_far, best_pos

        x_val = history[i][0][0]
        y_val = history[i][0][1]
        z_val = history[i][1]

        # 3D bod
        point3d.set_data([x_val], [y_val])
        point3d.set_3d_properties([z_val])

        # 2D bod + trajektorie
        xs = [h[0][0] for h in history[:i+1]]
        ys = [h[0][1] for h in history[:i+1]]
        trajectory.set_data(xs, ys)
        point2d.set_data([x_val], [y_val])

        # nejlepší hodnota zatím
        if z_val < best_so_far:
            best_so_far = z_val
            best_pos = (x_val, y_val)
        best_values.append(best_so_far)

        # update grafu konvergence
        line_best.set_data(range(1, len(best_values)+1), best_values)

        # update tabulky
        table._cells[(0, 1)].get_text().set_text(f"{i+1}/{len(history)}")
        table._cells[(1, 1)].get_text().set_text(f"({x_val:.2f}, {y_val:.2f})")
        table._cells[(2, 1)].get_text().set_text(f"{z_val:.4f}")
        table._cells[(3, 1)].get_text().set_text(
            f"{best_so_far:.4f} @ ({best_pos[0]:.2f}, {best_pos[1]:.2f})"
        )

        return point3d, trajectory, point2d, line_best, table

    ani = animation.FuncAnimation(
        fig, update, frames=len(history), interval=200, blit=False, repeat=False
    )

    ani.save(filename, writer="pillow", fps=3)
    print(f"Animace uložena do {filename}")
    plt.close(fig)
