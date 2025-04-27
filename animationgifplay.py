import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# ----------------------
# 1. Paramètres initiaux
# ----------------------

receivers_init = np.array([
    [5, 5],
    [5, 20],
    [5, 12],
    [5, 30],
    [5, 18],
])

defenders_init = np.array([
    [8, 5],
    [8, 30],
    [10, 12],
    [10, 18],
    [20, 10],
    [20, 25],
    [20, 18],
])

sigma = np.array([3, 3, 3, 3, 5, 5, 5])

# ----------------------
# 2. Fonctions auxiliaires
# ----------------------

def move_receivers(receivers_init, route_type="quick_slant", t=1.0):
    receivers = receivers_init.copy()
    if route_type == "quick_slant":
        for r in receivers:
            r[0] += (5 / np.sqrt(2)) * t
            r[1] += (5 / np.sqrt(2)) * t
    elif route_type == "curl":
        for r in receivers:
            r[0] += 8 * t
    elif route_type == "go":
        for r in receivers:
            r[0] += 15 * t
    elif route_type == "out":
        for r in receivers:
            r[0] += 8 * t
            r[1] += 5 * t
    elif route_type == "post":
        for r in receivers:
            r[0] += 12 * t
            r[1] += 7 * t
    return receivers

def coverage_map(X, Y, defenders, sigmas):
    Z = np.zeros_like(X)
    for (dx, dy), s in zip(defenders, sigmas):
        Z += np.exp(-((X - dy)**2 + (Y - dx)**2) / (2 * s**2))
    return Z

# ----------------------
# 3. Génération d'une animation
# ----------------------

def animate_route(route_type, save_folder="animations"):
    fig, ax = plt.subplots(figsize=(10, 8))
    x = np.linspace(0, 50, 100)
    y = np.linspace(0, 50, 100)
    X, Y = np.meshgrid(x, y)
    Z = coverage_map(X, Y, defenders_init, sigma)

    contour = ax.contourf(X, Y, Z, levels=50, cmap='coolwarm', alpha=0.6)
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Intensité de couverture')

    defenders_plot = ax.scatter(defenders_init[:,1], defenders_init[:,0], color='black', marker='x', label='Défenseurs')
    receivers_plot = ax.scatter([], [], color='yellow', label='Receveurs')

    ax.set_title(f"Animation - {route_type.replace('_', ' ').capitalize()}", fontsize=16)
    ax.set_xlabel("Largeur du terrain (m)", fontsize=14)
    ax.set_ylabel("Profondeur du terrain (m)", fontsize=14)
    ax.set_xlim(0, 50)
    ax.set_ylim(50, 0)
    ax.grid(True)
    ax.legend()

    def update(frame):
        t = frame / 20
        receivers_pos = move_receivers(receivers_init, route_type, t)
        receivers_plot.set_offsets(np.c_[receivers_pos[:,1], receivers_pos[:,0]])
        return receivers_plot,

    ani = animation.FuncAnimation(fig, update, frames=21, interval=200, blit=True)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    filepath = os.path.join(save_folder, f"animation_{route_type}.gif")
    ani.save(filepath, writer='pillow')
    print(f"GIF sauvegardé sous {filepath}")
    plt.close()

# ----------------------
# 4. Exécution principale
# ----------------------

def main():
    routes = ["quick_slant", "curl", "go", "out", "post"]
    for route in routes:
        animate_route(route)

if __name__ == "__main__":
    main()