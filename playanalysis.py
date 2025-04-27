import numpy as np
import matplotlib.pyplot as plt

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
# 3. Simulation des graphiques
# ----------------------

def simulate_static_graphics():
    np.random.seed(42)
    route_types = ["quick_slant", "curl", "go", "out", "post"]
    colors = {
        "quick_slant": "blue",
        "curl": "green",
        "go": "red",
        "out": "purple",
        "post": "orange"
    }

    all_results = {route: [] for route in route_types}

    for route in route_types:
        for _ in range(300):
            receivers = move_receivers(receivers_init, route)
            distances = []
            for receiver in receivers:
                weights = np.array([np.exp(-np.linalg.norm(receiver - defenders_init[i])**2 / (2 * sigma[i]**2)) for i in range(len(defenders_init))])
                distances_def = np.array([np.linalg.norm(receiver - defenders_init[i]) for i in range(len(defenders_init))])
                if weights.sum() == 0:
                    distances.append(distances_def.mean())
                else:
                    distances.append((weights * distances_def).sum() / weights.sum())
            all_results[route].extend(distances)

    # Histogrammes
    plt.figure(figsize=(12, 7))
    for route in route_types:
        plt.hist(all_results[route], bins=30, density=True, alpha=0.5, label=route.capitalize(), color=colors[route])
    plt.xlabel("Distance moyenne pondérée au défenseur (m)", fontsize=14)
    plt.ylabel("Densité de probabilité", fontsize=14)
    plt.legend(title="Type de route", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Carte couverture défensive
    x = np.linspace(0, 50, 100)
    y = np.linspace(0, 50, 100)
    X, Y = np.meshgrid(x, y)
    Z = coverage_map(X, Y, defenders_init, sigma)

    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z, levels=50, cmap='coolwarm')
    plt.colorbar(label='Intensité de couverture')
    plt.scatter(defenders_init[:,1], defenders_init[:,0], color='black', marker='x', label='Défenseurs')
    plt.xlabel("Largeur du terrain (verges)", fontsize=14)
    plt.ylabel("Profondeur du terrain (verges)", fontsize=14)
    plt.xlim(0, 40)
    plt.ylim(0, 40)
    plt.legend()
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Barres avec moyenne et écart-types
    means = [np.mean(all_results[route]) for route in route_types]
    stds = [np.std(all_results[route]) for route in route_types]

    plt.figure(figsize=(10, 6))
    plt.bar(route_types, means, yerr=stds, capsize=5, color=[colors[route] for route in route_types])
    plt.xlabel("Type de route", fontsize=14)
    plt.ylabel("Distance moyenne pondérée (verges)", fontsize=14)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

# ----------------------
# 4. Exécution principale
# ----------------------

def main():
    simulate_static_graphics()

if __name__ == "__main__":
    main()