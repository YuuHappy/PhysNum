import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constantes
N_ITERATIONS = 250
FIELD_WIDTH = 50  # verges
FIELD_HEIGHT = 30  # verges
DT = 0.1  # secondes
SIM_TIME = 5  # secondes
SIGMA_DEFENSEUR = 5  # verges

# Définir les tracés standards
def generate_route(route_type, position, vitesse, temps_total, dt):
    t = np.arange(0, temps_total, dt)
    if route_type == "slant":
        return np.array([position + vitesse * np.array([np.cos(np.pi/6), np.sin(np.pi/6)]) * ti for ti in t])
    elif route_type == "go":
        return np.array([position + vitesse * np.array([1, 0]) * ti for ti in t])
    elif route_type == "out":
        return np.array([position + vitesse * np.array([np.cos(np.pi/3), -np.sin(np.pi/3)]) * ti for ti in t])
    elif route_type == "post":
        return np.array([position + vitesse * np.array([np.cos(np.pi/6), -np.sin(np.pi/6)]) * ti for ti in t])
    elif route_type == "curl":
        return np.array([position + vitesse * np.array([np.cos(np.pi/2), 0]) * (ti if ti < temps_total/2 else temps_total/2) for ti in t])
    else:
        return np.array([position + vitesse * np.array([1, 0]) * ti for ti in t])

# Défenseur représenté par fonction d'onde gaussienne
class Defender:
    def __init__(self, position, sigma=SIGMA_DEFENSEUR):
        self.position = np.array(position)
        self.sigma = sigma

    def influence(self, r):
        return np.exp(-np.linalg.norm(r - self.position)**2 / (2 * self.sigma**2))

# Receiver
class Receiver:
    def __init__(self, route_type, start_position, vitesse=7):
        self.route_type = route_type
        self.start_position = np.array(start_position)
        self.vitesse = vitesse

    def simulate(self, temps_total, dt):
        base_route = generate_route(self.route_type, self.start_position, self.vitesse, temps_total, dt)
        bruit = np.random.normal(0, 0.5, base_route.shape)  # 0.5 verge de bruit par pas
        return base_route + bruit

# Distance pondérée
def distance_ponderee(receivers_pos, defenders):
    pond_dist = []
    for r in receivers_pos:
        numerateur = 0
        denominateur = 0
        for d in defenders:
            influence = d.influence(r)
            distance = np.linalg.norm(r - d.position)
            numerateur += influence * distance
            denominateur += influence
        pond_dist.append(numerateur / denominateur if denominateur != 0 else np.nan)
    return np.nanmean(pond_dist)

# Simulation Monte Carlo
def run_simulation(n_iterations=250):
    results = []
    tracés = ["slant", "go", "out", "post", "curl"]
    
    for offense1 in tracés:
        for offense2 in tracés:
            distances = []
            for _ in range(n_iterations):
                defenders = [Defender([15, 10]), Defender([35, 20])]
                receiver1 = Receiver(offense1, [0, 10])
                receiver2 = Receiver(offense2, [0, 20])
                traj1 = receiver1.simulate(SIM_TIME, DT)
                traj2 = receiver2.simulate(SIM_TIME, DT)
                traj = np.vstack([traj1, traj2])
                dist = distance_ponderee(traj, defenders)
                distances.append(dist)
            results.append({
                "Receveur 1": offense1,
                "Receveur 2": offense2,
                "Moyenne distance pondérée": np.nanmean(distances),
                "Écart-type": np.nanstd(distances)
            })
    return pd.DataFrame(results)

# Lancer la simulation
df_results = run_simulation()

# Heatmap affichage
pivot_mean = df_results.pivot(index="Receveur 1", columns="Receveur 2", values="Moyenne distance pondérée")
pivot_std = df_results.pivot(index="Receveur 1", columns="Receveur 2", values="Écart-type")

fig, ax = plt.subplots(figsize=(12, 10))
cax = ax.matshow(pivot_mean, cmap="Blues")
plt.colorbar(cax, label="Distance pondérée moyenne (verges)")
ax.set_xticks(np.arange(len(pivot_mean.columns)))
ax.set_yticks(np.arange(len(pivot_mean.index)))
ax.set_xticklabels(pivot_mean.columns, rotation=45, ha="left")
ax.set_yticklabels(pivot_mean.index)
for (i, j), val in np.ndenumerate(pivot_mean.values):
    ax.text(j, i, f"{val:.1f}", ha='center', va='center', color='black', fontsize=8)
plt.title("Distance pondérée moyenne selon les tracés")
plt.xlabel("Tracé Receveur 2")
plt.ylabel("Tracé Receveur 1")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(12, 10))
cax = ax.matshow(pivot_std, cmap="Greens")
plt.colorbar(cax, label="Écart-type (verges)")
ax.set_xticks(np.arange(len(pivot_std.columns)))
ax.set_yticks(np.arange(len(pivot_std.index)))
ax.set_xticklabels(pivot_std.columns, rotation=45, ha="left")
ax.set_yticklabels(pivot_std.index)
for (i, j), val in np.ndenumerate(pivot_std.values):
    ax.text(j, i, f"{val:.1f}", ha='center', va='center', color='black', fontsize=8)
plt.title("Écart-type de la distance pondérée selon les tracés")
plt.xlabel("Tracé Receveur 2")
plt.ylabel("Tracé Receveur 1")
plt.tight_layout()
plt.show()
