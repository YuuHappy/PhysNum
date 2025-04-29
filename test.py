import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Paramètres généraux
FIELD_LENGTH = 100  # yards
FIELD_WIDTH = 53.3  # yards
N_SIMULATIONS = 500
SIGMA_DEFENSE = 5  # largeur de la gaussienne du défenseur

# Types d'attaques (tracés)
offense_types = [
    "Slant", "Go", "Curl", "Post", "Out", "Screen Pass", "Draw", "Play Action", "RPO"
]

defense_types = [
    "Blitz", "Cover 0", "Cover 1", "Cover 2", "Cover 3", "Cover 4", "Man-to-Man"
]

# Fonctions de mouvement de tracés
def route_slant(t):
    return 5 + 8*t, 25 + 8*t

def route_go(t):
    return 5 + 20*t, 25

def route_curl(t):
    if t < 0.5:
        return 5 + 10*t, 25
    else:
        return 10 - 10*(t-0.5), 25

def route_post(t):
    return 5 + 15*t, 25 + 5*t

def route_out(t):
    return 5 + 10*t, 25 + 15*t

routes = {
    "Slant": route_slant,
    "Go": route_go,
    "Curl": route_curl,
    "Post": route_post,
    "Out": route_out,
    "Screen Pass": lambda t: (5 + 2*t, 25),
    "Draw": lambda t: (5, 25-2*t),
    "Play Action": route_post,
    "RPO": route_slant
}

# Densité de probabilité des défenseurs
def defense_density(x, y, defenders_pos):
    rho = 0
    for (xd, yd) in defenders_pos:
        rho += np.exp(-((x - xd)**2 + (y - yd)**2) / (2 * SIGMA_DEFENSE**2))
    return rho

# Génération des défenseurs selon type de couverture
def generate_defenders(defense_type):
    defenders = []
    if "Blitz" in defense_type:
        for _ in range(5):
            defenders.append((np.random.uniform(0, 20), np.random.uniform(10, 45)))
    elif "Cover 0" in defense_type:
        for _ in range(6):
            defenders.append((np.random.uniform(20, 40), np.random.uniform(0, FIELD_WIDTH)))
    elif "Cover 1" in defense_type:
        defenders.append((70, 26))
        for _ in range(5):
            defenders.append((np.random.uniform(20, 40), np.random.uniform(0, FIELD_WIDTH)))
    elif "Cover 2" in defense_type:
        defenders.append((70, 15))
        defenders.append((70, 40))
        for _ in range(5):
            defenders.append((np.random.uniform(20, 40), np.random.uniform(0, FIELD_WIDTH)))
    elif "Cover 3" in defense_type:
        defenders.append((70, 10))
        defenders.append((70, 26))
        defenders.append((70, 45))
        for _ in range(4):
            defenders.append((np.random.uniform(20, 40), np.random.uniform(0, FIELD_WIDTH)))
    elif "Cover 4" in defense_type:
        defenders.append((70, 10))
        defenders.append((70, 20))
        defenders.append((70, 35))
        defenders.append((70, 45))
        for _ in range(3):
            defenders.append((np.random.uniform(20, 40), np.random.uniform(0, FIELD_WIDTH)))
    elif "Man-to-Man" in defense_type:
        for _ in range(5):
            defenders.append((5, 25))
    return defenders

# Simulation d'un seul jeu
def simulate_one_play(offense_type, defense_type):
    defenders = generate_defenders(defense_type)
    rho_accumulated = 0
    for t in np.linspace(0, 1, 10):
        x, y = routes[offense_type](t)
        rho_accumulated += defense_density(x, y, defenders)
    mean_density = rho_accumulated / 10
    threshold = 0.5  # seuil critique de densité
    noise = np.random.normal(0, 0.05)
    success_probability = np.clip(1 - (mean_density/3) + noise, 0, 1)
    return np.random.rand() < success_probability

# Simulation complète
results_matrix = np.zeros((len(offense_types), len(defense_types)))

for i, offense in enumerate(offense_types):
    for j, defense in enumerate(defense_types):
        successes = 0
        for _ in range(N_SIMULATIONS):
            if simulate_one_play(offense, defense):
                successes += 1
        success_rate = successes / N_SIMULATIONS
        results_matrix[i, j] = success_rate

# Heatmap
plt.figure(figsize=(14, 9))
ax = sns.heatmap(results_matrix, annot=True, fmt=".1%", cmap="coolwarm", cbar_kws={'label': 'Taux de succès'})
ax.set_xticklabels(defense_types, rotation=45, ha="right")
ax.set_yticklabels(offense_types, rotation=0)
ax.set_xlabel("Défense")
ax.set_ylabel("Offense")
plt.title("Taux de succès offensif simulé avec défense par densité de probabilité")
plt.tight_layout()
plt.savefig("simulation_physique_avancee.png", dpi=300)
plt.show()
