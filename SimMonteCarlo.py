# === IMPORTS ===
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# === PARAMETRES GENERAUX ===
field_length = 100  # verges
field_width = 53
n_simulations = 500
n_defenders = 11
tackle_radius = 5

x = np.linspace(0, field_length, 200)
y = np.linspace(0, field_width, 100)
X, Y = np.meshgrid(x, y)

# === DEFINIR PLUSIEURS SCHEMAS DEFENSIFS ===
def heatmap_zone_standard(X, Y):
    return (
        0.5 * np.exp(-((X - 30)**2 + (Y - 26)**2) / (2 * 15**2)) +
        0.8 * np.exp(-((X - 60)**2 + (Y - 10)**2) / (2 * 10**2)) +
        0.8 * np.exp(-((X - 60)**2 + (Y - 43)**2) / (2 * 10**2)) +
        0.3 * np.exp(-((X - 80)**2 + (Y - 26)**2) / (2 * 20**2))
    )

def heatmap_blitz(X, Y):
    return (
        1.0 * np.exp(-((X - 10)**2 + (Y - 26)**2) / (2 * 10**2)) +
        0.5 * np.exp(-((X - 20)**2 + (Y - 10)**2) / (2 * 15**2)) +
        0.5 * np.exp(-((X - 20)**2 + (Y - 43)**2) / (2 * 15**2))
    )

def heatmap_zone_profonde(X, Y):
    return (
        0.9 * np.exp(-((X - 70)**2 + (Y - 26)**2) / (2 * 20**2)) +
        0.7 * np.exp(-((X - 85)**2 + (Y - 10)**2) / (2 * 20**2)) +
        0.7 * np.exp(-((X - 85)**2 + (Y - 43)**2) / (2 * 20**2))
    )

# Normalisation de heatmaps
def normalize_heatmap(heatmap):
    heatmap /= heatmap.max()
    return heatmap

# === TRAJECTOIRES ===
def passe_longue():
    x = np.linspace(0, 70, 100)
    y = np.full_like(x, field_width / 2)
    return x, y

# === GENERATION DES DEFENSEURS ===
def generate_defenders(n_defenders, heatmap, Xgrid, Ygrid):
    defenders = []
    flat_heatmap = heatmap.flatten()
    flat_heatmap /= flat_heatmap.sum()
    idx_choices = np.arange(flat_heatmap.size)
    chosen_idx = np.random.choice(idx_choices, size=n_defenders, p=flat_heatmap)
    for idx in chosen_idx:
        iy, ix = np.unravel_index(idx, heatmap.shape)
        x_pos = Xgrid[0, ix]
        y_pos = Ygrid[iy, 0]
        defenders.append((x_pos, y_pos))
    return defenders

# === SIMULATION D'UN GAIN ===
def simulate_gain(xtraj, ytraj, defenders, tackle_radius=5):
    for i in range(len(xtraj)):
        x_i = xtraj[i]
        y_i = ytraj[i]
        for (xd, yd) in defenders:
            distance = np.sqrt((x_i - xd)**2 + (y_i - yd)**2)
            if distance <= tackle_radius:
                return xtraj[i]
    return xtraj[-1]

# === METHODE DE SIMULATION MONTE CARLO ===
def monte_carlo_simulation(xtraj, ytraj, heatmap):
    gains = []
    for _ in range(n_simulations):
        defenders = generate_defenders(n_defenders, heatmap, X, Y)
        gain = simulate_gain(xtraj, ytraj, defenders, tackle_radius)
        gains.append(gain)
    return np.array(gains)

# === AFFICHER LES HISTOGRAMMES DE GAINS ===
def plot_histograms():
    xtraj, ytraj = passe_longue()

    gains_standard = monte_carlo_simulation(xtraj, ytraj, normalize_heatmap(heatmap_zone_standard(X, Y)))
    gains_blitz = monte_carlo_simulation(xtraj, ytraj, normalize_heatmap(heatmap_blitz(X, Y)))
    gains_profond = monte_carlo_simulation(xtraj, ytraj, normalize_heatmap(heatmap_zone_profonde(X, Y)))

    plt.figure(figsize=(10,6))
    plt.hist(gains_standard, bins=20, alpha=0.7, label=f"Zone Standard (Moy: {np.mean(gains_standard):.1f}v)")
    plt.hist(gains_blitz, bins=20, alpha=0.7, label=f"Blitz (Moy: {np.mean(gains_blitz):.1f}v)")
    plt.hist(gains_profond, bins=20, alpha=0.7, label=f"Zone Profonde (Moy: {np.mean(gains_profond):.1f}v)")
    plt.title("Simulation Monte-Carlo - Comparaison de schémas défensifs")
    plt.xlabel("Verges gagnées")
    plt.ylabel("Fréquence")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === CREER L'ANIMATION ===
def create_animation():
    xtraj, ytraj = passe_longue()
    defenders = generate_defenders(n_defenders, normalize_heatmap(heatmap_zone_standard(X, Y)), X, Y)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, field_length)
    ax.set_ylim(0, field_width)
    ax.set_title("Animation d'une passe longue avec défenseurs")

    ball, = ax.plot([], [], 'ro', label='Porteur')
    defenders_plot, = ax.plot([], [], 'kx', label='Défenseurs')

    def init():
        ball.set_data([], [])
        defenders_plot.set_data([], [])
        return ball, defenders_plot

    def animate(i):
        if i < len(xtraj):
            ball.set_data(xtraj[i], ytraj[i])
        defenders_plot.set_data(*zip(*defenders))
        return ball, defenders_plot

    ani = animation.FuncAnimation(fig, animate, frames=len(xtraj), init_func=init,
                                  interval=100, blit=True, repeat=False)
    plt.legend()
    plt.show()

# === EXECUTION ===
plot_histograms()
create_animation()
