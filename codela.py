# === IMPORTS ===
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# === CLASSES DE BASE ===

class Joueur:
    def __init__(self, waypoints, vitesse, couleur='blue', label='Offensif'):
        self.waypoints = waypoints
        self.vitesse = vitesse
        self.total_distance = self._calculer_total_distance()
        self.couleur = couleur
        self.label = label

    def _calculer_total_distance(self):
        distance = 0
        for i in range(len(self.waypoints) - 1):
            dx = self.waypoints[i+1][0] - self.waypoints[i][0]
            dy = self.waypoints[i+1][1] - self.waypoints[i][1]
            distance += np.sqrt(dx**2 + dy**2)
        return distance

    def position(self, t):
        distance_parcourue = self.vitesse * t
        distance_cumulee = 0
        for i in range(len(self.waypoints) - 1):
            (x0, y0) = self.waypoints[i]
            (x1, y1) = self.waypoints[i+1]
            segment_length = np.sqrt((x1-x0)**2 + (y1-y0)**2)
            if distance_cumulee + segment_length >= distance_parcourue:
                ratio = (distance_parcourue - distance_cumulee) / segment_length
                x = x0 + ratio * (x1 - x0)
                y = y0 + ratio * (y1 - y0)
                return (x, y)
            distance_cumulee += segment_length
        return self.waypoints[-1]

class Defenseur(Joueur):
    def __init__(self, waypoints, vitesse, r_bras, skill, couleur='red', label='Défenseur'):
        super().__init__(waypoints, vitesse, couleur, label)
        self.r_bras = r_bras
        self.skill = skill

    def densite(self, x, y, t):
        x_def, y_def = self.position(t)
        return self.skill * np.exp(-((x - x_def)**2 + (y - y_def)**2) / (2 * self.r_bras**2))

# === FORMATIONS ===

class FormationOffensive:
    def __init__(self, joueurs):
        self.joueurs = joueurs

    def positions(self, t):
        return [joueur.position(t) for joueur in self.joueurs]

class CouvertureDefensive:
    def __init__(self, defenseurs):
        self.defenseurs = defenseurs

    def positions(self, t):
        return [defenseur.position(t) for defenseur in self.defenseurs]

    def densite_totale(self, x, y, t):
        return sum(defenseur.densite(x, y, t) for defenseur in self.defenseurs)

# === INTERACTION ===

def interaction(densite, position, taille_objet):
    x, y = position
    rho = densite(x, y)
    u = np.random.uniform(0, 1)
    return u < rho

# === ROUTES OFFENSIVES ===

def route_passe_longue():
    return [(0, 26), (100, 26)]

def route_slant():
    return [(0, 26), (10, 30), (30, 40)]

def route_curl():
    return [(0, 26), (15, 26), (15, 20)]

def route_courte():
    return [(0, 26), (20, 26)]

# === PARAMETRES GENERAUX ===
field_length = 100
field_width = 53

# === EXEMPLE DE JEU COMPLET ===

def exemple_jeu_complet():
    # Création des joueurs offensifs
    porteur = Joueur(route_courte(), vitesse=5, couleur='blue', label='Porteur')
    receveur1 = Joueur(route_passe_longue(), vitesse=6, couleur='green', label='Receveur 1')
    receveur2 = Joueur(route_curl(), vitesse=5, couleur='purple', label='Receveur 2')

    formation_offensive = FormationOffensive([porteur, receveur1, receveur2])

    # Création des défenseurs
    defenseur1 = Defenseur(route_slant(), vitesse=4, r_bras=5, skill=0.8, couleur='red', label='Défenseur 1')
    defenseur2 = Defenseur(route_courte(), vitesse=4, r_bras=5, skill=0.7, couleur='red', label='Défenseur 2')

    couverture_defensive = CouvertureDefensive([defenseur1, defenseur2])

    # Création de la figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, field_length)
    ax.set_ylim(0, field_width)
    ax.set_title("Simulation d'un jeu offensif contre défense")
    ax.set_xlabel("Verges")
    ax.set_ylabel("Largeur du terrain")

    # Dessiner le terrain
    for yd in range(0, field_length+10, 10):
        ax.axvline(x=yd, color='white', linestyle='--', alpha=0.5)
    ax.axhspan(0, field_width, facecolor='lightgreen', alpha=0.3)

    # Création des plots dynamiques
    offensifs_plot, = ax.plot([], [], 'o', markersize=8, color='blue', label='Offense')
    receveurs_plot, = ax.plot([], [], 'o', markersize=8, color='green')
    defenseurs_plot, = ax.plot([], [], 'x', markersize=8, color='red', label='Défense')

    def init():
        offensifs_plot.set_data([], [])
        receveurs_plot.set_data([], [])
        defenseurs_plot.set_data([], [])
        return offensifs_plot, receveurs_plot, defenseurs_plot

    def animate(i):
        t = i * 0.2
        pos_off = formation_offensive.positions(t)
        pos_def = couverture_defensive.positions(t)

        if pos_off:
            x_off, y_off = zip(*pos_off)
            offensifs_plot.set_data(x_off[0:1], y_off[0:1])
            receveurs_plot.set_data(x_off[1:], y_off[1:])

        if pos_def:
            x_def, y_def = zip(*pos_def)
            defenseurs_plot.set_data(x_def, y_def)

        return offensifs_plot, receveurs_plot, defenseurs_plot

    ani = animation.FuncAnimation(fig, animate, frames=100, init_func=init,
                                  interval=100, blit=True, repeat=False)
    plt.legend()
    plt.show()

# === EXECUTION DU JEU COMPLET ===
exemple_jeu_complet()
