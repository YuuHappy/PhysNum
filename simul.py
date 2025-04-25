import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Fonction sigmoïde (forme classique d'une probabilité de succès)
def sigmoid(x, L=1, k=1, x0=0):
    """Forme générale d'une courbe sigmoïde"""
    return L / (1 + np.exp(-k * (x - x0)))

# Chargement de tes données (exemple simplifié)
tracking = pd.read_csv('tracking.csv')
plays = pd.read_csv('plays.csv')

# Étape 1: Isolation des frames autour du "ball_snap"
snap_tracking = tracking[tracking['event'] == 'ball_snap']

# Merge avec plays pour avoir yards_gained
snap_tracking = snap_tracking.merge(plays[['gameId', 'playId', 'yards_gained']], on=['gameId', 'playId'])

# Étape 2: Construction de variables physiques
# Exemple : distance minimale défenseur -> porteur de ballon

# Suppose que le porteur de ballon est le joueur avec "ball" dans son nom
# Tu devras ajuster selon la structure exacte de ton fichier
ball_carrier = snap_tracking[snap_tracking['displayName'] == 'football']

# Pour chaque jeu, trouver le défenseur le plus proche
distances = []
for playId in ball_carrier['playId'].unique():
    play_data = snap_tracking[snap_tracking['playId'] == playId]
    ball_pos = play_data[play_data['displayName'] == 'football'][['x', 'y']].values
    if ball_pos.shape[0] == 0:
        continue
    ball_pos = ball_pos[0]

    defenders = play_data[play_data['team'] != play_data['poss_team']][['x', 'y']]
    if defenders.empty:
        continue
    dists = np.linalg.norm(defenders.values - ball_pos, axis=1)
    distances.append((playId, dists.min()))

dist_df = pd.DataFrame(distances, columns=['playId', 'min_defender_distance'])

# Étape 3: Joindre avec yards gained
dist_df = dist_df.merge(plays[['playId', 'yards_gained']], on='playId')

# Définir succès/échec
dist_df['success'] = (dist_df['yards_gained'] >= 3).astype(int)

# Étape 4: Fit d'une courbe sigmoïde
xdata = dist_df['min_defender_distance']
ydata = dist_df['success']

# Ajuster la sigmoïde aux données
params, _ = curve_fit(sigmoid, xdata, ydata, method='dogbox')

# Affichage
x_fit = np.linspace(xdata.min(), xdata.max(), 100)
y_fit = sigmoid(x_fit, *params)

plt.scatter(xdata, ydata, label="Données brutes", alpha=0.5)
plt.plot(x_fit, y_fit, color='red', label="Fit sigmoïde")
plt.xlabel('Distance minimale défenseur-porteur (yards)')
plt.ylabel('Probabilité de succès')
plt.title('Modèle physique numérique - Probabilité de succès')
plt.legend()
plt.grid(True)
plt.show()

# Étape 5: Utilisation pratique
def predict_success_probability(distance):
    """Prédis la probabilité de succès d'un jeu selon distance défenseur-porteur"""
    return sigmoid(distance, *params)

# Exemple
print(f"Proba de succès si défenseur à 5 yards: {predict_success_probability(5):.2%}")
print(f"Proba de succès si défenseur à 2 yards: {predict_success_probability(2):.2%}")
