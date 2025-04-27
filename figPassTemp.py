import numpy as np
import matplotlib.pyplot as plt

def compute_air_density(temp_C):
    """
    Calcule un facteur de densité de l'air basé sur la température (°C).
    """
    T0 = 15 + 273.15  # Référence standard (15 °C) en Kelvin
    T = temp_C + 273.15
    return T0 / T

def simulate_pass_length_RK2(x0, y0, v0, theta_deg, temp_C=15, dt=0.01, base_c_d=0.05):
    """
    Simule une passe avec Runge-Kutta d'ordre 2 et retourne la distance maximale atteinte.
    """
    g = 9.81 * 1.094  # Gravité en yards/s²
    theta = np.radians(theta_deg)
    
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)
    
    x, y = x0, y0
    t = 0
    
    density_factor = compute_air_density(temp_C)
    c_d_effective = base_c_d * density_factor
    
    while y >= 0:
        v = np.sqrt(vx**2 + vy**2)
        
        # --- Étape 1 : évaluer dérivées au début ---
        ax1 = -c_d_effective * v * vx
        ay1 = -g - c_d_effective * v * vy
        
        # Prévision intermédiaire (point médian)
        vx_mid = vx + ax1 * dt/2
        vy_mid = vy + ay1 * dt/2
        
        # Calculer la vitesse moyenne
        v_mid = np.sqrt(vx_mid**2 + vy_mid**2)
        
        # --- Étape 2 : évaluer dérivées au milieu ---
        ax2 = -c_d_effective * v_mid * vx_mid
        ay2 = -g - c_d_effective * v_mid * vy_mid
        
        # Mise à jour finale
        vx += ax2 * dt
        vy += ay2 * dt
        
        x += vx * dt
        y += vy * dt
        
        t += dt
        
    return x  # Distance horizontale atteinte

#Paramètres de la passe
x0, y0 = 0, 1      # Départ 1 yard au-dessus du sol
v0 = 30            # Vitesse initiale en yards/s
theta_deg = 45     # Angle pour portée maximale

#Températures de -20°C à 40°C
temps = np.linspace(-20, 40, 100)
distances = []

for temp in temps:
    distance = simulate_pass_length_RK2(x0, y0, v0, theta_deg, temp_C=temp)
    distances.append(distance)

#Affichage
plt.figure(figsize=(10,6))
plt.plot(temps, distances, color='darkgreen')
plt.xlabel('Température extérieure (°C)')
plt.ylabel('Distance maximale de la passe (verges)')
plt.grid(True)
plt.show()
