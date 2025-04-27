import numpy as np
import matplotlib.pyplot as plt

def compute_air_density(temp_C):
    """
    Calcule un facteur de densité de l'air basé sur la température (°C).
    """
    T0 = 15 + 273.15  # Référence standard (15 °C) en Kelvin
    T = temp_C + 273.15
    return T0 / T

def simulate_pass_length(x0, y0, v0, theta_deg, temp_C=15, dt=0.01, base_c_d=0.05):
    """
    Simule une passe et retourne la distance maximale atteinte.
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
        
        ax = -c_d_effective * v * vx
        ay = -g - c_d_effective * v * vy
        
        vx += ax * dt
        vy += ay * dt
        
        x += vx * dt
        y += vy * dt
        
        t += dt
        
    return x  # Distance horizontale atteinte au moment où y=0

# 🔥 Paramètres de la passe
x0, y0 = 0, 1      # Départ à 1 yard d'altitude
v0 = 30            # Vitesse initiale en yards/s
theta_deg = 45     # Angle de tir optimal pour la distance max

# 🌡️ Températures de -20°C à 40°C
temps = np.linspace(-20, 40, 100)
distances = []

for temp in temps:
    distance = simulate_pass_length(x0, y0, v0, theta_deg, temp_C=temp)
    distances.append(distance)

# 🎨 Affichage
plt.figure(figsize=(10,6))
plt.plot(temps, distances, color='navy')
plt.xlabel('Température extérieure (°C)')
plt.ylabel('Distance maximale de la passe (yards)')
plt.grid(True)
plt.show()
