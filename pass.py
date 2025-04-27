import numpy as np
import matplotlib.pyplot as plt

def compute_air_density(temp_C):
    """
    Calcule un facteur de densité de l'air basé sur la température (°C).
    Référence simple: rho ∝ 1/T (loi des gaz parfaits).
    """
    T0 = 15 + 273.15  # Température standard (15°C) en Kelvin
    T = temp_C + 273.15
    return T0 / T  # Plus il fait chaud, plus c'est léger -> densité plus faible

def simulate_pass_with_drag_and_temp(x0, y0, v0, theta_deg, temp_C=15, dt=0.01, base_c_d=0.05):
    """
    Simule une passe de football en tenant compte du frottement de l'air
    modulé par la température.
    
    Paramètres:
        x0, y0 : Position initiale (yards)
        v0 : Vitesse initiale (yards/s)
        theta_deg : Angle de lancement (degrés)
        temp_C : Température en degrés Celsius
        dt : Pas de temps (secondes)
        base_c_d : Coefficient de frottement de référence à 15°C
        
    Retourne:
        x_array, y_array : Trajectoire complète
    """
    g = 9.81 * 1.094  # gravité en yards/s²
    theta = np.radians(theta_deg)
    
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)
    
    x, y = x0, y0
    t = 0
    x_array = []
    y_array = []
    
    # Calcul du coefficient de frottement effectif
    density_factor = compute_air_density(temp_C)
    c_d_effective = base_c_d * density_factor
    
    while y >= 0:
        x_array.append(x)
        y_array.append(y)
        
        v = np.sqrt(vx**2 + vy**2)
        
        # Forces avec frottement
        ax = -c_d_effective * v * vx
        ay = -g - c_d_effective * v * vy
        
        # Mise à jour des vitesses
        vx += ax * dt
        vy += ay * dt
        
        # Mise à jour des positions
        x += vx * dt
        y += vy * dt
        
        t += dt
        
    return np.array(x_array), np.array(y_array)

# 🏈 Exemple d'utilisation :
x0, y0 = 0, 1  # départ 1 yard au-dessus du sol
v0 = 30  # yards/s
theta_deg = 45

# Simuler pour différentes températures
x_cold, y_cold = simulate_pass_with_drag_and_temp(x0, y0, v0, theta_deg, temp_C=-10)  # froid
x_mild, y_mild = simulate_pass_with_drag_and_temp(x0, y0, v0, theta_deg, temp_C=15)   # standard
x_hot, y_hot = simulate_pass_with_drag_and_temp(x0, y0, v0, theta_deg, temp_C=35)     # chaud

# Trouver les interceptions y=1
def find_y_interception(x_array, y_array, target_y=1):
    for i in range(len(y_array) - 1):
        if y_array[i] >= target_y and y_array[i + 1] < target_y:
            # Interpolation linéaire pour trouver l'interception exacte
            t = (target_y - y_array[i]) / (y_array[i + 1] - y_array[i])
            return x_array[i] + t * (x_array[i + 1] - x_array[i])
    return None

x_cold_intercept = find_y_interception(x_cold, y_cold)
x_mild_intercept = find_y_interception(x_mild, y_mild)
x_hot_intercept = find_y_interception(x_hot, y_hot)

print(f"Interception y=1 (froid): {x_cold_intercept:.2f} yards")
print(f"Interception y=1 (tempéré): {x_mild_intercept:.2f} yards")
print(f"Interception y=1 (chaud): {x_hot_intercept:.2f} yards")

# 🎨 Tracer
plt.figure(figsize=(10,6))
plt.plot(x_cold, y_cold, label="Froid (-10°C)")
plt.plot(x_mild, y_mild, label="Tempéré (15°C)")
plt.plot(x_hot, y_hot, label="Chaud (35°C)")
plt.xlabel('Distance horizontale (yards)')
plt.ylabel('Hauteur (yards)')
plt.title('Trajectoire de la passe selon la température')
plt.legend()
plt.grid(True)
plt.show()