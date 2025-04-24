    import numpy as np
    import matplotlib.pyplot as plt

    # Taille du terrain
    field_length = 100  # verges
    field_width = 53    # verges

    # Grille spatiale
    x = np.linspace(0, field_length, 200)
    y = np.linspace(0, field_width, 100)
    X, Y = np.meshgrid(x, y)

    # Heatmap défensive synthétique (zones à haute densité défensive)
    defense_heatmap = (
        0.5 * np.exp(-((X - 30) ** 2 + (Y - 26) ** 2) / (2 * 15 ** 2)) +
        0.8 * np.exp(-((X - 60) ** 2 + (Y - 10) ** 2) / (2 * 10 ** 2)) +
        0.8 * np.exp(-((X - 60) ** 2 + (Y - 43) ** 2) / (2 * 10 ** 2)) +
        0.3 * np.exp(-((X - 80) ** 2 + (Y - 26) ** 2) / (2 * 20 ** 2))
    )
    defense_heatmap /= defense_heatmap.max()  # Normalisation

    # Fonctions trajectoires offensives
    def course_interieure():
        x = np.linspace(0, 40, 100)
        y = np.full_like(x, field_width / 2)
        return x, y

    def passe_courte():
        x = np.linspace(0, 30, 100)
        y = field_width / 2 + 10 * np.sin(np.linspace(0, np.pi, 100))
        return x, y

    def passe_longue():
        x = np.linspace(0, 70, 100)
        y = np.full_like(x, field_width / 2)
        return x, y

    # Estimation du gain de verges
    def gain_simulé(xtraj, ytraj, heatmap, Xgrid, Ygrid, seuil=0.5):
        gain = len(xtraj)
        for i in range(len(xtraj)):
            x_i = xtraj[i]
            y_i = ytraj[i]
            ix = np.abs(Xgrid[0] - x_i).argmin()
            iy = np.abs(Ygrid[:,0] - y_i).argmin()
            if heatmap[iy, ix] > seuil:
                gain = i
                break
        return x[gain] if gain < len(xtraj) else xtraj[-1]

    # Calcul des gains pour chaque trajectoire
    x1, y1 = course_interieure()
    x2, y2 = passe_courte()
    x3, y3 = passe_longue()

    gain1 = gain_simulé(x1, y1, defense_heatmap, X, Y, seuil=0.4)
    gain2 = gain_simulé(x2, y2, defense_heatmap, X, Y, seuil=0.4)
    gain3 = gain_simulé(x3, y3, defense_heatmap, X, Y, seuil=0.4)

    # Visualisation
    plt.figure(figsize=(12, 6))
    plt.imshow(defense_heatmap, extent=[0, field_length, 0, field_width], origin='lower', cmap='viridis', alpha=0.8)
    plt.plot(x1, y1, label=f"Course intérieure ({gain1:.1f} verges)", color="white", linewidth=2)
    plt.plot(x2, y2, label=f"Passe courte ({gain2:.1f} verges)", color="orange", linewidth=2)
    plt.plot(x3, y3, label=f"Passe longue ({gain3:.1f} verges)", color="cyan", linewidth=2)
    plt.legend(loc='upper right')
    plt.title("Simulation de gain de verges avec heatmap défensive")
    plt.xlabel("Distance (verges)")
    plt.ylabel("Largeur du terrain (verges)")
    plt.tight_layout()
    plt.grid(False)
    plt.show()