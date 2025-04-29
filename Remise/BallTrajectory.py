import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class FootballSimulator:
    def __init__(self):
        # Constantes physiques fondamentales
        self.g = 9.81  # m/s², accélération gravitationnelle
        self.yards_to_m = 0.9144  # conversion yards -> mètres
        self.m_to_yards = 1.0936  # conversion mètres -> yards
        self.mph_to_mps = 0.44704  # conversion mph -> m/s

        # Propriétés du ballon (valeurs NFL standard)
        self.mass = 0.43  # kg
        self.length = 0.28  # m
        self.diameter = 0.17  # m
        # Section efficace moyenne (approximation d'une ellipse)
        self.cross_area = np.pi * self.length * self.diameter / 4

        # Propriétés de référence de l'air
        self.rho_ref = 1.225  # kg/m³ (densité de l'air à 15°C et 1 atm)
        self.temp_ref = 15.0 + 273.15  # K (température de référence en Kelvin)

        # Coefficients aérodynamiques de base (issus de données expérimentales)
        self.Cd_base = 0.18  # Coefficient de traînée de base
        self.Cl_base = 0.25  # Coefficient de portance de base pour angles optimaux
        self.magnus_factor = 0.0002  # Facteur d'effet Magnus (calibré)

    def air_density(self, temp_C):
        # Conversion en Kelvin
        temp_K = temp_C + 273.15

        # Loi des gaz parfaits
        return self.rho_ref * (self.temp_ref / temp_K)

    def calculate_forces(self, state, v_wind, temp_C, rpm):
        """
        Calcule les forces aérodynamiques agissant sur le ballon.

        :param state: État actuel [x, y, z, vx, vy, vz]
        :param v_wind : Vecteur vent [vx, vy, vz] m/s
        :param temp_C : Température C
        :param rpm : Rpm du ballon
        :return: Forces [Fx, Fy, Fz] en Newtons
        """
        # composantes de vitesse
        vx, vy, vz = state[3:6]

        # Vitesse relative
        v_rel = np.array([vx - v_wind[0], vy - v_wind[1], vz - v_wind[2]])
        v_mag = np.linalg.norm(v_rel)

        if v_mag < 0.1:
            return np.zeros(3)


        v_dir = v_rel / v_mag

        # Densité de l'air à la température donnée
        rho = self.air_density(temp_C)

        # Force de traînée
        drag_mag = 0.5 * rho * v_mag ** 2 * self.Cd_base * self.cross_area
        F_drag = -drag_mag * v_dir

        # Force de portance
        if np.abs(v_dir[1]) > 0.99:
            lift_dir = np.array([1.0, 0.0, 0.0])
        else:
            lift_dir = np.array([-v_dir[0] * v_dir[1],
                                 1.0 - v_dir[1] ** 2,
                                 -v_dir[2] * v_dir[1]])

            # Vecteur unitaire
            lift_dir = lift_dir / np.linalg.norm(lift_dir)

        # F_lift
        lift_mag = 0.5 * rho * v_mag ** 2 * self.Cl_base * self.cross_area
        F_lift = lift_mag * lift_dir

        # Effet Magnus (rotation)
        if rpm > 0 and v_mag > 0.1:
            omega = rpm * 2 * np.pi / 60

            if abs(v_dir[1]) > 0.9:
                rot_axis = np.array([1.0, 0.0, 0.0])
            else:
                horiz_dir = np.array([v_dir[0], 0.0, v_dir[2]])
                horiz_mag = np.linalg.norm(horiz_dir)

                if horiz_mag > 0.01:
                    rot_axis = horiz_dir / horiz_mag
                else:
                    rot_axis = np.array([1.0, 0.0, 0.0])

            magnus_dir = np.cross(rot_axis, v_dir)
            magnus_dir = magnus_dir / np.linalg.norm(magnus_dir)

            # Magnitude de la force Magnus
            magnus_mag = self.magnus_factor * rho * omega * v_mag * self.cross_area
            F_magnus = magnus_mag * magnus_dir
        else:
            F_magnus = np.zeros(3)

        # Force totale = Trainée + Portance + Magnus
        F_total = F_drag + F_lift + F_magnus

        return F_total

    def calculate_derivatives(self, state, v_wind, temp_C, rpm):
        """
        Calcule les dérivées temporelles de l'état (vitesses et accélérations).

        :param state: État actuel [x, y, z, vx, vy, vz]
        :param v_wind: Vecteur vent [vx, vy, vz] en m/s
        :param temp_C: Température en degrés Celsius
        :param rpm: Rotations par minute du ballon
        :return: Dérivées [dx/dt, dy/dt, dz/dt, dvx/dt, dvy/dt, dvz/dt]
        """
        # Vitesses
        dxdt, dydt, dzdt = state[3], state[4], state[5]

        # Accélérations = forces / masse
        F = self.calculate_forces(state, v_wind, temp_C, rpm)
        dvxdt, dvydt, dvzdt = F[0] / self.mass, F[1] / self.mass - self.g, F[2] / self.mass

        return np.array([dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt])

    def rk2_step(self, state, dt, v_wind, temp_C, rpm):
        """
        Effectue un pas d'intégration avec méthode RK2
        """
        k1 = self.calculate_derivatives(state, v_wind, temp_C, rpm)
        mid_state = state + dt / 2 * k1
        k2 = self.calculate_derivatives(mid_state, v_wind, temp_C, rpm)
        new_state = state + dt * k2

        return new_state

    def simulate_trajectory(self, initial_velocity_mph, launch_angle_v, launch_angle_h,
                            temp_C, rpm, wind_mph=[0, 0, 0], dt=0.01, max_time=6.0):
        """
        Simule la trajectoire complète du ballon.
        """
        v0 = initial_velocity_mph * self.mph_to_mps
        wind = [w * self.mph_to_mps for w in wind_mph]

        angle_v_rad = np.radians(launch_angle_v)
        angle_h_rad = np.radians(launch_angle_h)

        vx0 = v0 * np.cos(angle_v_rad) * np.cos(angle_h_rad)
        vy0 = v0 * np.sin(angle_v_rad)
        vz0 = v0 * np.cos(angle_v_rad) * np.sin(angle_h_rad)

        state = np.array([0.0, 0.0, 0.0, vx0, vy0, vz0])

        times = [0.0]
        positions = [[0.0, 0.0, 0.0]]  # Position initiale
        velocities = [[vx0 / self.mph_to_mps, vy0 / self.mph_to_mps, vz0 / self.mph_to_mps]]

        # Simulation jusqu'à ce que le ballon touche le sol ou temps max
        time = 0.0
        while state[1] >= 0 and time < max_time:  # Tant que y >= 0
            state = self.rk2_step(state, dt, wind, temp_C, rpm)
            time += dt

            times.append(time)
            positions.append([state[0] * self.m_to_yards,
                              state[1] * self.m_to_yards,
                              state[2] * self.m_to_yards])
            velocities.append([state[3] / self.mph_to_mps,
                               state[4] / self.mph_to_mps,
                               state[5] / self.mph_to_mps])

            if state[1] < 0:
                ratio = positions[-2][1] / (positions[-2][1] - positions[-1][1])
                positions[-1][0] = positions[-2][0] + ratio * (positions[-1][0] - positions[-2][0])
                positions[-1][2] = positions[-2][2] + ratio * (positions[-1][2] - positions[-2][2])
                positions[-1][1] = 0.0  # Au sol

        return np.array(times), np.array(positions), np.array(velocities)

    def analyze_temperature_impact(self, initial_velocity_mph, launch_angle_v, rpm,
                                   temp_range=(-20, 40, 13)):
        """
        Analyse l'impact de la température sur la distance parcourue.
        """
        temps = np.linspace(*temp_range)
        distances = []
        max_heights = []

        for temp in temps:
            _, positions, _ = self.simulate_trajectory(initial_velocity_mph, launch_angle_v, 0, temp, rpm)

            # Calcul de la distance parcourue (horiz)
            dist = np.sqrt(positions[-1, 0] ** 2 + positions[-1, 2] ** 2)
            distances.append(dist)

            # Hauteur max
            max_heights.append(np.max(positions[:, 1]))

        return temps, np.array(distances), np.array(max_heights)

    def analyze_rotation_impact(self, initial_velocity_mph, launch_angle_v, temp_C,
                                rpm_range=(0, 900, 10)):
        """
        Analyse l'impact de la rotation sur la distance parcourue.
        """
        rpms = np.linspace(*rpm_range)
        distances = []
        max_heights = []

        for rpm in rpms:
            _, positions, _ = self.simulate_trajectory(initial_velocity_mph, launch_angle_v, 0, temp_C, rpm)

            dist = np.sqrt(positions[-1, 0] ** 2 + positions[-1, 2] ** 2)
            distances.append(dist)

            # Hauteur maxi
            max_heights.append(np.max(positions[:, 1]))

        return rpms, np.array(distances), np.array(max_heights)

    def plot_trajectory(self, times, positions, temp_C=None, rpm=None, title=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # distance horizontale totale
        horiz_dist = np.sqrt(positions[:, 0] ** 2 + positions[:, 2] ** 2)

        ax.plot(horiz_dist, positions[:, 1])
        ax.set_xlabel('Distance horizontale (verges)')
        ax.set_ylabel('Hauteur (verges)')

        if title is None:
            title = "Trajectoire du ballon de football"
            if temp_C is not None:
                title += f" - Température: {temp_C}°C"
            if rpm is not None:
                title += f" - Rotation: {rpm} RPM"
        ax.set_title(title)
        ax.grid(True)
        ax.set_ylim(bottom=0)
        return ax

    def plot_temperature_impact(self, initial_velocity_mph, launch_angle_v, rpm, temp_range=(-20, 40, 13)):
        """
        Trace l'impact de la température sur la distance parcourue.
        """
        temps, distances, heights = self.analyze_temperature_impact(initial_velocity_mph, launch_angle_v, rpm, temp_range)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(temps, distances, 'o-', color='blue')
        ax1.set_ylabel('Distance (verges)')
        ax1.set_title(f'Impact de la température - Vitesse: {initial_velocity_mph} mph, Angle: {launch_angle_v}°, Rotation: {rpm} RPM')
        ax1.grid(True)

        ax2.plot(temps, heights, 'o-', color='green')
        ax2.set_xlabel('Température (°C)')
        ax2.set_ylabel('Hauteur max (verges)')
        ax2.grid(True)

        temp_diff = temps[-1] - temps[0]
        dist_diff = distances[-1] - distances[0]
        pct_change = (dist_diff / distances[0]) * 100

        stats_text = (f"Variation: {dist_diff:.1f} verges ({pct_change:.1f}%)\n"
                      f"entre {temps[0]:.0f}°C et {temps[-1]:.0f}°C")

        ax1.annotate(stats_text, xy=(0.02, 0.05), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

        plt.tight_layout()
        return fig

    def plot_rotation_impact(self, initial_velocity_mph, launch_angle_v, temp_C, rpm_range=(0, 900, 10)):
        """
        Trace l'impact de la rotation sur la distance parcourue.
        """
        rpms, distances, heights = self.analyze_rotation_impact(initial_velocity_mph, launch_angle_v, temp_C, rpm_range)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(rpms, distances, 'o-', color='blue')
        ax1.set_ylabel('Distance (verges)')
        ax1.set_title(f'Impact de la rotation - Vitesse: {initial_velocity_mph} mph, Angle: {launch_angle_v}°, Température: {temp_C}°C')
        ax1.grid(True)

        ax2.plot(rpms, heights, 'o-', color='green')
        ax2.set_xlabel('Vitesse de rotation (RPM)')
        ax2.set_ylabel('Hauteur max (verges)')
        ax2.grid(True)

        rpm_diff = rpms[-1] - rpms[0]
        dist_diff = distances[-1] - distances[0]
        pct_change = (dist_diff / distances[0]) * 100 if distances[0] > 0 else 0

        stats_text = (f"Variation: {dist_diff:.1f} verges ({pct_change:.1f}%)\n"
                      f"entre {rpms[0]:.0f} RPM et {rpms[-1]:.0f} RPM")

        ax1.annotate(stats_text, xy=(0.02, 0.05), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

        plt.tight_layout()
        return fig

    def plot_comparison_trajectories(self, initial_velocity_mph, launch_angle_v, conditions):
        """
        Trace et compare plusieurs trajectoires pour différentes conditions.
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        for condition in conditions:
            temp = condition.get("temp", 15)
            rpm = condition.get("rpm", 600)
            wind = condition.get("wind", [0, 0, 0])
            label = condition.get("label", f"{temp}°C, {rpm} RPM")

            _, positions, _ = self.simulate_trajectory(
                initial_velocity_mph, launch_angle_v, 0, temp, rpm, wind)

            horiz_dist = np.sqrt(positions[:, 0] ** 2 + positions[:, 2] ** 2)
            ax.plot(horiz_dist, positions[:, 1], label=label)
            ax.scatter(horiz_dist[-1], 0, marker='x')
            ax.annotate(f"{horiz_dist[-1]:.1f} verges",
                        (horiz_dist[-1], 0.5),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center')

        ax.set_xlabel('Distance horizontale (verges)')
        ax.set_ylabel('Hauteur (verges)')
        ax.set_title(f'Comparaison des trajectoires - Vitesse: {initial_velocity_mph} mph, Angle: {launch_angle_v}°')
        ax.grid(True)
        ax.set_ylim(bottom=0)
        ax.legend(loc='upper right')
        plt.tight_layout()
        return fig

def run_football_analysis():
    """
    Exécute une analyse complète
    """
    simulator = FootballSimulator()

    # Paramètres de base
    velocity = 45  # mph (vitesse qb NFL)
    angle = 30  # degrés
    temp = 15  # °C (température ref)
    rpm = 600  # rpm

    print("Simulation de la trajectoire de référence")
    times, positions, velocities = simulator.simulate_trajectory(velocity, angle, 0, temp, rpm)

    # Calculer la distance horizontale
    distance = np.sqrt(positions[-1, 0] ** 2 + positions[-1, 2] ** 2)
    max_height = np.max(positions[:, 1])

    print(f"Distance parcourue: {distance:.1f} verges")
    print(f"Hauteur maximale: {max_height:.1f} verges")
    print(f"Temps de vol: {times[-1]:.2f} secondes")

    # Tracer la trajectoire de référence
    plt.figure(figsize=(10, 6))
    simulator.plot_trajectory(times, positions, temp, rpm, title="Trajectoire de référence")
    plt.savefig("trajectoire_reference.png")
    plt.close()

    # impact de la température
    fig_temp = simulator.plot_temperature_impact(velocity, angle, rpm)
    plt.savefig("impact_temperature.png")
    plt.close()

    # impact de la rotation
    fig_rot = simulator.plot_rotation_impact(velocity, angle, temp)
    plt.savefig("impact_rotation.png")
    plt.close()

    # Comparer différentes conditions
    conditions = [
        {"temp": -10, "rpm": 600, "label": "Froid (-10°C)"},
        {"temp": 15, "rpm": 600, "label": "Standard (15°C)"},
        {"temp": 30, "rpm": 600, "label": "Chaud (30°C)"},
        {"temp": 15, "rpm": 0, "label": "Sans rotation"}
    ]
    fig_comp = simulator.plot_comparison_trajectories(velocity, angle, conditions)
    plt.savefig("comparaison_trajectoires.png")
    plt.close()

    # Impact de la température et de la rotation
    _, pos_cold, _ = simulator.simulate_trajectory(velocity, angle, 0, -10, rpm)
    _, pos_warm, _ = simulator.simulate_trajectory(velocity, angle, 0, 30, rpm)
    _, pos_no_spin, _ = simulator.simulate_trajectory(velocity, angle, 0, temp, 0)

    dist_cold = np.sqrt(pos_cold[-1, 0] ** 2 + pos_cold[-1, 2] ** 2)
    dist_warm = np.sqrt(pos_warm[-1, 0] ** 2 + pos_warm[-1, 2] ** 2)
    dist_no_spin = np.sqrt(pos_no_spin[-1, 0] ** 2 + pos_no_spin[-1, 2] ** 2)

    temp_effect = ((dist_warm - dist_cold) / dist_warm) * 100
    spin_effect = ((distance - dist_no_spin) / distance) * 100

    print("\nAnalyse des effets relatifs:")
    print(f"Impact de la température: réduction de {temp_effect:.1f}% à -10°C vs 30°C")
    print(f"Impact de la rotation: réduction de {spin_effect:.1f}% sans rotation")

    print("\nAnalyse complétée. Les graphiques ont été enregistrés.")

    return {
        "simulator": simulator,
        "reference": {
            "distance": distance,
            "max_height": max_height,
            "flight_time": times[-1]
        },
        "effects": {
            "temperature": temp_effect,
            "rotation": spin_effect
        }
    }


# Pour exécuter l'analyse
if __name__ == "__main__":
    results = run_football_analysis()