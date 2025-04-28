"""Module 2 - Première Simulation Monte Carlo

Ce module est responsable de:
1. Utiliser les paramètres extraits des données NFL pour simuler des jeux
2. Générer un grand nombre de drives avec des choix stratégiques variés
3. Produire un ensemble de données d'entraînement pour le modèle d'apprentissage machine
"""

import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from tqdm import tqdm  # Pour les barres de progression
import seaborn as sns

viz_dir = "Visuals"
output_dir = "OutputFiles"


class FootballField:
    """Classe représentant le terrain et la position d'une équipe."""

    def __init__(self):
        self.length = 100  # Yards
        self.reset()

    def reset(self):
        """Réinitialise le terrain pour un nouveau drive à la ligne des 20 yards."""
        self.position = 20  # Position par défaut: ligne des 20 yards
        self.down = 1
        self.yards_to_go = 10
        self.points = 0
        self.drive_history = []

    def move(self, yards_gained):
        """
        Déplace le ballon de yards_gained yards sur le terrain.

        Args:
            yards_gained: Nombre de yards gagnés (négatif si perte)

        Returns:
            str: État du drive ('CONTINUE', 'TOUCHDOWN', 'TURNOVER_ON_DOWNS')
        """
        self.position += yards_gained
        self.drive_history.append({
            'position': self.position,
            'down': self.down,
            'yards_to_go': self.yards_to_go,
            'yards_gained': yards_gained
        })

        # Vérifier si premier down obtenu
        if yards_gained >= self.yards_to_go:
            self.down = 1
            self.yards_to_go = 10
        else:
            self.down += 1
            self.yards_to_go -= yards_gained

        # Limiter la position à la longueur du terrain
        self.position = min(self.position, self.length)

        # Vérifier si touchdown
        if self.position >= self.length:
            return "TOUCHDOWN"

        # Vérifier si le drive continue
        if self.down > 4:
            return "TURNOVER_ON_DOWNS"  # 4ème tentative échouée

        return "CONTINUE"  # Le jeu continue

class NFLPlayResult:
    """Classe pour générer les résultats des jeux basés sur des distributions de probabilités calibrées."""

    def __init__(self, params, seed=None):
        """
        Initialise les distributions pour les résultats de jeux avec des paramètres réels.

        Args:
            params: Dictionnaire de paramètres extraits des données réelles
            seed: Seed pour la reproductibilité
        """
        if seed is not None:
            np.random.seed(seed)

        self.params = params

        # Paramètres de base
        self.run_mean = params['run_mean']
        self.run_std = params['run_std']
        self.pass_mean = params['pass_mean']
        self.pass_std = params['pass_std']
        self.run_fumble_prob = params['fumble_rate']
        self.pass_interception_prob = params['interception_rate']
        self.pass_completion_rate = params['completion_rate']

        # Paramètres avancés
        self.down_adjustments = params.get('down_adjustments', {})
        self.field_adjustments = params.get('field_adjustments', {})

    def _get_field_position_category(self, position):
        """Convertit une position sur le terrain en catégorie."""
        if position <= 20:
            return '0-20'
        elif position <= 50:
            return '20-50'
        elif position <= 80:
            return '50-80'
        else:
            return '80-100'

    def run_play(self, position, down, yards_to_go):
        """
        Simule le résultat d'un jeu de course avec des paramètres basés sur les données réelles.

        Args:
            position: Position sur le terrain (yards)
            down: Le down actuel (1-4)
            yards_to_go: Yards à franchir pour un premier down

        Returns:
            tuple: (résultat, yards gagnés)
        """
        # Vérifier si fumble
        if np.random.random() < self.run_fumble_prob:
            return "FUMBLE", 0

        # Ajustements de base
        mean_adjustment = 0
        std_adjustment = 0

        # Ajustements par down
        if down in self.down_adjustments and 'run' in self.down_adjustments[down]:
            mean_adjustment += self.down_adjustments[down]['run']['mean_adjust']
            std_adjustment += self.down_adjustments[down]['run']['std_adjust']

        # Ajustements par position sur le terrain
        field_cat = self._get_field_position_category(position)
        if field_cat in self.field_adjustments and 'run' in self.field_adjustments[field_cat]:
            mean_adjustment += self.field_adjustments[field_cat]['run']['mean_adjust']

        # Ajustements spécifiques additionnels
        if down == 3 and yards_to_go <= 2:  # 3ème tentative et courte distance
            mean_adjustment += 0.5  # Bonus en situation de courte distance
        elif down == 4:  # 4ème tentative
            mean_adjustment -= 0.5  # Malus en 4ème tentative (stress)

        if position >= 90:  # Proche de l'en-but adverse
            mean_adjustment -= 1.0  # Plus difficile de gagner des yards
            std_adjustment -= 0.5  # Moins de variabilité

        # Générer le résultat avec la distribution ajustée
        yards = np.random.normal(self.run_mean + mean_adjustment, max(0.5, self.run_std + std_adjustment))  # Éviter un écart-type négatif

        # Arrondir à l'entier le plus proche
        return "SUCCESS", int(round(yards))

    def pass_play(self, position, down, yards_to_go):
        """
        Simule le résultat d'un jeu de passe avec des paramètres basés sur les données réelles.

        Args:
            position: Position sur le terrain (yards)
            down: Le down actuel (1-4)
            yards_to_go: Yards à franchir pour un premier down

        Returns:
            tuple: (résultat, yards gagnés)
        """
        # Vérifier si interception
        if np.random.random() < self.pass_interception_prob:
            return "INTERCEPTION", 0

        # Vérifier si la passe est complète
        if np.random.random() > self.pass_completion_rate:
            return "SUCCESS", 0  # Passe incomplète, 0 yards

        # Ajustements de base
        mean_adjustment = 0
        std_adjustment = 0

        # Ajustements par down
        if down in self.down_adjustments and 'pass' in self.down_adjustments[down]:
            mean_adjustment += self.down_adjustments[down]['pass']['mean_adjust']
            std_adjustment += self.down_adjustments[down]['pass']['std_adjust']

        # Ajustements par position sur le terrain
        field_cat = self._get_field_position_category(position)
        if field_cat in self.field_adjustments and 'pass' in self.field_adjustments[field_cat]:
            mean_adjustment += self.field_adjustments[field_cat]['pass']['mean_adjust']

        # Ajustements spécifiques additionnels
        if down == 3 and yards_to_go > 5:  # 3ème tentative et longue distance
            mean_adjustment += 1.0  # Bonus en situation de longue distance
            std_adjustment += 1.0  # Plus de variabilité
        elif yards_to_go <= 8:
            mean_adjustment += 0.5  # Bonus en situation de longue distance
            std_adjustment += 0.5
        if down == 4:  # 4ème tentative
            mean_adjustment += 0.5  # Bonus en 4ème tentative (tout ou rien)
            std_adjustment += 1.5  # Plus de variabilité

        if position >= 90:  # Proche de l'en-but adverse
            mean_adjustment -= 1.0  # Plus difficile de gagner des yards
            std_adjustment -= 0.5  # Moins de variabilité

        # Générer le résultat avec la distribution ajustée
        yards = np.random.normal(self.pass_mean + mean_adjustment, max(0.5, self.pass_std + std_adjustment))  # Éviter un écart-type négatif

        # Arrondir à l'entier le plus proche
        return "SUCCESS", int(round(yards))

class NFLPlaySelector:
    """Classe pour sélectionner le type de jeu en fonction des données réelles."""

    def __init__(self, params):
        """
        Initialise le sélecteur de jeu avec des probabilités basées sur les données réelles.

        Args:
            params: Dictionnaire de paramètres extraits des données réelles
        """
        self.play_choice_stats = params.get('play_choice_stats', None)

    def _get_yardage_category(self, yards_to_go):
        """Convertit les yards à franchir en catégorie."""
        if yards_to_go <= 3:
            return '1-3'
        elif yards_to_go <= 6:
            return '4-6'
        elif yards_to_go <= 10:
            return '7-10'
        elif yards_to_go <= 20:
            return '11-20'
        else:
            return '20+'

    def _get_field_position_category(self, position):
        """Convertit une position sur le terrain en catégorie."""
        if position <= 20:
            return '0-20'
        elif position <= 50:
            return '20-50'
        elif position <= 80:
            return '50-80'
        else:
            return '80-100'

    def select_play(self, situation):
        """
        Sélectionne un type de jeu (run/pass) en fonction de la situation et des données réelles.

        Args:
            situation: Dictionnaire avec position, down, yards_to_go

        Returns:
            str: "run" ou "pass"
        """
        down = situation['down']
        yardage_cat = self._get_yardage_category(situation['yards_to_go'])
        field_cat = self._get_field_position_category(situation['position'])

        # Si nous avons des statistiques de choix de jeu
        if self.play_choice_stats is not None:
            # Trouver la ligne correspondante dans les stats
            mask = (self.play_choice_stats['down'] == down) & \
                   (self.play_choice_stats['yardage_category'] == yardage_cat) & \
                   (self.play_choice_stats['field_position_category'] == field_cat)

            # Si on trouve une correspondance exacte
            if mask.sum() > 0:
                row = self.play_choice_stats[mask].iloc[0]
                if 'run' in row and 'pass' in row:
                    if np.random.random() < row['run']:
                        return 'run'
                    else:
                        return 'pass'

        # Si pas de correspondance exacte ou pas de stats, utiliser des valeurs par défaut
        # Stratégie basée sur les tendances générales du football
        if down == 1:
            return 'run' if np.random.random() < 0.65 else 'pass'
        elif down == 2:
            if situation['yards_to_go'] <= 5:
                return 'run' if np.random.random() < 0.7 else 'pass'
            else:
                return 'run' if np.random.random() < 0.4 else 'pass'
        elif down == 3:
            if situation['yards_to_go'] <= 3:
                return 'run' if np.random.random() < 0.7 else 'pass'
            else:
                return 'run' if np.random.random() < 0.20 else 'pass'
        else:  # 4ème down
            if situation['yards_to_go'] <= 1:
                return 'run' if np.random.random() < 0.7 else 'pass'
            else:
                return 'run' if np.random.random() < 0.3 else 'pass'


def simulate_single_drive(play_selector, play_result, field):
    """
    Simule un seul drive de football.

    Args:
        play_selector: Fonction ou instance qui choisit le type de jeu
        play_result: Instance de NFLPlayResult pour simuler les résultats
        field: Instance de FootballField représentant le terrain

    Returns:
        dict: Résultats du drive (points, terminaison, etc.)
    """
    field.reset()
    drive_result = {"plays": [], "outcome": None, "points": 0, "yards_gained": 0}

    while True:
        # Situation actuelle
        situation = {
            "position": field.position,
            "down": field.down,
            "yards_to_go": field.yards_to_go
        }

        # Choisir le type de jeu (run ou pass)
        if isinstance(play_selector, NFLPlaySelector):
            play_type = play_selector.select_play(situation)
        else:
            play_type = play_selector(situation)

        # Simuler le résultat du jeu
        if play_type == "run":
            result, yards = play_result.run_play(**situation)
        else:  # play_type == "pass"
            result, yards = play_result.pass_play(**situation)

        # Enregistrer les données du jeu
        play_data = {
            "position": field.position,
            "down": field.down,
            "yards_to_go": field.yards_to_go,
            "play_type": play_type,
            "result": result,
            "yards_gained": yards
        }
        drive_result["plays"].append(play_data)

        # Mettre à jour le total des yards gagnés
        if result == "SUCCESS":
            drive_result["yards_gained"] += yards

        # Traiter les turnovers
        if result in ["FUMBLE", "INTERCEPTION"]:
            drive_result["outcome"] = result
            break

        # Déplacer le ballon et vérifier la situation
        move_result = field.move(yards if result == "SUCCESS" else 0)

        if move_result == "TOUCHDOWN":
            drive_result["outcome"] = "TOUCHDOWN"
            drive_result["points"] = 7  # Supposons un TD + PAT réussi
            break
        elif move_result == "TURNOVER_ON_DOWNS":
            drive_result["outcome"] = "TURNOVER_ON_DOWNS"
            break

    return drive_result


def random_play_selector(situation):
    """
    Sélectionne aléatoirement un type de jeu sans biais.

    Args:
        situation: Dictionnaire avec position, down, yards_to_go

    Returns:
        str: "run" ou "pass" avec 50% de probabilité chacun
    """
    return "run" if np.random.random() < 0.5 else "pass"


def fixed_strategy_selector(situation):
    """
    Stratégie fixe basée sur des règles simples.

    Args:
        situation: Dictionnaire avec position, down, yards_to_go

    Returns:
        str: "run" ou "pass" selon des règles prédéfinies
    """
    down = situation["down"]
    yards_to_go = situation["yards_to_go"]

    # Logique simple:
    # - Course sur 1er/2ème down avec <4 yards à faire
    # - Course sur 4ème down avec ≤2 yards à faire
    # - Passe dans toutes les autres situations
    if (down in [1, 2] and yards_to_go <= 4) or (down == 4 and yards_to_go <= 2):
        return "run"
    else:
        return "pass"


def run_heavy_selector(situation):
    """Stratégie privilégiant fortement les courses (70% du temps)."""
    return "run" if np.random.random() < 0.7 else "pass"


def pass_heavy_selector(situation):
    """Stratégie privilégiant fortement les passes (70% du temps)."""
    return "run" if np.random.random() < 0.3 else "pass"


def generate_training_data(nfl_params, num_drives=10000, verbose=True):
    """
    Génère des données d'entraînement en simulant de nombreux drives avec différentes stratégies.
    """
    play_result = NFLPlayResult(nfl_params, seed=42)  # Pour reproductibilité
    nfl_selector = NFLPlaySelector(nfl_params)
    field = FootballField()

    all_plays = []
    strategies = [
        (random_play_selector, "random", num_drives // 5),
        (fixed_strategy_selector, "fixed", num_drives // 5),
        (run_heavy_selector, "run_heavy", num_drives // 5),
        (pass_heavy_selector, "pass_heavy", num_drives // 5),
        (nfl_selector, "nfl_based", num_drives // 5)
    ]

    if verbose:
        print(f"Génération de données d'entraînement à partir de {num_drives} drives simulés...")

    for selector, strategy_name, n_drives in strategies:
        if verbose:
            print(f"Utilisation de la stratégie: {strategy_name}")
            drives_iterator = tqdm(range(n_drives))
        else:
            drives_iterator = range(n_drives)

        for _ in drives_iterator:
            drive_result = simulate_single_drive(selector, play_result, field)

            # Ajouter l'information sur la stratégie utilisée
            for play in drive_result["plays"]:
                play["strategy"] = strategy_name
                play["drive_outcome"] = drive_result["outcome"]
                play["drive_points"] = drive_result["points"]
                all_plays.append(play)

    # Convertir en DataFrame
    df = pd.DataFrame(all_plays)

    # Créer une colonne pour l'étiquette d'apprentissage (la valeur du jeu)
    # Utiliser des flottants dès le début pour éviter les problèmes de conversion
    df["play_value"] = np.where(df["drive_points"] > 0, 1.0, 0.0)

    # Ajouter une valeur pour les jeux en fonction du nombre de yards gagnés
    # Système de récompense gradué selon la performance
    df.loc[(df["yards_gained"] >= 20) & (df["play_value"] == 0), "play_value"] = 0.85  # Gain très significatif
    df.loc[(df["yards_gained"] >= 10) & (df["yards_gained"] < 20) & (df["play_value"] == 0), "play_value"] = 0.5  # Gain important
    df.loc[(df["yards_gained"] >= 5) & (df["yards_gained"] < 10) & (df["play_value"] == 0), "play_value"] = 0.25  # Gain modéré
    df.loc[(df["yards_gained"] >= 1) & (df["yards_gained"] < 5) & (df["play_value"] == 0), "play_value"] = 0.05  # Petit gain
    df.loc[(df["yards_gained"] == 0) & (df["play_value"] == 0), "play_value"] = 0.0  # Aucun gain

    # Pénaliser les jeux qui ont conduit à des turnovers (valeur négative)
    df.loc[df["result"].isin(["FUMBLE", "INTERCEPTION"]), "play_value"] = 0.0

    if verbose:
        print(f"Génération terminée: {len(df)} jeux simulés.")

        # Afficher quelques statistiques des données générées
        print("\nDistribution des résultats de drive:")
        print(df["drive_outcome"].value_counts(normalize=True).round(3) * 100)

        print("\nDistribution des types de jeu par down:")
        print(df.groupby(["down", "play_type"]).size().unstack().fillna(0).astype(int))

        print("\nValeur moyenne des jeux par type:")
        print(df.groupby("play_type")["play_value"].mean().round(3))

    return df


def analyze_simulation_data(df):
    """
    Analyse les données de simulation et crée des visualisations.

    Args:
        df: DataFrame contenant les données de simulation
    """

    # 1. Comparer les résultats des drives par stratégie
    plt.figure(figsize=(12, 6))
    drive_outcomes = df.groupby(["strategy", "drive_outcome"]).size().unstack().fillna(0)
    drive_outcomes_pct = drive_outcomes.div(drive_outcomes.sum(axis=1), axis=0) * 100

    drive_outcomes_pct.plot(kind="bar", stacked=True)
    plt.title("Résultats des drives par stratégie")
    plt.xlabel("Stratégie")
    plt.ylabel("Pourcentage")
    plt.legend(title="Résultat")
    plt.savefig(f"{viz_dir}/drive_outcomes_by_strategy_module2.png")
    plt.close()

    # 2. Distribution des yards gagnés par type de jeu
    plt.figure(figsize=(12, 6))
    # Filtrer les valeurs extrêmes pour une meilleure visualisation
    df_filtered = df[df["yards_gained"].between(-10, 30)]

    for play_type, group in df_filtered.groupby("play_type"):
        plt.hist(group["yards_gained"], bins=20, alpha=0.7, density=True, label=play_type)
    plt.title("Distribution des yards gagnés par type de jeu")
    plt.xlabel("Yards gagnés")
    plt.ylabel("Densité")
    plt.legend(title="Type de jeu")
    plt.savefig(f"{viz_dir}/yards_gained_distribution_module2.png")
    plt.close()

    # 3. Yards moyens gagnés par down et type de jeu
    plt.figure(figsize=(10, 6))
    yards_by_down = df.groupby(["down", "play_type"])["yards_gained"].mean().unstack()
    yards_by_down.plot(kind="bar")
    plt.title("Yards moyens gagnés par down et type de jeu")
    plt.xlabel("Down")
    plt.ylabel("Yards moyens gagnés")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{viz_dir}/yards_by_down_and_type_module2.png")
    plt.close()

    # 4. Proportion de courses/passes par down
    plt.figure(figsize=(10, 6))
    play_type_counts = df.groupby(["down", "play_type"]).size().unstack()
    play_type_pct = play_type_counts.div(play_type_counts.sum(axis=1), axis=0) * 100

    play_type_pct.plot(kind="bar", stacked=True)
    plt.title("Proportion de courses/passes par down")
    plt.xlabel("Down")
    plt.ylabel("Pourcentage")
    plt.legend(title="Type de jeu")

    # Ajouter les pourcentages sur les barres
    for i, down in enumerate(play_type_pct.index):
        pct_run = play_type_pct.iloc[i, 0]  # Supposons que 'run' est la première colonne
        plt.text(i, pct_run / 2, f"{pct_run:.1f}%", ha='center', color='white', fontweight='bold')

        pct_pass = play_type_pct.iloc[i, 1]  # Supposons que 'pass' est la deuxième colonne
        plt.text(i, pct_run + pct_pass / 2, f"{pct_pass:.1f}%", ha='center', color='white', fontweight='bold')

    plt.savefig(f"{viz_dir}/play_type_proportion_by_down_module2.png")
    plt.close()

    # 5. Valeur moyenne des jeux par stratégie et type de jeu
    plt.figure(figsize=(12, 6))
    value_by_strategy = df.groupby(["strategy", "play_type"])["play_value"].mean().unstack()
    value_by_strategy.plot(kind="bar")
    plt.title("Valeur moyenne des jeux par stratégie et type de jeu")
    plt.xlabel("Stratégie")
    plt.ylabel("Valeur moyenne")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{viz_dir}/play_value_by_strategy_module2.png")
    plt.close()

    print("Analyse des données de simulation terminée.")


def main():
    """
    Fonction principale pour exécuter la première simulation Monte Carlo.
    """
    # Charger les paramètres extraits des données NFL
    params_file = f"{output_dir}/nfl_simulation_params.pkl"

    if not os.path.exists(params_file):
        print(f"Erreur: Le fichier {params_file} n'existe pas.")
        print("Veuillez d'abord exécuter le module 1 (preparation_donnees.py).")
        return

    with open(params_file, "rb") as f:
        nfl_params = pickle.load(f)

    # Générer les données d'entraînement
    num_drives = 50000  # On simule un grand nombre de drives pour avoir des données suffisantes
    train_data = generate_training_data(nfl_params, num_drives=num_drives)

    # Analyser les données simulées
    analyze_simulation_data(train_data)

    # Sauvegarder les données d'entraînement
    train_data.to_csv(f"{output_dir}/simulation_training_data.csv", index=False)
    print("Données d'entraînement sauvegardées dans simulation_training_data.csv")

    return train_data


if __name__ == "__main__":
    main()