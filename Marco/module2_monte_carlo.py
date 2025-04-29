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

    def reset(self, starting_position=20):
        """Réinitialise le terrain pour un nouveau drive à la position spécifiée."""
        self.position = starting_position
        self.down = 1
        self.yards_to_go = 10
        self.points = 0
        self.drive_history = []

    def move(self, yards_gained):
        """
        Déplace le ballon de yards_gained yards sur le terrain avec une logique plus réaliste.
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
            # Ajuster yards_to_go si proche de la end zone
            if self.position + self.yards_to_go > self.length:
                self.yards_to_go = self.length - self.position
        else:
            self.down += 1
            self.yards_to_go -= yards_gained

            # Assurer que yards_to_go n'est jamais négatif
            self.yards_to_go = max(0, self.yards_to_go)

        # Limiter la position à la longueur du terrain
        self.position = min(self.position, self.length)

        # Vérifier si touchdown
        if self.position >= self.length:
            return "TOUCHDOWN"

        # Vérifier si en position de field goal au 4e down
        if self.down == 4 and self.position >= 65:  # À l'intérieur des 35 yards adverses
            return "FIELD_GOAL_RANGE"

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

        # Probabilités de réussite des field goals par distance
        self.field_goal_success_rate = {
            '65-70': 0.55,  # ~55% de réussite de 35-40 yards
            '70-75': 0.65,  # ~65% de réussite de 30-35 yards
            '75-80': 0.75,  # ~75% de réussite de 25-30 yards
            '80-85': 0.85,  # ~85% de réussite de 20-25 yards
            '85-90': 0.92,  # ~92% de réussite de 15-20 yards
            '90-95': 0.95,  # ~95% de réussite de 10-15 yards
            '95-100': 0.98,  # ~98% de réussite de 0-10 yards
        }

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

    def _get_field_goal_category(self, position):
        """Convertit une position sur le terrain en catégorie pour les field goals."""
        # Rendre les catégories plus réalistes
        ranges = [(65, 70), (70, 75), (75, 80), (80, 85), (85, 90), (90, 95), (95, 100)]
        for lower, upper in ranges:
            if lower <= position < upper:
                return f"{lower}-{upper}"
        # Cas par défaut (si position < 65)
        return "65-70"

    def attempt_field_goal(self, position):
        """
        Simule une tentative de field goal avec des probabilités plus réalistes.
        """
        # Obtenir les statistiques réelles de field goal de NFL - à ajuster si nécessaire
        field_goal_success_rate = {
            '65-70': 0.45,  # ~45% de réussite de 35-40 yards (moins qu'avant)
            '70-75': 0.60,  # ~60% de réussite de 30-35 yards
            '75-80': 0.75,  # ~75% de réussite de 25-30 yards
            '80-85': 0.85,  # ~85% de réussite de 20-25 yards
            '85-90': 0.90,  # ~90% de réussite de 15-20 yards
            '90-95': 0.95,  # ~95% de réussite de 10-15 yards
            '95-100': 0.97,  # ~97% de réussite de 0-10 yards
        }

        # Déterminer la catégorie de field goal
        fg_cat = self._get_field_goal_category(position)

        # Calculer la probabilité de réussite
        success_rate = field_goal_success_rate.get(fg_cat, 0.40)

        # Déterminer si le field goal est réussi
        if np.random.random() < success_rate:
            return "FIELD_GOAL_SUCCESS", 3
        else:
            return "FIELD_GOAL_FAILURE", 0

    def run_play(self, position, down, yards_to_go):
        """
        Simule le résultat d'un jeu de course avec des paramètres plus réalistes.
        """
        # Vérifier si fumble (avec le taux réel de NFL)
        if np.random.random() < self.run_fumble_prob:
            return "FUMBLE", 0

        # Ajustements de base - AJOUT D'UN BONUS GLOBAL pour augmenter l'efficacité
        mean_adjustment = 0.5  # Bonus global pour rendre les gains plus réalistes
        std_adjustment = 0.0

        # Ajustements par down
        if down in self.down_adjustments and 'run' in self.down_adjustments[down]:
            mean_adjustment += self.down_adjustments[down]['run']['mean_adjust']
            std_adjustment += self.down_adjustments[down]['run']['std_adjust']

        # Ajustements par position sur le terrain
        field_cat = self._get_field_position_category(position)
        if field_cat in self.field_adjustments and 'run' in self.field_adjustments[
            field_cat]:
            mean_adjustment += self.field_adjustments[field_cat]['run']['mean_adjust']
            std_adjustment += self.field_adjustments[field_cat]['run']['std_adjust']

        # Ajustements spécifiques additionnels - AMÉLIORÉS
        if down == 3 and yards_to_go <= 2:  # 3ème et courte distance
            mean_adjustment += 1.0  # Bonus augmenté en situation de courte distance critique
        elif down == 4 and yards_to_go <= 2:  # 4ème et courte distance
            mean_adjustment += 1.5  # Bonus encore plus grand en 4e down (tout ou rien)
        elif down == 3 and yards_to_go > 5:  # 3ème et longue distance
            mean_adjustment -= 0.5  # Course moins efficace sur longue distance en 3e down

        # Ajustement en fonction de yards_to_go
        if yards_to_go <= 3:
            mean_adjustment += 0.5  # Meilleure performance sur courte distance
        elif yards_to_go >= 10:
            mean_adjustment -= 0.3  # Moins efficace sur longue distance

        if position >= 90:  # Proche de l'en-but adverse
            mean_adjustment -= 0.5  # Plus difficile mais moins qu'avant
            std_adjustment -= 0.2  # Moins de variabilité

        # Générer le résultat avec la distribution ajustée
        yards = np.random.normal(
            self.run_mean + mean_adjustment,
            max(0.5, self.run_std + std_adjustment)  # Éviter un écart-type négatif
        )

        # Ajouter une chance spéciale de gros gain (représente les courses explosives)
        if np.random.random() < 0.05:  # 5% de chance d'une course explosive
            bonus_yards = np.random.uniform(7, 20)
            yards += bonus_yards

        # Arrondir à l'entier le plus proche
        return "SUCCESS", int(round(yards))

    def pass_play(self, position, down, yards_to_go):
        """
        Simule le résultat d'un jeu de passe avec des paramètres plus réalistes.
        """
        # Vérifier si interception (avec le taux réel de NFL)
        if np.random.random() < self.pass_interception_prob:
            return "INTERCEPTION", 0

        # Vérifier si la passe est complète (avec le taux réel de NFL)
        # AJUSTÉ: légère augmentation du taux de complétion pour correspondre aux statistiques réelles
        completion_rate = self.pass_completion_rate * 1.05  # 5% d'augmentation
        completion_rate = min(completion_rate, 0.70)  # Plafonné à 70%

        if np.random.random() > completion_rate:
            return "SUCCESS", 0  # Passe incomplète, 0 yards

        # Ajustements de base - AJOUT D'UN BONUS GLOBAL
        mean_adjustment = 0.7  # Bonus global pour des passes plus efficaces
        std_adjustment = 0.0

        # Ajustements par down
        if down in self.down_adjustments and 'pass' in self.down_adjustments[down]:
            mean_adjustment += self.down_adjustments[down]['pass']['mean_adjust']
            std_adjustment += self.down_adjustments[down]['pass']['std_adjust']

        # Ajustements par position sur le terrain
        field_cat = self._get_field_position_category(position)
        if field_cat in self.field_adjustments and 'pass' in self.field_adjustments[
            field_cat]:
            mean_adjustment += self.field_adjustments[field_cat]['pass']['mean_adjust']
            std_adjustment += self.field_adjustments[field_cat]['pass']['std_adjust']

        # Ajustements spécifiques additionnels - AMÉLIORÉS
        if down == 3 and yards_to_go > 7:  # 3ème tentative et longue distance
            mean_adjustment += 1.5  # Bonus augmenté: les équipes tentent des passes plus longues
            std_adjustment += 1.0  # Plus de variabilité
        elif down == 4:  # 4ème tentative
            mean_adjustment += 2.0  # Bonus augmenté en 4e down (tout ou rien)
            std_adjustment += 1.5  # Plus de variabilité
        elif down == 1 and yards_to_go == 10:  # Premier down standard
            mean_adjustment += 1.0  # Passes plus longues au 1er down

        # Ajustement en fonction de yards_to_go
        if yards_to_go <= 3:
            mean_adjustment -= 0.5  # Passes courtes sur distance courte
        elif yards_to_go >= 15:
            mean_adjustment += 1.0  # Passes plus longues sur très longue distance

        if position >= 90:  # Proche de l'en-but adverse
            mean_adjustment -= 1.0  # Plus difficile en zone rouge
            std_adjustment -= 0.5  # Moins de variabilité

        # Générer le résultat avec la distribution ajustée
        yards = np.random.normal(
            self.pass_mean + mean_adjustment,
            max(0.5, self.pass_std + std_adjustment)  # Éviter un écart-type négatif
        )

        # Ajouter une chance de gros gain (représente les passes explosives)
        if np.random.random() < 0.08:  # 8% de chance d'une passe explosive
            bonus_yards = np.random.uniform(10, 30)
            yards += bonus_yards

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
        # Paramètres de décision pour les field goals
        self.field_goal_decision_threshold = {
            1: 0.05,  # Presque jamais en 1er down
            2: 0.05,  # Presque jamais en 2e down
            3: 0.10,  # Rare en 3e down, mais possible en fin de match
            4: 0.90,  # Très courant en 4e down
        }

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
        Sélectionne un type de jeu avec une logique plus fidèle aux statistiques NFL.
        """
        down = situation['down']
        yards_to_go = situation['yards_to_go']
        position = situation['position']
        yardage_cat = self._get_yardage_category(yards_to_go)
        field_cat = self._get_field_position_category(position)

        # Si nous avons des statistiques de choix de jeu dans les données NFL
        if self.play_choice_stats is not None:
            # Trouver la ligne correspondante dans les stats
            mask = (self.play_choice_stats['down'] == down) & \
                   (self.play_choice_stats['yardage_category'] == yardage_cat) & \
                   (self.play_choice_stats['field_position_category'] == field_cat)

            # Si on trouve une correspondance exacte, utiliser les probabilités réelles
            if mask.sum() > 0:
                row = self.play_choice_stats[mask].iloc[0]
                if 'run' in row and 'pass' in row:
                    if np.random.random() < row['run']:
                        return 'run'
                    else:
                        return 'pass'

        # Valeurs par défaut plus réalistes basées sur les tendances NFL si pas de correspondance exacte
        # 1er down: ~60% de passes en NFL moderne
        if down == 1:
            if yards_to_go == 10:  # 1er et 10 standard
                return 'run' if np.random.random() < 0.40 else 'pass'
            elif yards_to_go <= 3:  # 1er et courte distance
                return 'run' if np.random.random() < 0.60 else 'pass'
            else:  # 1er down avec distance non standard
                return 'run' if np.random.random() < 0.35 else 'pass'

        # 2e down: dépend beaucoup de la distance
        elif down == 2:
            if yards_to_go <= 3:  # 2e et courte
                return 'run' if np.random.random() < 0.55 else 'pass'
            elif yards_to_go <= 7:  # 2e et moyenne distance
                return 'run' if np.random.random() < 0.45 else 'pass'
            else:  # 2e et longue
                return 'run' if np.random.random() < 0.25 else 'pass'

        # 3e down: fortement influencé par la distance
        elif down == 3:
            if yards_to_go <= 2:  # 3e et très courte
                return 'run' if np.random.random() < 0.60 else 'pass'
            elif yards_to_go <= 4:  # 3e et courte
                return 'run' if np.random.random() < 0.35 else 'pass'
            else:  # 3e et moyenne/longue
                return 'run' if np.random.random() < 0.15 else 'pass'

        # 4e down: presque toujours passe sauf très courte distance
        else:
            if yards_to_go <= 1:  # 4e et très courte (inches)
                return 'run' if np.random.random() < 0.70 else 'pass'
            elif yards_to_go <= 3:  # 4e et courte
                return 'run' if np.random.random() < 0.40 else 'pass'
            else:  # 4e et moyenne/longue
                return 'run' if np.random.random() < 0.10 else 'pass'

    def decide_field_goal(self, situation):
        """
        Décide si tenter un field goal en fonction de la situation, avec une logique plus réaliste.
        """
        # Récupérer les données de décision des field goals statistiques réelles
        # En 4e down uniquement sauf situations désespérées
        if situation['down'] < 4:
            # En downs 1-3, presque jamais de field goal sauf end-game scenarios
            return False

        # En 4e down, la décision dépend de la position et de la distance
        if situation['position'] < 60:  # Au-delà des 40 yards adverses
            return False  # Trop loin

        # Entre 40 et 35 yards adverses (position 60-65)
        if situation['position'] < 65:
            return situation['yards_to_go'] > 10  # Seulement si très longue distance

        # Entre 35 et 30 yards adverses (position 65-70)
        if situation['position'] < 70:
            return situation['yards_to_go'] > 7  # Field goal si distance moyenne/longue

        # Entre 30 et 25 yards adverses (position 70-75)
        if situation['position'] < 75:
            return situation[
                'yards_to_go'] > 4  # Field goal sauf si très courte distance

        # À l'intérieur des 25 yards (position 75+)
        return situation[
            'yards_to_go'] > 2  # Presque toujours field goal sauf très courte distance


def simulate_single_drive(play_selector, play_result, field, drive_id,
                          starting_position=20):
    """
    Simule un seul drive avec une logique plus réaliste incluant les punts.
    """
    field.reset(starting_position=starting_position)
    drive_result = {
        "drive_id": drive_id,
        "plays": [],
        "outcome": None,
        "points": 0,
        "yards_gained": 0
    }

    # Ajouter une limite maximum de jeux par drive pour éviter les boucles
    max_plays = 15
    play_count = 0

    while play_count < max_plays:
        play_count += 1

        # Situation actuelle
        situation = {
            "position": field.position,
            "down": field.down,
            "yards_to_go": field.yards_to_go
        }

        # Logique de punt plus réaliste en 4e down
        if situation['down'] == 4:
            # Punt si mauvaise position sur le terrain (avant le milieu de terrain)
            if situation['position'] < 50 and situation['yards_to_go'] > 1:
                # Simuler un punt (on ne le fait pas vraiment, juste terminer le drive)
                play_data = {
                    "position": field.position,
                    "down": field.down,
                    "yards_to_go": field.yards_to_go,
                    "play_type": "punt",
                    "result": "PUNT",
                    "yards_gained": 0,
                    "play_index_in_drive": play_count
                }
                drive_result["plays"].append(play_data)
                drive_result["outcome"] = "PUNT"
                break

        # Décider si tenter un field goal
        attempt_fg = False
        if field.position >= 60:  # Position minimale pour un field goal
            if isinstance(play_selector, NFLPlaySelector):
                attempt_fg = play_selector.decide_field_goal(situation)
            elif field.down == 4 and field.position >= 65:
                # Pour les autres stratégies, décision simplifiée
                attempt_fg = True

        if attempt_fg:
            # Tenter un field goal
            fg_result, points = play_result.attempt_field_goal(field.position)

            # Enregistrer les données du jeu
            play_data = {
                "position": field.position,
                "down": field.down,
                "yards_to_go": field.yards_to_go,
                "play_type": "field_goal",
                "result": fg_result,
                "yards_gained": 0,
                "play_index_in_drive": play_count
            }
            drive_result["plays"].append(play_data)

            # Mettre à jour le résultat du drive
            drive_result["outcome"] = fg_result
            drive_result["points"] = points
            break

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
            "yards_gained": yards,
            "play_index_in_drive": play_count
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
            drive_result["points"] = 7  # TD + PAT réussi
            break
        elif move_result == "FIELD_GOAL_RANGE" and field.down == 4:
            # En 4e down et en position de field goal, tenter un field goal
            fg_result, points = play_result.attempt_field_goal(field.position)

            # Ajouter le résultat du field goal
            play_data = {
                "position": field.position,
                "down": field.down,
                "yards_to_go": field.yards_to_go,
                "play_type": "field_goal",
                "result": fg_result,
                "yards_gained": 0,
                "play_index_in_drive": play_count + 1
            }
            drive_result["plays"].append(play_data)

            drive_result["outcome"] = fg_result
            drive_result["points"] = points
            break
        elif move_result == "TURNOVER_ON_DOWNS":
            drive_result["outcome"] = "TURNOVER_ON_DOWNS"
            break

    # Ajouter le nombre total de jeux dans le drive
    drive_result["play_count"] = play_count

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

    Args:
        nfl_params: Paramètres extraits des données NFL
        num_drives: Nombre total de drives à simuler
        verbose: Afficher des informations pendant la simulation

    Returns:
        pd.DataFrame: Données d'entraînement générées
    """
    play_result = NFLPlayResult(nfl_params, seed=42)  # Pour reproductibilité
    nfl_selector = NFLPlaySelector(nfl_params)
    field = FootballField()

    all_plays = []
    all_drives = []  # Liste pour stocker les résultats de tous les drives

    strategies = [
        (random_play_selector, "random", num_drives // 5),
        (fixed_strategy_selector, "fixed", num_drives // 5),
        (run_heavy_selector, "run_heavy", num_drives // 5),
        (pass_heavy_selector, "pass_heavy", num_drives // 5),
        (nfl_selector, "nfl_based", num_drives // 5)
    ]

    if verbose:
        print(
            f"Génération de données d'entraînement à partir de {num_drives} drives simulés...")

    drive_id = 0  # Compteur pour attribuer un ID unique à chaque drive

    for selector, strategy_name, n_drives in strategies:
        if verbose:
            print(f"Utilisation de la stratégie: {strategy_name}")
            drives_iterator = tqdm(range(n_drives))
        else:
            drives_iterator = range(n_drives)

        for _ in drives_iterator:
            drive_id += 1  # Incrémenter pour chaque nouveau drive

            # Varier la position de départ pour plus de réalisme
            starting_position = np.random.choice(
                [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
                p=[0.35, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                   0.03, 0.02]
            )

            # Simuler le drive
            drive_result = simulate_single_drive(selector, play_result, field,
                                                 starting_position)

            # Ajouter des informations au résultat du drive
            drive_info = {
                "drive_id": drive_id,
                "strategy": strategy_name,
                "outcome": drive_result["outcome"],
                "points": drive_result["points"],
                "yards_gained": drive_result["yards_gained"],
                "num_plays": len(drive_result["plays"]),
                "starting_position": starting_position
            }
            all_drives.append(drive_info)

            # Ajouter l'information sur la stratégie et le drive à chaque jeu
            for play in drive_result["plays"]:
                play["drive_id"] = drive_id
                play["strategy"] = strategy_name
                play["drive_outcome"] = drive_result["outcome"]
                play["drive_points"] = drive_result["points"]
                play["drive_starting_position"] = starting_position
                play["play_index"] = len(all_plays) + 1  # Index unique pour chaque jeu
                all_plays.append(play)

    # Convertir en DataFrame
    plays_df = pd.DataFrame(all_plays)
    drives_df = pd.DataFrame(all_drives)

    # Sauvegarder également les informations des drives pour référence
    drives_df.to_csv(f"{output_dir}/drives_summary.csv", index=False)

    # Créer une colonne pour l'étiquette d'apprentissage (la valeur du jeu)
    plays_df["play_value"] = 0.0

    # Valoriser les jeux qui ont conduit à des points
    plays_df.loc[plays_df["drive_points"] > 0, "play_value"] = 1.0

    # Valoriser les jeux en fonction du gain de yards (pour les jeux sans points)
    plays_df.loc[(plays_df["yards_gained"] >= 20) & (plays_df[
                                                         "play_value"] == 0), "play_value"] = 0.8  # Gain très significatif
    plays_df.loc[(plays_df["yards_gained"] >= 10) & (plays_df["yards_gained"] < 20) & (
                plays_df["play_value"] == 0), "play_value"] = 0.5  # Gain important
    plays_df.loc[(plays_df["yards_gained"] >= 5) & (plays_df["yards_gained"] < 10) & (
                plays_df["play_value"] == 0), "play_value"] = 0.3  # Gain modéré
    plays_df.loc[(plays_df["yards_gained"] >= 1) & (plays_df["yards_gained"] < 5) & (
                plays_df["play_value"] == 0), "play_value"] = 0.1  # Petit gain

    # Pénaliser les jeux qui ont conduit à des turnovers
    plays_df.loc[
        plays_df["result"].isin(["FUMBLE", "INTERCEPTION"]), "play_value"] = 0.0

    # Valoriser les field goals réussis
    plays_df.loc[plays_df["result"] == "FIELD_GOAL_SUCCESS", "play_value"] = 0.7

    # Pénaliser les field goals manqués
    plays_df.loc[plays_df["result"] == "FIELD_GOAL_FAILURE", "play_value"] = 0.0

    if verbose:
        print(
            f"Génération terminée: {len(plays_df)} jeux simulés dans {len(drives_df)} drives.")

        # Afficher quelques statistiques des données générées
        print("\nDistribution des résultats de drive:")
        outcome_counts = drives_df["outcome"].value_counts(normalize=True).round(
            3) * 100
        print(outcome_counts)

        print("\nJeux moyens par drive:")
        plays_per_drive = len(plays_df) / len(drives_df)
        print(f"{plays_per_drive:.2f}")

        print("\nDistribution des types de jeu par down:")
        play_type_by_down = plays_df.groupby(
            ["down", "play_type"]).size().unstack().fillna(0).astype(int)
        print(play_type_by_down)

        print("\nValeur moyenne des jeux par type:")
        play_value_by_type = plays_df.groupby("play_type")["play_value"].mean().round(3)
        print(play_value_by_type)

        print("\nPoints moyens par drive:")
        points_per_drive = drives_df["points"].mean()
        print(f"{points_per_drive:.3f}")

    return plays_df, drives_df  # Retourner à la fois les jeux et les drives


def analyze_simulation_data(plays_df, drives_df):
    """
    Analyse les données de simulation et crée des visualisations.

    Args:
        plays_df: DataFrame contenant les jeux simulés
        drives_df: DataFrame contenant les drives simulés
    """
    # Créer les répertoires de sortie s'ils n'existent pas
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Comparer les résultats des drives par stratégie
    plt.figure(figsize=(12, 7))

    # Ajouter une colonne pour catégoriser les résultats de manière plus lisible
    result_categories = {
        'TOUCHDOWN': 'Touchdown',
        'FIELD_GOAL_SUCCESS': 'Field Goal',
        'FIELD_GOAL_FAILURE': 'Field Goal Manqué',
        'FUMBLE': 'Turnover',
        'INTERCEPTION': 'Turnover',
        'TURNOVER_ON_DOWNS': 'Turnover sur Downs'
    }

    # Ajouter une colonne de catégorie de résultat
    drives_df['result_category'] = drives_df['outcome'].map(
        lambda x: result_categories.get(x, 'Autre'))

    # Compter les résultats par stratégie et catégorie
    drive_counts = drives_df.groupby(
        ['strategy', 'result_category']).size().unstack().fillna(0)
    drive_pct = drive_counts.div(drive_counts.sum(axis=1), axis=0) * 100

    # Créer la visualisation
    ax = drive_pct.plot(kind="bar", stacked=True, colormap="viridis", figsize=(12, 7))
    plt.title("Résultats des drives par stratégie", fontsize=14)
    plt.xlabel("Stratégie", fontsize=12)
    plt.ylabel("Pourcentage", fontsize=12)
    plt.legend(title="Résultat")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.xticks(rotation=45)

    # Ajouter les pourcentages pour les catégories principales
    for i, strategy in enumerate(drive_pct.index):
        touchdown_pct = drive_pct.loc[
            strategy, 'Touchdown'] if 'Touchdown' in drive_pct.columns else 0
        fieldgoal_pct = drive_pct.loc[
            strategy, 'Field Goal'] if 'Field Goal' in drive_pct.columns else 0

        if touchdown_pct > 5:  # Seulement si assez grand pour être visible
            plt.text(i, touchdown_pct / 2, f"{touchdown_pct:.1f}%",
                     ha='center', color='white', fontweight='bold')

        if fieldgoal_pct > 5:
            plt.text(i, touchdown_pct + fieldgoal_pct / 2, f"{fieldgoal_pct:.1f}%",
                     ha='center', color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{viz_dir}/drive_outcomes_by_strategy_module2.png")
    plt.close()

    # 2. Distribution des yards gagnés par type de jeu
    plt.figure(figsize=(12, 6))
    # Filtrer pour n'inclure que les jeux de course et de passe et les valeurs dans une plage raisonnable
    run_pass_df = plays_df[(plays_df["play_type"].isin(["run", "pass"])) &
                           (plays_df["yards_gained"].between(-10, 30))]

    sns.histplot(data=run_pass_df, x="yards_gained", hue="play_type",
                 alpha=0.7, bins=20, kde=True, palette=["green", "blue"])
    plt.title("Distribution des yards gagnés par type de jeu", fontsize=14)
    plt.xlabel("Yards gagnés", fontsize=12)
    plt.ylabel("Fréquence", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(title="Type de jeu")
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/yards_gained_distribution_module2.png")
    plt.close()

    # 3. Yards moyens gagnés par down et type de jeu
    plt.figure(figsize=(12, 7))
    # Filtrer pour n'inclure que les jeux de course et de passe
    run_pass_df = plays_df[plays_df["play_type"].isin(["run", "pass"])]
    yards_by_down = run_pass_df.groupby(["down", "play_type"])[
        "yards_gained"].mean().unstack()
    ax = yards_by_down.plot(kind="bar", color=["green", "blue"])
    plt.title("Yards moyens gagnés par down et type de jeu", fontsize=14)
    plt.xlabel("Down", fontsize=12)
    plt.ylabel("Yards moyens gagnés", fontsize=12)
    plt.grid(True, alpha=0.3)

    # Ajouter les valeurs sur les barres
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{p.get_height():.1f}",
                    (p.get_x() + p.get_width() / 2., p.get_height() + 0.1),
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{viz_dir}/yards_by_down_and_type_module2.png")
    plt.close()

    # 4. Proportion de courses/passes par down
    plt.figure(figsize=(12, 7))
    play_type_counts = run_pass_df.groupby(["down", "play_type"]).size().unstack()
    play_type_pct = play_type_counts.div(play_type_counts.sum(axis=1), axis=0) * 100

    ax = play_type_pct.plot(kind="bar", stacked=True, color=["green", "blue"])
    plt.title("Proportion de courses/passes par down", fontsize=14)
    plt.xlabel("Down", fontsize=12)
    plt.ylabel("Pourcentage", fontsize=12)
    plt.legend(title="Type de jeu")
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # Ajouter les pourcentages sur les barres
    for i, down in enumerate(play_type_pct.index):
        # Pour les courses (en bas)
        pct_run = play_type_pct.iloc[i, 0]  # 'run' est la première colonne
        plt.text(i, pct_run / 2, f"{pct_run:.1f}%", ha='center', color='white',
                 fontweight='bold')

        # Pour les passes (en haut)
        pct_pass = play_type_pct.iloc[i, 1]  # 'pass' est la deuxième colonne
        plt.text(i, pct_run + pct_pass / 2, f"{pct_pass:.1f}%", ha='center',
                 color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{viz_dir}/play_type_proportion_by_down_module2.png")
    plt.close()

    # 5. Valeur moyenne des jeux par stratégie et type de jeu
    plt.figure(figsize=(14, 8))
    value_by_strategy = plays_df.groupby(["strategy", "play_type"])[
        "play_value"].mean().unstack().fillna(0)
    ax = value_by_strategy.plot(kind="bar", cmap="viridis")
    plt.title("Valeur moyenne des jeux par stratégie et type de jeu", fontsize=14)
    plt.xlabel("Stratégie", fontsize=12)
    plt.ylabel("Valeur moyenne", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Type de jeu")

    # Ajouter les valeurs sur les barres
    for i, p in enumerate(ax.patches):
        if p.get_height() > 0.05:  # Seulement si la valeur est significative
            ax.annotate(f"{p.get_height():.2f}",
                        (p.get_x() + p.get_width() / 2., p.get_height() + 0.01),
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{viz_dir}/play_value_by_strategy_module2.png")
    plt.close()

    # 6. Points moyens par drive selon la stratégie
    plt.figure(figsize=(12, 7))
    points_by_strategy = drives_df.groupby('strategy')['points'].mean()

    ax = points_by_strategy.plot(kind='bar', color='steelblue')
    plt.title("Points moyens par drive selon la stratégie", fontsize=14)
    plt.xlabel("Stratégie", fontsize=12)
    plt.ylabel("Points moyens par drive", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Ajouter les valeurs sur les barres
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{p.get_height():.2f}",
                    (p.get_x() + p.get_width() / 2., p.get_height() + 0.05),
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{viz_dir}/points_per_drive_by_strategy_module2.png")
    plt.close()

    # 7. Comparaison des résultats par position de départ
    plt.figure(figsize=(14, 7))
    # Créer des catégories pour la position de départ
    drives_df['starting_position_category'] = pd.cut(
        drives_df['starting_position'],
        bins=[0, 30, 50, 70, 100],
        labels=['0-30', '30-50', '50-70', '70-100']
    )

    # Calculer les points moyens par catégorie de position
    points_by_position = drives_df.groupby('starting_position_category')[
        'points'].mean()

    ax = points_by_position.plot(kind='bar', color='teal')
    plt.title("Points moyens par position de départ", fontsize=14)
    plt.xlabel("Position de départ (yards)", fontsize=12)
    plt.ylabel("Points moyens par drive", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Ajouter les valeurs sur les barres
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{p.get_height():.2f}",
                    (p.get_x() + p.get_width() / 2., p.get_height() + 0.05),
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{viz_dir}/points_by_starting_position_module2.png")
    plt.close()

    # 8. Nombre moyen de jeux par drive selon la stratégie
    plt.figure(figsize=(12, 7))
    plays_per_drive = drives_df.groupby('strategy')['num_plays'].mean()

    ax = plays_per_drive.plot(kind='bar', color='orange')
    plt.title("Nombre moyen de jeux par drive selon la stratégie", fontsize=14)
    plt.xlabel("Stratégie", fontsize=12)
    plt.ylabel("Jeux moyens par drive", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Ajouter les valeurs sur les barres
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{p.get_height():.1f}",
                    (p.get_x() + p.get_width() / 2., p.get_height() + 0.1),
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{viz_dir}/plays_per_drive_by_strategy_module2.png")
    plt.close()

    print("Analyse des données de simulation terminée.")


if __name__ == "__main__":
    import time
    import os

    # Mesurer le temps d'exécution
    start_time = time.time()

    # Créer les répertoires de sortie si nécessaire
    viz_dir = "Visuals"
    output_dir = "OutputFiles"
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("SIMULATION MONTE CARLO DE FOOTBALL NFL")
    print("=" * 80)

    # Vérifier si le module 1 a été exécuté
    params_file = f"{output_dir}/nfl_simulation_params.pkl"
    if not os.path.exists(params_file):
        print(f"ERREUR: Le fichier {params_file} n'existe pas.")
        print("Veuillez d'abord exécuter le module 1 (module1_cleanDataset.py).")
        print("Exécution terminée.")
        exit(1)

    print("Chargement des paramètres NFL depuis le module 1...")
    with open(params_file, "rb") as f:
        nfl_params = pickle.load(f)

    print("Paramètres chargés avec succès!")
    print(
        f"- Yards moyens en course: {nfl_params['run_mean']:.2f} (écart-type: {nfl_params['run_std']:.2f})")
    print(
        f"- Yards moyens en passe: {nfl_params['pass_mean']:.2f} (écart-type: {nfl_params['pass_std']:.2f})")
    print(f"- Taux d'interception: {nfl_params['interception_rate']:.4f}")
    print(f"- Taux de fumble: {nfl_params['fumble_rate']:.4f}")
    print(f"- Taux de complétion de passe: {nfl_params['completion_rate']:.4f}")

    print("\nDémarrage de la simulation Monte Carlo...")

    # Nombre de drives à simuler (ajustez selon vos besoins)
    num_drives = 50000  # Réduit pour des tests plus rapides
    print(f"Nombre de drives à simuler: {num_drives}")

    # Génération des données
    try:
        # La fonction retourne maintenant deux DataFrames: plays_df et drives_df
        plays_df, drives_df = generate_training_data(nfl_params, num_drives=num_drives)

        print(f"Génération des données terminée:")
        print(f"- {len(plays_df)} jeux simulés")
        print(f"- {len(drives_df)} drives simulés")
        print(f"- Moyenne de {len(plays_df) / len(drives_df):.2f} jeux par drive")

        # Analyse des données
        print("\nAnalyse des données et création des visualisations...")
        analyze_simulation_data(plays_df, drives_df)

        # Sauvegarde des données
        plays_output_file = f"{output_dir}/simulation_training_data.csv"
        drives_output_file = f"{output_dir}/drives_summary.csv"

        plays_df.to_csv(plays_output_file, index=False)
        drives_df.to_csv(drives_output_file, index=False)

        print(f"Données sauvegardées dans:")
        print(f"- {plays_output_file}")
        print(f"- {drives_output_file}")

        # Statistiques finales
        print("\n" + "=" * 80)
        print("RÉSUMÉ DE LA SIMULATION")
        print("=" * 80)

        # Statistiques globales
        print(f"Nombre total de jeux simulés: {len(plays_df)}")
        print(f"Nombre total de drives simulés: {len(drives_df)}")
        print(f"Nombre moyen de jeux par drive: {len(plays_df) / len(drives_df):.2f}")

        # Statistiques par type de résultat
        outcome_counts = drives_df["outcome"].value_counts()
        outcome_pcts = drives_df["outcome"].value_counts(normalize=True) * 100

        print("\nRésultats des drives:")
        for outcome, count in outcome_counts.items():
            print(f"- {outcome}: {count} drives ({outcome_pcts[outcome]:.1f}%)")

        # Points moyens par drive
        points_per_drive = drives_df["points"].mean()
        print(f"\nPoints moyens par drive: {points_per_drive:.2f}")

        # Taux de touchdown
        td_rate = (drives_df["outcome"] == "TOUCHDOWN").mean() * 100
        print(f"Taux de touchdown: {td_rate:.2f}%")

        # Taux de field goal réussis
        fg_success_rate = (drives_df["outcome"] == "FIELD_GOAL_SUCCESS").mean() * 100
        print(f"Taux de field goals réussis: {fg_success_rate:.2f}%")

        # Taux de turnover
        turnover_rate = (drives_df["outcome"].isin(
            ["FUMBLE", "INTERCEPTION", "TURNOVER_ON_DOWNS"])).mean() * 100
        print(f"Taux de turnover: {turnover_rate:.2f}%")

        # Répartition des jeux
        play_type_counts = plays_df["play_type"].value_counts()
        play_type_pcts = plays_df["play_type"].value_counts(normalize=True) * 100

        print("\nRépartition des types de jeu:")
        for play_type, count in play_type_counts.items():
            print(f"- {play_type}: {count} jeux ({play_type_pcts[play_type]:.1f}%)")

        # Temps d'exécution
        execution_time = time.time() - start_time
        print(f"\nTemps d'exécution total: {execution_time:.2f} secondes")

        print("\nSimulation Monte Carlo terminée avec succès!")
        print("Les données sont prêtes pour l'apprentissage machine (module 3).")

    except Exception as e:
        print(f"\nERREUR lors de la simulation: {str(e)}")
        import traceback

        traceback.print_exc()
        print("\nLa simulation a échoué. Veuillez vérifier les erreurs ci-dessus.")