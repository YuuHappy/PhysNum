"""Module 2 - Simulation Monte Carlo Améliorée

Ce module est responsable de:
1. Utiliser les paramètres extraits des données NFL pour simuler des jeux de football
2. Générer un grand nombre de drives en utilisant des choix stratégiques basés sur les tendances NFL
3. Ajouter des logiques spécifiques pour les situations de field goal et de punts
4. Produire un ensemble de données d'entraînement de haute qualité pour le modèle d'apprentissage machine
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

    def __init__(self, starting_position=20):
        """
        Initialise un nouveau terrain de football.

        Args:
            starting_position: Position de départ par défaut (ligne des 20 yards)
        """
        self.length = 100  # Yards
        self.starting_position = starting_position
        self.reset()

    def reset(self, starting_position=None):
        """
        Réinitialise le terrain pour un nouveau drive.

        Args:
            starting_position: Position de départ personnalisée (si None, utilise la valeur par défaut)
        """
        self.position = starting_position if starting_position is not None else self.starting_position
        self.down = 1
        self.yards_to_go = 10
        self.points = 0
        self.drive_history = []
        self.plays_count = 0

    def move(self, yards_gained):
        """
        Déplace le ballon de yards_gained yards sur le terrain.

        Args:
            yards_gained: Nombre de yards gagnés (négatif si perte)

        Returns:
            str: État du drive ('CONTINUE', 'TOUCHDOWN', 'TURNOVER_ON_DOWNS', 'FIELD_GOAL_RANGE')
        """
        # Enregistrer l'état avant le mouvement
        previous_position = self.position
        previous_down = self.down
        previous_yards_to_go = self.yards_to_go

        # Déplacer le ballon
        self.position += yards_gained

        # Limiter la position à la longueur du terrain
        self.position = min(self.position, self.length)
        self.position = max(self.position, 0)  # Éviter les positions négatives

        # Ajouter à l'historique du drive
        self.drive_history.append({
            'position': previous_position,
            'down': previous_down,
            'yards_to_go': previous_yards_to_go,
            'yards_gained': yards_gained,
            'new_position': self.position
        })

        self.plays_count += 1

        # Vérifier si touchdown
        if self.position >= self.length:
            return "TOUCHDOWN"

        # Vérifier si premier down obtenu
        if yards_gained >= previous_yards_to_go:
            self.down = 1
            self.yards_to_go = 10
            # Ajuster yards_to_go si proche de la end zone
            if self.position + self.yards_to_go > self.length:
                self.yards_to_go = self.length - self.position
        else:
            self.down += 1
            self.yards_to_go = previous_yards_to_go - yards_gained

        # Vérifier si le drive continue
        if self.down > 4:
            # Vérifier si à portée de field goal (40 yards ou moins de la end zone)
            if self.position >= 60:  # À 40 yards ou moins de la end zone
                return "FIELD_GOAL_RANGE"
            else:
                return "TURNOVER_ON_DOWNS"  # 4ème tentative échouée hors de portée de FG

        return "CONTINUE"  # Le jeu continue

    def get_situation(self):
        """
        Retourne la situation actuelle sur le terrain.

        Returns:
            dict: Situation actuelle (position, down, yards_to_go)
        """
        return {
            "position": self.position,
            "down": self.down,
            "yards_to_go": self.yards_to_go
        }


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

        # Paramètres de turnover
        if 'fumble_rate_run' in params:
            self.run_fumble_prob = params['fumble_rate_run']
        else:
            self.run_fumble_prob = params.get('fumble_rate', 0.015)

        if 'fumble_rate_pass' in params:
            self.pass_fumble_prob = params['fumble_rate_pass']
        else:
            self.pass_fumble_prob = params.get('fumble_rate', 0.01)

        self.pass_interception_prob = params['interception_rate']

        # Paramètres de passe
        self.pass_completion_rate = params['completion_rate']

        if 'completion_rate_by_down' in params:
            self.completion_rate_by_down = params['completion_rate_by_down']
        else:
            self.completion_rate_by_down = None

        # Paramètres avancés
        self.down_adjustments = params.get('down_adjustments', {})
        self.field_adjustments = params.get('field_adjustments', {})

        # Statistiques de réussite de premier down
        self.first_down_stats = params.get('first_down_stats', None)

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
        if field_cat in self.field_adjustments and 'run' in self.field_adjustments[
            field_cat]:
            mean_adjustment += self.field_adjustments[field_cat]['run'].get(
                'mean_adjust', 0)
            std_adjustment += self.field_adjustments[field_cat]['run'].get('std_adjust',
                                                                           0)

        # Ajustements spécifiques additionnels
        if down == 3 and yards_to_go <= 2:  # 3ème tentative et courte distance
            mean_adjustment += 0.5  # Bonus en situation de courte distance
        elif down == 4 and yards_to_go <= 2:  # 4ème tentative et courte distance
            mean_adjustment += 0.3  # Bonus léger pour 4ème et courte
        elif down == 4:  # 4ème tentative
            mean_adjustment -= 0.5  # Malus en 4ème tentative (stress)

        if position >= 90:  # Proche de l'en-but adverse
            mean_adjustment -= 1.0  # Plus difficile de gagner des yards
            std_adjustment -= 0.5  # Moins de variabilité

        # Utiliser les statistiques de réussite de premier down si disponibles
        if self.first_down_stats is not None:
            yardage_cat = self._get_yardage_category(yards_to_go)
            fd_stats = self.first_down_stats[
                (self.first_down_stats['play_type'] == 'run') &
                (self.first_down_stats['yardage_category'] == yardage_cat)
                ]

            if not fd_stats.empty:
                fd_prob = fd_stats.iloc[0]['first_down_probability']

                # Si nous avons une bonne probabilité de réussir un premier down,
                # nous ajustons la moyenne pour refléter cette probabilité
                if np.random.random() < fd_prob:
                    return "SUCCESS", yards_to_go + np.random.randint(0,
                                                                      5)  # Réussite du premier down + yards bonus

        # Générer le résultat avec la distribution ajustée
        yards = np.random.normal(
            self.run_mean + mean_adjustment,
            max(0.5, self.run_std + std_adjustment)  # Éviter un écart-type négatif
        )

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

        # Vérifier si fumble après réception
        if np.random.random() < self.pass_fumble_prob:
            return "FUMBLE", 0

        # Déterminer le taux de complétion basé sur le down si disponible
        completion_rate = self.pass_completion_rate
        if self.completion_rate_by_down is not None and down in self.completion_rate_by_down:
            completion_rate = self.completion_rate_by_down[down]

        # Vérifier si la passe est complète
        if np.random.random() > completion_rate:
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
        if field_cat in self.field_adjustments and 'pass' in self.field_adjustments[
            field_cat]:
            mean_adjustment += self.field_adjustments[field_cat]['pass'].get(
                'mean_adjust', 0)
            std_adjustment += self.field_adjustments[field_cat]['pass'].get(
                'std_adjust', 0)

        # Ajustements spécifiques additionnels
        if down == 3 and yards_to_go > 5:  # 3ème tentative et longue distance
            mean_adjustment += 1.0  # Bonus en situation de longue distance
            std_adjustment += 1.0  # Plus de variabilité
        elif yards_to_go <= 5:  # Courte distance
            mean_adjustment -= 1.0  # Malus pour passes courtes
        elif yards_to_go >= 15:  # Très longue distance
            mean_adjustment += 3.0  # Bonus pour passes longues
            std_adjustment += 2.0  # Plus de variabilité

        if down == 4:  # 4ème tentative
            mean_adjustment += 0.5  # Bonus en 4ème tentative (tout ou rien)
            std_adjustment += 1.5  # Plus de variabilité

        if position >= 90:  # Proche de l'en-but adverse
            mean_adjustment -= 1.0  # Plus difficile de gagner des yards
            std_adjustment -= 0.5  # Moins de variabilité

        # Utiliser les statistiques de réussite de premier down si disponibles
        if self.first_down_stats is not None:
            yardage_cat = self._get_yardage_category(yards_to_go)
            fd_stats = self.first_down_stats[
                (self.first_down_stats['play_type'] == 'pass') &
                (self.first_down_stats['yardage_category'] == yardage_cat)
                ]

            if not fd_stats.empty:
                fd_prob = fd_stats.iloc[0]['first_down_probability']

                # Si nous avons une bonne probabilité de réussir un premier down,
                # nous ajustons la moyenne pour refléter cette probabilité
                if np.random.random() < fd_prob:
                    return "SUCCESS", yards_to_go + np.random.randint(0,
                                                                      10)  # Réussite du premier down + yards bonus

        # Générer le résultat avec la distribution ajustée
        yards = np.random.normal(
            self.pass_mean + mean_adjustment,
            max(0.5, self.pass_std + std_adjustment)  # Éviter un écart-type négatif
        )

        # Arrondir à l'entier le plus proche
        return "SUCCESS", int(round(yards))

    def attempt_field_goal(self, position):
        """
        Simule une tentative de field goal basée sur la position sur le terrain.

        Args:
            position: Position sur le terrain (yards)

        Returns:
            tuple: (résultat, points)
        """
        # Distance du field goal = position + 17 yards (10 pour l'end zone + 7 pour le snap)
        fg_distance = 100 - position + 17

        # Probabilité de réussite basée sur la distance
        if fg_distance <= 20:  # Extra point ou très court
            success_prob = 0.99
        elif fg_distance <= 30:  # Courte distance
            success_prob = 0.95
        elif fg_distance <= 40:  # Distance moyenne
            success_prob = 0.85
        elif fg_distance <= 50:  # Longue distance
            success_prob = 0.70
        elif fg_distance <= 55:  # Très longue distance
            success_prob = 0.55
        else:  # Distance extrême
            success_prob = 0.40

        # Simuler la tentative
        if np.random.random() < success_prob:
            return "FIELD_GOAL_SUCCESS", 3
        else:
            return "FIELD_GOAL_MISSED", 0

    def punt(self, position):
        """
        Simule un punt basé sur la position sur le terrain.

        Args:
            position: Position sur le terrain (yards)

        Returns:
            int: Nouvelle position après le punt
        """
        # Distance moyenne du punt: 45 yards avec un écart-type de 5 yards
        punt_distance = max(0, np.random.normal(45, 5))

        # La nouvelle position est limitée par l'end zone
        new_position = max(0, position + punt_distance)
        new_position = min(new_position,
                           99)  # Touchback à la ligne des 20 yards (80 + 20)

        # Si le punt entre dans l'end zone, c'est un touchback (balle placée à la ligne des 20 yards)
        if new_position > 80:
            return 20

        return 100 - new_position  # Convertir en position pour l'autre équipe


class NFLPlaySelector:
    """Classe pour sélectionner le type de jeu en fonction des données réelles."""

    def __init__(self, params):
        """
        Initialise le sélecteur de jeu avec des probabilités basées sur les données réelles.

        Args:
            params: Dictionnaire de paramètres extraits des données réelles
        """
        self.play_choice_stats = params.get('play_choice_stats', None)
        self.random_seed = 42
        np.random.seed(self.random_seed)

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
        Sélectionne un type de jeu (run/pass) en fonction de la situation et des tendances NFL.

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
            # Trouver les lignes correspondantes dans les stats
            mask = (self.play_choice_stats['down'] == down) & \
                   (self.play_choice_stats['yardage_category'] == yardage_cat) & \
                   (self.play_choice_stats['field_position_category'] == field_cat)

            # Si on trouve des correspondances exactes
            matches = self.play_choice_stats[mask]
            if len(matches) > 0:
                # Choisir la ligne avec le plus grand nombre de jeux si plusieurs correspondent
                if len(matches) > 1:
                    row = matches.sort_values('num_plays', ascending=False).iloc[0]
                else:
                    row = matches.iloc[0]

                # Déterminer le type de jeu en fonction des probabilités
                if 'run' in row and not pd.isna(row['run']):
                    run_prob = row['run']
                    if np.random.random() < run_prob:
                        return 'run'
                    else:
                        return 'pass'

            # Si pas de correspondance exacte, chercher une correspondance partielle
            # En priorisant d'abord le down et la position sur le terrain
            partial_mask = (self.play_choice_stats['down'] == down) & \
                           (self.play_choice_stats[
                                'field_position_category'] == field_cat)

            partial_matches = self.play_choice_stats[partial_mask]
            if len(partial_matches) > 0:
                # Agréger les probabilités
                run_prob = partial_matches['run'].mean()
                if np.random.random() < run_prob:
                    return 'run'
                else:
                    return 'pass'

            # Ensuite, essayer juste le down
            down_mask = (self.play_choice_stats['down'] == down)
            down_matches = self.play_choice_stats[down_mask]
            if len(down_matches) > 0:
                run_prob = down_matches['run'].mean()
                if np.random.random() < run_prob:
                    return 'run'
                else:
                    return 'pass'

        # Si aucune statistique n'est disponible ou pas de correspondance, utiliser une logique de base
        # Pour chaque down et situation, établir des probabilités raisonnables
        if down == 1:
            # Premier down: tendance à courir sauf si loin de la ligne à atteindre
            if situation['yards_to_go'] > 10:
                return 'pass' if np.random.random() < 0.7 else 'run'
            else:
                return 'run' if np.random.random() < 0.6 else 'pass'
        elif down == 2:
            # Deuxième down: dépend fortement de la distance
            if situation['yards_to_go'] <= 3:
                return 'run' if np.random.random() < 0.65 else 'pass'
            elif situation['yards_to_go'] <= 6:
                return 'run' if np.random.random() < 0.5 else 'pass'
            else:
                return 'pass' if np.random.random() < 0.7 else 'run'
        elif down == 3:
            # Troisième down: passe à longue distance, course à courte distance
            if situation['yards_to_go'] <= 2:
                return 'run' if np.random.random() < 0.7 else 'pass'
            elif situation['yards_to_go'] <= 5:
                return 'run' if np.random.random() < 0.4 else 'pass'
            else:
                return 'pass' if np.random.random() < 0.9 else 'run'
        else:  # down == 4
            # Quatrième down: presque toujours passe sauf très courte distance
            if situation['yards_to_go'] == 1:
                return 'run' if np.random.random() < 0.8 else 'pass'
            elif situation['yards_to_go'] <= 3:
                return 'run' if np.random.random() < 0.5 else 'pass'
            else:
                return 'pass' if np.random.random() < 0.9 else 'run'

    def decide_fourth_down_action(self, situation):
        """
        Décide de l'action à prendre lors d'un 4ème down.

        Args:
            situation: Dictionnaire avec position, down, yards_to_go

        Returns:
            str: "go_for_it", "field_goal", ou "punt"
        """
        position = situation['position']
        yards_to_go = situation['yards_to_go']

        # Si nous sommes à moins de 40 yards de la end zone, tenter un field goal
        if position >= 60:
            return "field_goal"

        # Si nous sommes près de la ligne du milieu et qu'il ne reste pas beaucoup à faire
        elif position >= 40 and yards_to_go <= 2:
            return "go_for_it"

        # Si nous sommes dans notre propre moitié de terrain et qu'il reste peu à faire
        elif position < 40 and yards_to_go <= 1 and position >= 30:
            # 25% de chance d'y aller pour 4e et 1 près du milieu du terrain
            return "go_for_it" if np.random.random() < 0.25 else "punt"

        # Sinon, punt
        else:
            return "punt"


def simulate_single_drive(play_selector, play_result, field, starting_position=None):
    """
    Simule un seul drive de football.

    Args:
        play_selector: Fonction ou instance qui choisit le type de jeu
        play_result: Instance de NFLPlayResult pour simuler les résultats
        field: Instance de FootballField représentant le terrain
        starting_position: Position de départ optionnelle (si None, utilise la position par défaut)

    Returns:
        dict: Résultats du drive (points, terminaison, etc.)
    """
    field.reset(starting_position)
    drive_result = {
        "plays": [],
        "outcome": None,
        "points": 0,
        "yards_gained": 0,
        "starting_position": field.position
    }

    max_plays = 15  # Limite de sécurité pour éviter les boucles infinies
    play_count = 0

    while play_count < max_plays:
        play_count += 1

        # Situation actuelle
        situation = field.get_situation()

        # Gestion spéciale pour le 4ème down si le sélecteur est une instance de NFLPlaySelector
        if situation['down'] == 4 and isinstance(play_selector, NFLPlaySelector):
            fourth_down_action = play_selector.decide_fourth_down_action(situation)

            if fourth_down_action == "field_goal":
                # Tentative de field goal
                result, points = play_result.attempt_field_goal(situation['position'])
                play_data = {
                    "position": situation['position'],
                    "down": situation['down'],
                    "yards_to_go": situation['yards_to_go'],
                    "play_type": "field_goal",
                    "result": result,
                    "yards_gained": 0,
                    "points": points
                }
                drive_result["plays"].append(play_data)
                drive_result["outcome"] = result
                drive_result["points"] = points
                break

            elif fourth_down_action == "punt":
                # Punt
                new_position = play_result.punt(situation['position'])
                play_data = {
                    "position": situation['position'],
                    "down": situation['down'],
                    "yards_to_go": situation['yards_to_go'],
                    "play_type": "punt",
                    "result": "PUNT",
                    "yards_gained": 0,
                    "new_position": new_position
                }
                drive_result["plays"].append(play_data)
                drive_result["outcome"] = "PUNT"
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
            "position": situation['position'],
            "down": situation['down'],
            "yards_to_go": situation['yards_to_go'],
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
        elif move_result == "FIELD_GOAL_RANGE":
            # Tentative automatique de field goal en 4ème down dans la zone de field goal
            result, points = play_result.attempt_field_goal(field.position)

            # Ajouter la tentative de field goal à l'historique
            play_data = {
                "position": field.position,
                "down": 4,
                "yards_to_go": field.yards_to_go,
                "play_type": "field_goal",
                "result": result,
                "yards_gained": 0,
                "points": points
            }
            drive_result["plays"].append(play_data)

            drive_result["outcome"] = result
            drive_result["points"] = points
            break
        elif move_result == "TURNOVER_ON_DOWNS":
            drive_result["outcome"] = "TURNOVER_ON_DOWNS"
            break

    # Si nous atteignons la limite de jeux, marquer comme incomplet
    if play_count >= max_plays and drive_result["outcome"] is None:
        drive_result["outcome"] = "INCOMPLETE_DRIVE"

    # Calculer quel