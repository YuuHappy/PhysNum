import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
import random

# Constantes du terrain
FIELD_LENGTH = 100  # verges
FIELD_WIDTH = 53.3  # verges
LOS = 20  # ligne de scrimmage (yard line)

# Paramètres de simulation
NUM_MONTE_CARLO = 2000  # Nombre de simulations Monte Carlo par jeu
TIME_STEPS = 50  # Pas de temps pour chaque simulation
MAX_PLAY_TIME = 7.0  # Temps maximum de jeu en secondes


class FootballSimulator:
    def __init__(self):
        self.defensive_formations = self._initialize_defensive_formations()
        self.offensive_plays = self._initialize_offensive_plays()
        self.routes = self._initialize_routes()

    def _initialize_defensive_formations(self):
        """Initialise les formations défensives avec les positions des joueurs"""
        return {
            "Cover 2": {
                "description": "Deux safety profonds, couverture de zone",
                "players": [
                    {"position": "CB1", "x": LOS + 5, "y": 5, "sigma": 4.0},
                    {"position": "CB2", "x": LOS + 5, "y": 48.3, "sigma": 4.0},
                    {"position": "FS", "x": LOS + 25, "y": 17.7, "sigma": 6.0},
                    {"position": "SS", "x": LOS + 25, "y": 35.6, "sigma": 6.0},
                    {"position": "LB1", "x": LOS + 10, "y": 17.7, "sigma": 5.0},
                    {"position": "LB2", "x": LOS + 10, "y": 26.65, "sigma": 5.0},
                    {"position": "LB3", "x": LOS + 10, "y": 35.6, "sigma": 5.0},
                ]
            },
            "Cover 3": {
                "description": "Trois défenseurs profonds, couverture de zone",
                "players": [
                    {"position": "CB1", "x": LOS + 25, "y": 10, "sigma": 6.0},
                    {"position": "FS", "x": LOS + 25, "y": 26.65, "sigma": 6.0},
                    {"position": "CB2", "x": LOS + 25, "y": 43.3, "sigma": 6.0},
                    {"position": "LB1", "x": LOS + 5, "y": 13.3, "sigma": 5.0},
                    {"position": "LB2", "x": LOS + 5, "y": 26.65, "sigma": 5.0},
                    {"position": "LB3", "x": LOS + 5, "y": 40, "sigma": 5.0},
                    {"position": "SS", "x": LOS + 10, "y": 26.65, "sigma": 5.0},
                ]
            },
            "Man-to-Man": {
                "description": "Couverture homme à homme",
                "players": [
                    {"position": "CB1", "x": LOS + 2, "y": 10, "sigma": 3.0,
                     "follows": "WR1"},
                    {"position": "CB2", "x": LOS + 2, "y": 43.3, "sigma": 3.0,
                     "follows": "WR2"},
                    {"position": "FS", "x": LOS + 20, "y": 26.65, "sigma": 5.0},
                    {"position": "SS", "x": LOS + 10, "y": 20, "sigma": 4.0,
                     "follows": "TE"},
                    {"position": "LB1", "x": LOS + 5, "y": 17.7, "sigma": 4.5},
                    {"position": "LB2", "x": LOS + 5, "y": 26.65, "sigma": 4.5},
                    {"position": "LB3", "x": LOS + 5, "y": 35.6, "sigma": 4.5,
                     "follows": "RB"},
                ]
            },
            "Blitz 1": {
                "description": "Un pass rusher supplémentaire",
                "players": [
                    {"position": "CB1", "x": LOS + 5, "y": 5, "sigma": 4.0},
                    {"position": "CB2", "x": LOS + 5, "y": 48.3, "sigma": 4.0},
                    {"position": "FS", "x": LOS + 25, "y": 17.7, "sigma": 6.0},
                    {"position": "SS", "x": LOS + 25, "y": 35.6, "sigma": 6.0},
                    {"position": "LB1", "x": LOS + 5, "y": 17.7, "sigma": 5.0},
                    {"position": "LB2", "x": LOS + 10, "y": 26.65, "sigma": 5.0},
                    # LB3 blitze avec une trajectoire vers l'avant
                    {"position": "LB3", "x": LOS + 2, "y": 35.6, "sigma": 4.0,
                     "vx": -3.0, "vy": 0},
                ]
            },
            "Blitz 2": {
                "description": "Deux pass rushers supplémentaires",
                "players": [
                    {"position": "CB1", "x": LOS + 5, "y": 5, "sigma": 4.0},
                    {"position": "CB2", "x": LOS + 5, "y": 48.3, "sigma": 4.0},
                    {"position": "FS", "x": LOS + 25, "y": 26.65, "sigma": 6.0},
                    # SS blitze
                    {"position": "SS", "x": LOS + 2, "y": 35.6, "sigma": 4.0,
                     "vx": -3.0, "vy": -1.0},
                    {"position": "LB1", "x": LOS + 5, "y": 17.7, "sigma": 5.0},
                    {"position": "LB2", "x": LOS + 10, "y": 26.65, "sigma": 5.0},
                    # LB3 blitze
                    {"position": "LB3", "x": LOS + 2, "y": 35.6, "sigma": 4.0,
                     "vx": -3.0, "vy": 1.0},
                ]
            },
            "Blitz 3": {
                "description": "Trois pass rushers supplémentaires (all-out blitz)",
                "players": [
                    {"position": "CB1", "x": LOS + 5, "y": 5, "sigma": 4.0},
                    {"position": "CB2", "x": LOS + 5, "y": 48.3, "sigma": 4.0},
                    {"position": "FS", "x": LOS + 25, "y": 26.65, "sigma": 6.0},
                    # SS, LB1, LB2 en blitz
                    {"position": "SS", "x": LOS + 2, "y": 35.6, "sigma": 4.0,
                     "vx": -3.0, "vy": -1.0},
                    {"position": "LB1", "x": LOS + 2, "y": 17.7, "sigma": 4.0,
                     "vx": -3.0, "vy": 1.0},
                    {"position": "LB2", "x": LOS + 2, "y": 26.65, "sigma": 4.0,
                     "vx": -3.0, "vy": 0.0},
                    {"position": "LB3", "x": LOS + 2, "y": 35.6, "sigma": 4.0,
                     "vx": -3.0, "vy": 1.0},
                ]
            },
            "4-3 Front": {
                "description": "Quatre defensive line, trois linebackers",
                "players": [
                    {"position": "CB1", "x": LOS + 10, "y": 5, "sigma": 4.0},
                    {"position": "CB2", "x": LOS + 10, "y": 48.3, "sigma": 4.0},
                    {"position": "FS", "x": LOS + 25, "y": 17.7, "sigma": 6.0},
                    {"position": "SS", "x": LOS + 25, "y": 35.6, "sigma": 6.0},
                    {"position": "LB1", "x": LOS + 5, "y": 17.7, "sigma": 5.0},
                    {"position": "LB2", "x": LOS + 5, "y": 26.65, "sigma": 5.0},
                    {"position": "LB3", "x": LOS + 5, "y": 35.6, "sigma": 5.0},
                    # 4 defensive linemen avec sigma plus petit (plus concentré)
                    {"position": "DL1", "x": LOS, "y": 13.3, "sigma": 3.0},
                    {"position": "DL2", "x": LOS, "y": 22.0, "sigma": 3.0},
                    {"position": "DL3", "x": LOS, "y": 31.3, "sigma": 3.0},
                    {"position": "DL4", "x": LOS, "y": 40.0, "sigma": 3.0},
                ]
            },
            "3-4 Front": {
                "description": "Trois defensive line, quatre linebackers",
                "players": [
                    {"position": "CB1", "x": LOS + 10, "y": 5, "sigma": 4.0},
                    {"position": "CB2", "x": LOS + 10, "y": 48.3, "sigma": 4.0},
                    {"position": "FS", "x": LOS + 25, "y": 17.7, "sigma": 6.0},
                    {"position": "SS", "x": LOS + 25, "y": 35.6, "sigma": 6.0},
                    # 4 linebackers au lieu de 3
                    {"position": "LB1", "x": LOS + 5, "y": 13.3, "sigma": 5.0},
                    {"position": "LB2", "x": LOS + 5, "y": 22.0, "sigma": 5.0},
                    {"position": "LB3", "x": LOS + 5, "y": 31.3, "sigma": 5.0},
                    {"position": "LB4", "x": LOS + 5, "y": 40.0, "sigma": 5.0},
                    # 3 defensive linemen au lieu de 4
                    {"position": "DL1", "x": LOS, "y": 17.7, "sigma": 3.0},
                    {"position": "DL2", "x": LOS, "y": 26.65, "sigma": 3.0},
                    {"position": "DL3", "x": LOS, "y": 35.6, "sigma": 3.0},
                ]
            }
        }

    def _initialize_offensive_plays(self):
        """Initialise les jeux offensifs avec les routes des receveurs"""
        return {
            "Pass rapide (slant)": {
                "description": "Passe rapide avec receveur coupant vers l'intérieur",
                "receivers": [
                    {"position": "WR1", "x": LOS, "y": 10, "route": "slant",
                     "primary": True},
                    {"position": "WR2", "x": LOS, "y": 43.3, "route": "flat"},
                    {"position": "TE", "x": LOS, "y": 30, "route": "post"},
                    {"position": "RB", "x": LOS - 5, "y": 26.65, "route": "flat"},
                ],
                "qb": {"x": LOS - 5, "y": 26.65},
                "avg_depth": 5,
                "play_type": "pass",
                "time_to_throw": 1.8
            },
            "Passe longue (go route)": {
                "description": "Passe profonde en ligne droite",
                "receivers": [
                    {"position": "WR1", "x": LOS, "y": 10, "route": "go",
                     "primary": True},
                    {"position": "WR2", "x": LOS, "y": 43.3, "route": "in"},
                    {"position": "TE", "x": LOS, "y": 30, "route": "curl"},
                    {"position": "RB", "x": LOS - 5, "y": 26.65,
                     "route": "block_then_flat"},
                ],
                "qb": {"x": LOS - 5, "y": 26.65},
                "avg_depth": 20,
                "play_type": "pass",
                "time_to_throw": 3.2
            },
            "Course intérieure": {
                "description": "Course entre les tackles",
                "receivers": [
                    {"position": "WR1", "x": LOS, "y": 10, "route": "block"},
                    {"position": "WR2", "x": LOS, "y": 43.3, "route": "block"},
                    {"position": "TE", "x": LOS, "y": 20, "route": "block"},
                    {"position": "RB", "x": LOS - 5, "y": 26.65, "route": "inside_run",
                     "primary": True},
                ],
                "qb": {"x": LOS - 5, "y": 26.65, "handoff": True},
                "avg_depth": 4,
                "play_type": "run",
                "time_to_throw": 0.7
            },
            "Course extérieure": {
                "description": "Course vers l'extérieur",
                "receivers": [
                    {"position": "WR1", "x": LOS, "y": 10, "route": "block"},
                    {"position": "WR2", "x": LOS, "y": 43.3, "route": "block"},
                    {"position": "TE", "x": LOS, "y": 20, "route": "block"},
                    {"position": "RB", "x": LOS - 5, "y": 26.65, "route": "outside_run",
                     "primary": True},
                ],
                "qb": {"x": LOS - 5, "y": 26.65, "handoff": True},
                "avg_depth": 5,
                "play_type": "run",
                "time_to_throw": 0.8
            },
            "Play Action": {
                "description": "Faux handoff puis passe",
                "receivers": [
                    {"position": "WR1", "x": LOS, "y": 10, "route": "post",
                     "primary": True},
                    {"position": "WR2", "x": LOS, "y": 43.3, "route": "in"},
                    {"position": "TE", "x": LOS, "y": 30, "route": "seam"},
                    {"position": "RB", "x": LOS - 5, "y": 26.65,
                     "route": "fake_run_then_flat"},
                ],
                "qb": {"x": LOS - 5, "y": 26.65, "fake_handoff": True},
                "avg_depth": 12,
                "play_type": "pass",
                "time_to_throw": 2.8
            },
            "Read Option": {
                "description": "QB lit défenseur pour décider de la remise ou garde",
                "receivers": [
                    {"position": "WR1", "x": LOS, "y": 10, "route": "block"},
                    {"position": "WR2", "x": LOS, "y": 43.3, "route": "block"},
                    {"position": "TE", "x": LOS, "y": 20, "route": "block"},
                    {"position": "RB", "x": LOS - 5, "y": 30, "route": "option_run",
                     "primary": True},
                ],
                "qb": {"x": LOS - 5, "y": 26.65, "option": True},
                "avg_depth": 6,
                "play_type": "run",
                "time_to_throw": 1.2
            },
            "RPO": {
                "description": "Run-pass option",
                "receivers": [
                    {"position": "WR1", "x": LOS, "y": 10, "route": "slant",
                     "primary": True},
                    {"position": "WR2", "x": LOS, "y": 43.3, "route": "block"},
                    {"position": "TE", "x": LOS, "y": 20, "route": "block"},
                    {"position": "RB", "x": LOS - 5, "y": 26.65, "route": "inside_run"},
                ],
                "qb": {"x": LOS - 5, "y": 26.65, "rpo": True},
                "avg_depth": 7,
                "play_type": "mixed",
                "time_to_throw": 1.5
            },
            "Draw": {
                "description": "Handoff retardé déguisé en passe",
                "receivers": [
                    {"position": "WR1", "x": LOS, "y": 10, "route": "clear_out"},
                    {"position": "WR2", "x": LOS, "y": 43.3, "route": "clear_out"},
                    {"position": "TE", "x": LOS, "y": 20, "route": "delay_block"},
                    {"position": "RB", "x": LOS - 5, "y": 26.65, "route": "draw_run",
                     "primary": True},
                ],
                "qb": {"x": LOS - 5, "y": 26.65, "draw": True},
                "avg_depth": 6,
                "play_type": "run",
                "time_to_throw": 1.7
            },
            "Screen Pass": {
                "description": "Passe derrière la ligne de scrimmage avec bloqueurs",
                "receivers": [
                    {"position": "WR1", "x": LOS, "y": 10, "route": "block_downfield"},
                    {"position": "WR2", "x": LOS, "y": 43.3,
                     "route": "block_downfield"},
                    {"position": "TE", "x": LOS, "y": 20, "route": "block_downfield"},
                    {"position": "RB", "x": LOS - 5, "y": 26.65, "route": "screen",
                     "primary": True},
                ],
                "qb": {"x": LOS - 5, "y": 26.65},
                "avg_depth": 5,
                "play_type": "pass",
                "time_to_throw": 2.0
            }
        }

    def _initialize_routes(self):
        """Définit les fonctions de routes pour différents types de tracés"""
        routes = {}

        # Slant - coupe vers l'intérieur
        def route_slant(x, y, t, params):
            """Route qui coupe en diagonale à travers le terrain"""
            speed = params.get("speed", 5.0)
            cut_time = params.get("cut_time", 1.5)

            if t < cut_time:
                x_new = x + speed * t
                y_new = y
            else:
                straight_dist = speed * cut_time
                cut_dist = speed * (t - cut_time)
                x_new = x + straight_dist + cut_dist * 0.707
                if y < FIELD_WIDTH / 2:
                    y_new = y + cut_dist * 0.707
                else:
                    y_new = y - cut_dist * 0.707

            return x_new, y_new

        routes["slant"] = route_slant

        # Go - ligne droite profonde
        def route_go(x, y, t, params):
            """Route en ligne droite profonde"""
            speed = params.get("speed", 5.0)
            x_new = x + speed * t
            return x_new, y

        routes["go"] = route_go

        # Post - coupe vers le milieu du terrain
        def route_post(x, y, t, params):
            """Route qui coupe vers le milieu du terrain"""
            speed = params.get("speed", 5.0)
            cut_time = params.get("cut_time", 2.0)

            if t < cut_time:
                x_new = x + speed * t
                y_new = y
            else:
                straight_dist = speed * cut_time
                cut_dist = speed * (t - cut_time)
                x_new = x + straight_dist + cut_dist * 0.866
                if y < FIELD_WIDTH / 2:
                    y_new = y + cut_dist * 0.5
                else:
                    y_new = y - cut_dist * 0.5
            return x_new, y_new

        routes["post"] = route_post

        # Out
        def route_out(x, y, t, params):
            """Route qui coupe vers la sideline"""
            speed = params.get("speed", 5.0)
            cut_time = params.get("cut_time", 2.0)

            if t < cut_time:
                x_new = x + speed * t
                y_new = y
            else:
                straight_dist = speed * cut_time
                cut_dist = speed * (t - cut_time)
                x_new = x + straight_dist + cut_dist * 0.866
                if y < FIELD_WIDTH / 2:
                    y_new = y - cut_dist * 0.5
                else:
                    y_new = y + cut_dist * 0.5

            return x_new, y_new

        routes["out"] = route_out

        # Curl
        def route_curl(x, y, t, params):
            """Route où le receveur court puis se retourne vers le QB"""
            speed = params.get("speed", 5.0)
            curl_time = params.get("curl_time", 2.0)

            if t < curl_time:
                x_new = x + speed * t
                y_new = y
            else:
                straight_dist = speed * curl_time
                curl_dist = speed * 0.7 * (t - curl_time)
                x_new = x + straight_dist - curl_dist * 0.3
                y_new = y

            return x_new, y_new

        routes["curl"] = route_curl

        # Flat
        def route_flat(x, y, t, params):
            """Route parallèle à la ligne de scrimmage"""
            speed = params.get("speed", 5.0)
            direction = 1 if y < FIELD_WIDTH / 2 else -1

            x_new = x + speed * 0.3 * t
            y_new = y + direction * speed * t

            return x_new, y_new

        routes["flat"] = route_flat

        def route_in(x, y, t, params):
            """Route in coupant à travers le milieu"""
            speed = params.get("speed", 5.0)
            cut_time = params.get("cut_time", 2.0)

            if t < cut_time:
                x_new = x + speed * t
                y_new = y
            else:
                straight_dist = speed * cut_time
                cut_dist = speed * (t - cut_time)
                x_new = x + straight_dist
                if y < FIELD_WIDTH / 2:
                    y_new = y + cut_dist
                else:
                    y_new = y - cut_dist

            return x_new, y_new

        routes["in"] = route_in

        # Seam
        def route_seam(x, y, t, params):
            """Route verticale dans la couture"""
            speed = params.get("speed", 5.0)
            x_new = x + speed * t
            middle_adjust = (FIELD_WIDTH / 2 - y) * 0.02
            y_new = y + middle_adjust * t

            return x_new, y_new

        routes["seam"] = route_seam

        # Inside run
        def route_inside_run(x, y, t, params):
            """Course intérieure entre les tackles"""
            speed = params.get("speed", 4.5)
            hole_position = params.get("hole_position", 26.65)

            x_new = x + speed * t
            y_diff = hole_position - y
            y_new = y + np.sign(y_diff) * min(abs(y_diff), speed * 0.7 * t)

            return x_new, y_new

        routes["inside_run"] = route_inside_run

        # Outside run
        def route_outside_run(x, y, t, params):
            """Course extérieure vers la sideline"""
            speed = params.get("speed", 4.5)
            direction = params.get("direction", 1 if y < FIELD_WIDTH / 2 else -1)

            x_new = x + speed * 0.8 * t
            y_new = y + direction * speed * 0.6 * t

            return x_new, y_new

        routes["outside_run"] = route_outside_run

        # Bloc - mouvement minimal
        def route_block(x, y, t, params):
            """Assignment de bloc - mouvement minimal"""
            return x + 0.5, y

        routes["block"] = route_block

        # Block then flat
        def route_block_then_flat(x, y, t, params):
            """Bloc puis vers le flat"""
            speed = params.get("speed", 4.0)
            block_time = params.get("block_time", 1.5)
            direction = 1 if y < FIELD_WIDTH / 2 else -1

            if t < block_time:
                x_new = x + 0.2
                y_new = y
            else:
                x_new = x + 0.2 + speed * 0.3 * (t - block_time)
                y_new = y + direction * speed * (t - block_time)

            return x_new, y_new

        routes["block_then_flat"] = route_block_then_flat

        # Block downfield
        def route_block_downfield(x, y, t, params):
            """bloquer sur screen"""
            speed = params.get("speed", 5.0)
            x_new = x + speed * t
            return x_new, y

        routes["block_downfield"] = route_block_downfield

        # Delay block
        def route_delay_block(x, y, t, params):
            """bloc pour draw play"""
            speed = params.get("speed", 4.0)
            delay_time = params.get("delay_time", 1.0)

            if t < delay_time:
                return x, y
            else:
                return x + speed * 0.5 * (t - delay_time), y

        routes["delay_block"] = route_delay_block

        # Screen
        def route_screen(x, y, t, params):
            """screen pour running back"""
            speed = params.get("speed", 4.5)
            delay_time = params.get("delay_time", 1.0)
            direction = params.get("direction", 1 if y < FIELD_WIDTH / 2 else -1)

            if t < delay_time:
                x_new = x
                y_new = y
            else:
                x_new = x + speed * 0.2 * (t - delay_time)
                y_new = y + direction * speed * (t - delay_time)

            return x_new, y_new

        routes["screen"] = route_screen

        # Fake run then flat
        def route_fake_run_then_flat(x, y, t, params):
            speed = params.get("speed", 4.5)
            fake_time = params.get("fake_time", 1.2)
            direction = 1 if y < FIELD_WIDTH / 2 else -1

            if t < fake_time:
                x_new = x + speed * 0.5 * t
                y_new = y
            else:
                x_new = x + speed * 0.5 * fake_time + speed * 0.2 * (t - fake_time)
                y_new = y + direction * speed * (t - fake_time)

            return x_new, y_new

        routes["fake_run_then_flat"] = route_fake_run_then_flat

        # Clear out
        def route_clear_out(x, y, t, params):
            speed = params.get("speed", 5.0)
            x_new = x + speed * t
            direction = 0.1 if y < FIELD_WIDTH / 2 else -0.1
            y_new = y + direction * speed * t

            return x_new, y_new

        routes["clear_out"] = route_clear_out

        # Draw run
        def route_draw_run(x, y, t, params):
            """draw play"""
            speed = params.get("speed", 4.5)
            delay_time = params.get("delay_time", 1.2)

            if t < delay_time:
                x_new = x
                y_new = y
            else:
                x_new = x + speed * 1.2 * (t - delay_time)
                y_new = y

            return x_new, y_new

        routes["draw_run"] = route_draw_run

        # Option run
        def route_option_run(x, y, t, params):
            speed = params.get("speed", 4.5)
            direction = params.get("direction", 1 if y < FIELD_WIDTH / 2 else -1)

            x_new = x + speed * 0.7 * t
            y_new = y + direction * speed * 0.7 * t

            return x_new, y_new

        routes["option_run"] = route_option_run

        return routes

    def gaussian_wave_function(self, x, y, defender_x, defender_y, sigma):
        """Calcule la fonction d'onde gaussienne"""
        distance_squared = (x - defender_x) ** 2 + (y - defender_y) ** 2
        return np.exp(-distance_squared / (2 * sigma ** 2))

    def calculate_coverage_density(self, x, y, defenders, time):
        """Calcule la densité de couverture à un point"""
        total_coverage = 0
        for defender in defenders:
            defender_x = defender.get("x", 0)
            defender_y = defender.get("y", 0)

            vx = defender.get("vx", 0)
            vy = defender.get("vy", 0)
            defender_x += vx * time
            defender_y += vy * time

            sigma = defender.get("sigma", 5.0)
            coverage = self.gaussian_wave_function(x, y, defender_x, defender_y, sigma)
            total_coverage += coverage

        return total_coverage

    def update_player_positions(self, play, defense, time):
        offensive_positions = []
        for receiver in play["receivers"]:
            position = receiver["position"]
            x = receiver["x"]
            y = receiver["y"]
            route_type = receiver["route"]

            route_func = self.routes.get(route_type,self.routes["go"])
            speed_variation = np.random.normal(1.0, 0.1)

            new_x, new_y = route_func(x, y, time, {"speed": 5.0 * speed_variation})

            new_x += np.random.normal(0, 0.2)  # variation en x
            new_y += np.random.normal(0, 0.2)  # variation en y

            offensive_positions.append({
                "position": position,
                "x": new_x,
                "y": new_y,
                "is_primary": receiver.get("primary", False)
            })

        qb_x = play["qb"]["x"]
        qb_y = play["qb"]["y"]
        qb_x += 0.2 * np.sin(time) + np.random.normal(0, 0.1)
        qb_y += np.random.normal(0, 0.1)
        offensive_positions.append({
            "position": "QB",
            "x": qb_x,
            "y": qb_y,
            "is_primary": False
        })

        defensive_positions = []
        for defender in defense["players"]:
            position = defender["position"]
            x = defender["x"]
            y = defender["y"]

            vx = defender.get("vx", 0)
            vy = defender.get("vy", 0)

            speed_variation = np.random.normal(1.0, 0.15)
            vx *= speed_variation
            vy *= speed_variation

            x += vx * time
            y += vy * time

            if "follows" in defender:
                target = defender["follows"]
                for off_player in offensive_positions:
                    if off_player["position"] == target:
                        follow_speed = 0.8 * np.random.normal(1.0,0.15)
                        x = defender["x"] + ( off_player["x"] - defender["x"]) * follow_speed
                        y = defender["y"] + ( off_player["y"] - defender["y"]) * follow_speed
                        break

            x += np.random.normal(0, 0.25)
            y += np.random.normal(0, 0.25)

            defensive_positions.append({
                "position": position,
                "x": x,
                "y": y,
                "sigma": defender.get("sigma", 5.0)
            })

        return offensive_positions, defensive_positions

    def calculate_weighted_distance(self, receiver_x, receiver_y, defenders, time):
        """Calcule la distance pondérée"""

        total_weighted_distance = 0.0
        total_weight = 0.0

        for defender in defenders:
            defender_x = defender.get("x", 0) + defender.get("vx", 0) * time
            defender_y = defender.get("y", 0) + defender.get("vy", 0) * time
            sigma = defender.get("sigma", 5.0)

            # Calcule la fonction d'onde
            wave_function = self.gaussian_wave_function(receiver_x, receiver_y,
                                                        defender_x, defender_y, sigma)

            # Calcule la distance euclidienne
            distance = np.sqrt((receiver_x - defender_x) ** 2 + (receiver_y - defender_y) ** 2)

            total_weighted_distance += wave_function * distance
            total_weight += wave_function

        if total_weight > 0:
            return total_weighted_distance / total_weight
        else:
            return float('inf')

    def run_monte_carlo_simulation(self, offensive_play_name, defensive_formation_name,
                                   num_simulations=NUM_MONTE_CARLO):
        """simulation Monte Carlo pour un jeu offensive contre une défense"""
        play = self.offensive_plays[offensive_play_name]
        defense = self.defensive_formations[defensive_formation_name]

        success_count = 0
        total_gain = 0
        all_distances = []
        all_coverages = []

        time_to_throw = play.get("time_to_throw", 2.5)

        for _ in range(num_simulations):
            throw_time = time_to_throw * np.random.normal(1.0, 0.1)  # Variation de ±10%

            off_positions, def_positions = self.update_player_positions(play, defense, throw_time)

            primary_player = None
            for player in off_positions:
                if player.get("is_primary", False):
                    primary_player = player
                    break

            if not primary_player and play["play_type"] == "pass":
                for player in off_positions:
                    if player["position"].startswith("WR") or player[
                        "position"] == "TE":
                        primary_player = player
                        break

            if not primary_player:
                for player in off_positions:
                    if player["position"] == "RB":
                        primary_player = player
                        break

            if primary_player:
                coverage = self.calculate_coverage_density(
                    primary_player["x"],
                    primary_player["y"],
                    defense["players"],
                    throw_time)

                distance = self.calculate_weighted_distance(
                    primary_player["x"],
                    primary_player["y"],
                    defense["players"],
                    throw_time)

                success_threshold = 0.35 + 0.05 * distance - 0.1 * coverage
                success_prob = np.clip(success_threshold, 0.1, 0.9)

                if play["play_type"] == "pass":
                    base_gain = play["avg_depth"] * (1 - 0.3 * coverage) * np.random.normal(1.0, 0.3)
                else:  # run
                    base_gain = play["avg_depth"] * (1 - 0.4 * coverage) * np.random.normal(1.0, 0.25)

                if np.random.random() < success_prob:
                    success_count += 1
                    gain = base_gain
                else:
                    gain = max(-1,base_gain * 0.2 - 0.5)

                total_gain += gain
                all_distances.append(distance)
                all_coverages.append(coverage)

        # Calcule les statistiques
        success_rate = success_count / num_simulations
        avg_gain = total_gain / num_simulations if num_simulations > 0 else 0
        avg_distance = np.mean(all_distances) if all_distances else 0
        std_distance = np.std(all_distances) if all_distances else 0
        avg_coverage = np.mean(all_coverages) if all_coverages else 0

        return {
            "success_rate": success_rate,
            "avg_gain": avg_gain,
            "avg_distance": avg_distance,
            "std_distance": std_distance,
            "avg_coverage": avg_coverage,
            "all_distances": all_distances
        }

    def run_all_simulations(self):
        results = []

        for off_play_name in tqdm(self.offensive_plays.keys(), desc="Jeux offensifs"):
            for def_formation_name in self.defensive_formations.keys():
                print(f"Simulation {off_play_name} vs {def_formation_name}...")
                sim_result = self.run_monte_carlo_simulation(off_play_name,
                                                             def_formation_name)

                results.append({
                    "Offense": off_play_name,
                    "Defense": def_formation_name,
                    "SuccessRate": sim_result["success_rate"],
                    "AvgGain": sim_result["avg_gain"],
                    "AvgDistance": sim_result["avg_distance"],
                    "StdDistance": sim_result["std_distance"],
                    "AvgCoverage": sim_result["avg_coverage"]
                })

        return pd.DataFrame(results)

    def plot_results(self, results_df):
        pivot_success = results_df.pivot(index="Offense", columns="Defense", values="SuccessRate")
        pivot_gain = results_df.pivot(index="Offense", columns="Defense", values="AvgGain")

        success_display = pivot_success.applymap(lambda x: f"{x * 100:.1f}%")
        gain_display = pivot_gain.applymap(lambda x: f"{x:.1f}")

        # Plot heatmap pour taux de succès
        plt.figure(figsize=(16, 10))
        ax = sns.heatmap(pivot_success, cmap="Blues", annot=success_display.values, fmt="", cbar_kws={'label': 'Taux de succès'})
        plt.title("Taux de succès par combinaison Offense/Défense")
        plt.xlabel("Défense")
        plt.ylabel("Offense")
        plt.tight_layout()
        plt.savefig("success_rate_heatmap.png")
        plt.close()

        # Plot heatmap pour gain moyen
        plt.figure(figsize=(16, 10))
        ax = sns.heatmap(pivot_gain, cmap="Greens", annot=gain_display.values, fmt="", cbar_kws={'label': 'Verges moyennes'})
        plt.title("Gains moyens (verges) par combinaison Offense/Défense")
        plt.xlabel("Défense")
        plt.ylabel("Offense")
        plt.tight_layout()
        plt.savefig("avg_gain_heatmap.png")
        plt.close()

        return success_display, gain_display

    def generate_coverage_map(self, defensive_formation, field_resolution=1.0):
        defense = self.defensive_formations[defensive_formation]
        x_range = np.arange(LOS, LOS + 40, field_resolution)
        y_range = np.arange(0, FIELD_WIDTH, field_resolution)
        coverage_grid = np.zeros((len(y_range), len(x_range)))

        for i, y in enumerate(y_range):
            for j, x in enumerate(x_range):
                coverage_grid[i, j] = self.calculate_coverage_density(x, y, defense["players"], 0)

        plt.figure(figsize=(12, 8))
        plt.imshow(coverage_grid, extent=[LOS, LOS + 40, 0, FIELD_WIDTH], origin='lower', cmap='viridis', aspect='auto')

        for defender in defense["players"]:
            plt.plot(defender["x"], defender["y"], 'ro', markersize=10,
                     label=defender["position"] if "position" not in
                                                   plt.gca().get_legend_handles_labels()[1] else "")

            circle = plt.Circle((defender["x"], defender["y"]), defender["sigma"],
                                fill=False, color='r', linestyle='--', alpha=0.3)
            plt.gca().add_patch(circle)

        plt.colorbar(label='Densité de couverture')
        plt.title(f'Carte de couverture défensive: {defensive_formation}')
        plt.xlabel('Verges (profondeur)')
        plt.ylabel('Verges (largeur)')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.savefig(f"coverage_map_{defensive_formation.replace(' ', '_')}.png")
        plt.close()

        return coverage_grid


def main():
    simulator = FootballSimulator()


    for def_formation in tqdm(simulator.defensive_formations.keys(), desc="couverture"):
        simulator.generate_coverage_map(def_formation)

    # toutes les simulations
    results = simulator.run_all_simulations()

    # visualisations
    success_display, gain_display = simulator.plot_results(results)

    print("\nRésultats (Taux de succès):")
    print(success_display)

    print("\nRésultats (Gain moyen en verges):")
    print(gain_display)

    # Sauvegarde les résultats dans un CSV
    results.to_csv("football_simulation_results.csv", index=False)

    print("\nSimulation terminée")

if __name__ == "__main__":
    main()