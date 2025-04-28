"""
Module 4 - Seconde Simulation Monte Carlo et Évaluation

Ce module est responsable de:
1. Charger le modèle d'apprentissage machine entraîné
2. Exécuter une seconde simulation Monte Carlo avec le modèle prenant les décisions
3. Comparer les performances avec d'autres stratégies et avec les données réelles de la NFL
4. Analyser les résultats et tirer des conclusions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import joblib
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

viz_dir = "Visuals"
output_dir = "OutputFiles"

# Importer les classes nécessaires du Module 2
from module2_monte_carlo import (
    FootballField,
    NFLPlayResult,
    NFLPlaySelector,
    simulate_single_drive,
    random_play_selector,
    fixed_strategy_selector,
    run_heavy_selector,
    pass_heavy_selector
)


def load_model_and_params():
    """
    Charge le modèle entraîné et les paramètres nécessaires.

    Returns:
        tuple: (modèle, noms des caractéristiques, paramètres NFL, statistiques des drives réels)
    """
    # Charger le modèle
    model_path = f"{output_dir}/football_strategy_model.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Le fichier {model_path} n'existe pas. Veuillez d'abord exécuter le module 3.")

    model = joblib.load(model_path)
    print("Modèle chargé avec succès.")

    # Charger les noms des caractéristiques
    with open(f"{output_dir}/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    # Charger les paramètres NFL
    with open(f"{output_dir}/nfl_simulation_params.pkl", "rb") as f:
        nfl_params = pickle.load(f)

    # Charger les statistiques des drives réels
    with open(f"{output_dir}/nfl_drive_stats.pkl", "rb") as f:
        real_drive_stats = pickle.load(f)

    return model, feature_names, nfl_params, real_drive_stats


def model_play_selector(model, feature_names):
    """
    Crée une fonction de sélection de jeu basée sur le modèle d'apprentissage.

    Args:
        model: Modèle entraîné
        feature_names: Noms des caractéristiques utilisées par le modèle

    Returns:
        function: Fonction de sélection de jeu
    """

    def selector(situation):
        """
        Sélectionne le type de jeu en utilisant le modèle.

        Args:
            situation: Dictionnaire avec position, down, yards_to_go

        Returns:
            str: "run" ou "pass" selon la prédiction du modèle
        """
        # Créer un exemple avec les caractéristiques de base
        features = {
            "position": situation["position"],
            "down": situation["down"],
            "yards_to_go": situation["yards_to_go"],
        }

        # Ajouter les caractéristiques supplémentaires
        features["position_squared"] = features["position"] ** 2

        # Éviter la division par zéro
        position = max(features["position"], 0.001)  # Évite la division par zéro
        features["yards_to_position_ratio"] = features["yards_to_go"] / position

        # Créer un DataFrame pour la prédiction avec les noms de colonnes appropriés
        features_df = pd.DataFrame([features])[feature_names]

        # Normaliser les caractéristiques
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features_df)

        # Créer un DataFrame normalisé pour conserver les noms de colonnes
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

        # Prédire le type de jeu
        prediction = model.predict(X_scaled_df)[0]

        return prediction

    return selector


def compare_strategies(model, feature_names, nfl_params, num_drives=1000, verbose=True):
    """
    Compare différentes stratégies en simulant un grand nombre de drives.

    Args:
        model: Modèle entraîné
        feature_names: Noms des caractéristiques utilisées par le modèle
        nfl_params: Paramètres extraits des données NFL
        num_drives: Nombre de drives à simuler pour chaque stratégie
        verbose: Afficher les barres de progression et les messages

    Returns:
        pd.DataFrame: Comparaison des stratégies
    """
    # Créer les objets pour la simulation
    play_result = NFLPlayResult(nfl_params, seed=0)  # Seed différent de l'entraînement
    nfl_selector = NFLPlaySelector(nfl_params)
    field = FootballField()

    # Définir les stratégies à comparer
    strategies = [
        (random_play_selector, "Aléatoire (50/50)"),
        (fixed_strategy_selector, "Stratégie fixe"),
        (run_heavy_selector, "Orientée course (70%)"),
        (pass_heavy_selector, "Orientée passe (70%)"),
        (nfl_selector, "Basée sur NFL"),
        (model_play_selector(model, feature_names), "Modèle ML")
    ]

    # Stocker les résultats
    results = []

    if verbose:
        print(f"Comparaison de {len(strategies)} stratégies avec {num_drives} drives chacune...")

    for selector, name in strategies:
        if verbose:
            print(f"\nSimulation avec la stratégie: {name}")

        # Initialiser les compteurs
        outcomes = {"TOUCHDOWN": 0, "TURNOVER_ON_DOWNS": 0, "FUMBLE": 0,
                    "INTERCEPTION": 0}
        total_points = 0
        total_yards = 0
        total_plays = 0
        play_types = {"run": 0, "pass": 0}

        # Simuler les drives
        if verbose:
            drives_iterator = tqdm(range(num_drives))
        else:
            drives_iterator = range(num_drives)

        for _ in drives_iterator:
            drive_result = simulate_single_drive(selector, play_result, field)

            # Mettre à jour les statistiques
            outcomes[drive_result["outcome"]] += 1
            total_points += drive_result["points"]
            total_yards += drive_result["yards_gained"]
            total_plays += len(drive_result["plays"])

            # Compter les types de jeux
            for play in drive_result["plays"]:
                play_types[play["play_type"]] += 1

        # Calculer les statistiques
        stats = {
            "strategy": name,
            "points_per_drive": total_points / num_drives,
            "yards_per_drive": total_yards / num_drives,
            "plays_per_drive": total_plays / num_drives,
            "touchdown_rate": outcomes["TOUCHDOWN"] / num_drives,
            "turnover_rate": (outcomes["FUMBLE"] + outcomes[
                "INTERCEPTION"]) / num_drives,
            "turnover_on_downs_rate": outcomes["TURNOVER_ON_DOWNS"] / num_drives,
            "run_percentage": play_types["run"] / (
                        play_types["run"] + play_types["pass"]) * 100,
            "pass_percentage": play_types["pass"] / (
                        play_types["run"] + play_types["pass"]) * 100,
        }

        results.append(stats)

        if verbose:
            print(f"Résultats pour {name}:")
            print(f"  Points par drive: {stats['points_per_drive']:.2f}")
            print(f"  Yards par drive: {stats['yards_per_drive']:.2f}")
            print(f"  Taux de touchdown: {stats['touchdown_rate']:.2%}")
            print(f"  % Courses: {stats['run_percentage']:.1f}%, % Passes: {stats['pass_percentage']:.1f}%")

    # Convertir en DataFrame
    results_df = pd.DataFrame(results)

    # Sauvegarder les résultats
    results_df.to_csv(f"{output_dir}/strategy_comparison.csv", index=False)

    return results_df


def compare_with_real_nfl(comparison_df, real_drive_stats):
    """
    Compare les résultats des simulations avec les statistiques réelles de la NFL.

    Args:
        comparison_df: DataFrame contenant la comparaison des stratégies
        real_drive_stats: Statistiques des drives réels de la NFL
    """
    # Créer une ligne pour les statistiques réelles
    real_stats = {
        "strategy": "NFL Réelle",
        "points_per_drive": real_drive_stats["points_per_drive"],
        "yards_per_drive": real_drive_stats["avg_yards_per_drive"],
        "plays_per_drive": real_drive_stats["avg_plays_per_drive"],
        "touchdown_rate": real_drive_stats["touchdown_rate"],
        # Les autres statistiques ne sont pas disponibles pour les données réelles
        "turnover_rate": np.nan,
        "turnover_on_downs_rate": np.nan,
        "run_percentage": np.nan,
        "pass_percentage": np.nan
    }

    # Ajouter aux résultats des simulations
    comparison_with_real = pd.concat([comparison_df, pd.DataFrame([real_stats])], ignore_index=True)

    return comparison_with_real


def visualize_comparison(comparison_df):
    """
    Visualise les résultats de la comparaison des stratégies.

    Args:
        comparison_df: DataFrame contenant la comparaison des stratégies
    """


    # 1. Points par drive
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x="strategy", y="points_per_drive", data=comparison_df, palette="viridis")
    plt.title("Points moyens par drive", fontsize=14)
    plt.xlabel("Stratégie")
    plt.ylabel("Points par drive")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Ajouter les valeurs sur les barres
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{p.get_height():.2f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{viz_dir}/points_per_drive_comparison_module4.png")
    plt.close()

    # 2. Taux de touchdown
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x="strategy", y="touchdown_rate", data=comparison_df,
                     palette="viridis")
    plt.title("Taux de touchdown par drive", fontsize=14)
    plt.xlabel("Stratégie")
    plt.ylabel("Taux de touchdown")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Ajouter les valeurs sur les barres
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{p.get_height():.2%}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{viz_dir}/touchdown_rate_comparison_module4.png")
    plt.close()

    # 3. Répartition course/passe
    plt.figure(figsize=(12, 6))

    # Filtrer les colonnes pertinentes
    run_pass_df = comparison_df[
        ["strategy", "run_percentage", "pass_percentage"]].copy()

    # Convertir en format long pour le graphique
    run_pass_long = pd.melt(run_pass_df, id_vars=["strategy"],
                            value_vars=["run_percentage", "pass_percentage"],
                            var_name="play_type", value_name="percentage")

    # Remplacer les noms pour l'affichage
    run_pass_long["play_type"] = run_pass_long["play_type"].replace({
        "run_percentage": "Course",
        "pass_percentage": "Passe"
    })

    # Créer le graphique à barres empilées
    ax = sns.barplot(x="strategy", y="percentage", hue="play_type", data=run_pass_long,palette=["green", "blue"])
    plt.title("Répartition courses/passes par stratégie", fontsize=14)
    plt.xlabel("Stratégie")
    plt.ylabel("Pourcentage")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Type de jeu")

    # Ajouter les pourcentages sur les barres
    for i, p in enumerate(ax.patches):
        percentage = p.get_height()
        if not np.isnan(percentage):  # Ignorer les valeurs NaN
            ax.annotate(f"{percentage:.1f}%",
                        (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                        ha='center', va='center', color="white", fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{viz_dir}/run_pass_distribution_module4.png")
    plt.close()

    # 4. Graphique en radar pour comparer les performances globales
    # Sélectionner les métriques pertinentes
    metrics = ["points_per_drive", "yards_per_drive", "touchdown_rate",
               "plays_per_drive"]

    # Normaliser les métriques pour le graphique en radar
    normalized_df = comparison_df.copy()
    for metric in metrics:
        if metric in normalized_df.columns:
            max_val = normalized_df[metric].max()
            if max_val > 0:  # Éviter la division par zéro
                normalized_df[metric] = normalized_df[metric] / max_val

    # Créer le graphique en radar
    plt.figure(figsize=(10, 10))

    # Nombre de variables
    categories = metrics
    N = len(categories)

    # Nombre de stratégies
    strategies = normalized_df["strategy"].tolist()

    # Créer l'angle pour chaque variable
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Fermer le graphique

    # Initialiser le graphique
    ax = plt.subplot(111, polar=True)

    # Dessiner une ligne pour chaque stratégie
    for i, strategy in enumerate(strategies):
        values = normalized_df.loc[
            normalized_df["strategy"] == strategy, metrics].values.flatten().tolist()
        values += values[:1]  # Fermer la ligne

        # Dessiner la ligne
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=strategy)

        # Remplir l'aire
        ax.fill(angles, values, alpha=0.1)

    # Ajouter les étiquettes
    plt.xticks(angles[:-1], categories)

    # Ajouter la légende
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.title("Comparaison des performances par stratégie", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/performance_radar_module4.png")
    plt.close()

    print("Visualisations sauvegardées.")


def analyze_model_performance(comparison_df):
    """
    Analyse et interprète les performances du modèle ML par rapport aux autres stratégies.

    Args:
        comparison_df: DataFrame contenant la comparaison des stratégies
    """
    # Filtrer les lignes pour isoler le modèle ML et les autres stratégies
    model_stats = comparison_df[comparison_df["strategy"] == "Modèle ML"].iloc[0]
    other_strategies = comparison_df[comparison_df["strategy"] != "Modèle ML"].copy()

    # Calculer l'amélioration en pourcentage pour les métriques clés
    improvements = {}
    for metric in ["points_per_drive", "yards_per_drive", "touchdown_rate"]:
        best_other = other_strategies[metric].max()
        improvement_pct = (model_stats[metric] - best_other) / best_other * 100
        improvements[metric] = improvement_pct

    # Identifier la meilleure stratégie alternative pour chaque métrique
    best_alternatives = {}
    for metric in ["points_per_drive", "yards_per_drive", "touchdown_rate"]:
        best_idx = other_strategies[metric].idxmax()
        best_alternatives[metric] = other_strategies.loc[best_idx, "strategy"]

    # Résumer les résultats
    print("\nAnalyse des performances du modèle ML:")
    print(f"Points par drive: {model_stats['points_per_drive']:.2f} ({improvements['points_per_drive']:.1f}% vs. {best_alternatives['points_per_drive']})")
    print(f"Yards par drive: {model_stats['yards_per_drive']:.2f} ({improvements['yards_per_drive']:.1f}% vs. {best_alternatives['yards_per_drive']})")
    print(f"Taux de touchdown: {model_stats['touchdown_rate']:.2%} ({improvements['touchdown_rate']:.1f}% vs. {best_alternatives['touchdown_rate']})")

    # Comparer avec la NFL réelle si disponible
    nfl_real = comparison_df[comparison_df["strategy"] == "NFL Réelle"]
    if not nfl_real.empty:
        nfl_stats = nfl_real.iloc[0]
        print("\nComparaison avec la NFL réelle:")
        for metric in ["points_per_drive", "yards_per_drive", "touchdown_rate",
                       "plays_per_drive"]:
            if not np.isnan(nfl_stats[metric]):
                diff_pct = (model_stats[metric] - nfl_stats[metric]) / nfl_stats[
                    metric] * 100
                print(
                    f"{metric}: {model_stats[metric]:.2f} vs. {nfl_stats[metric]:.2f} ({diff_pct:.1f}%)")

    # Analyser la distribution des jeux
    print(f"\nRépartition courses/passes du modèle ML: {model_stats['run_percentage']:.1f}% / {model_stats['pass_percentage']:.1f}%")

    # Créer un résumé textuel pour le rapport
    summary = f"""
Résumé des performances:
------------------------
Le modèle d'apprentissage machine a obtenu en moyenne {model_stats['points_per_drive']:.2f} points par drive,
ce qui représente une amélioration de {improvements['points_per_drive']:.1f}% par rapport à la meilleure
stratégie alternative ({best_alternatives['points_per_drive']}).

Le taux de touchdown du modèle ({model_stats['touchdown_rate']:.2%}) est 
{'supérieur' if improvements['touchdown_rate'] > 0 else 'inférieur'} de {abs(improvements['touchdown_rate']):.1f}%
à la meilleure stratégie alternative.

En termes de répartition des jeux, le modèle a opté pour {model_stats['run_percentage']:.1f}% de courses
et {model_stats['pass_percentage']:.1f}% de passes, ce qui reflète sa capacité à adapter ses décisions
en fonction des situations de jeu spécifiques.
"""

    # Sauvegarder le résumé
    with open(f"{output_dir}/model_performance_summary.txt", "w") as f:
        f.write(summary)

    print("\nRésumé des performances sauvegardé dans output/model_performance_summary.txt")

    return improvements, best_alternatives


def main():
    """
    Fonction principale pour l'évaluation du modèle par une seconde simulation Monte Carlo.
    """
    # Charger le modèle et les paramètres
    model, feature_names, nfl_params, real_drive_stats = load_model_and_params()

    # Comparer les stratégies
    comparison_df = compare_strategies(model, feature_names, nfl_params, num_drives=5000)

    # Ajouter les données réelles de la NFL pour comparaison
    comparison_with_real = compare_with_real_nfl(comparison_df, real_drive_stats)

    # Visualiser les comparaisons
    visualize_comparison(comparison_with_real)

    # Analyser les performances du modèle
    improvements, best_alternatives = analyze_model_performance(comparison_with_real)



    print("\nÉvaluation complète du modèle terminée!")

    return comparison_with_real

if __name__ == "__main__":
    main()
