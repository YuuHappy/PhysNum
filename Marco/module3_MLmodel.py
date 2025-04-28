"""
Module 3 - Apprentissage Machine

Ce module est responsable de:
1. Charger les données d'entraînement générées par la simulation Monte Carlo
2. Préparer les données pour l'apprentissage machine
3. Entraîner un modèle pour prédire le meilleur type de jeu dans chaque situation
4. Évaluer et analyser les performances du modèle
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import joblib


viz_dir = "Visuals"
output_dir = "OutputFiles"


def load_training_data():
    """
    Charge les données d'entraînement générées par la simulation Monte Carlo.

    Returns:
        pd.DataFrame: Données d'entraînement pour l'apprentissage machine
    """
    file_path = f"{output_dir}/simulation_training_data.csv"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas. Veuillez d'abord exécuter le module 2.")

    df = pd.read_csv(file_path)
    print(f"Données chargées: {len(df)} jeux")

    return df

def prepare_data_for_training(df):
    """
    Prépare les données pour l'entraînement du modèle.

    Args:
        df: DataFrame contenant les données de simulation

    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    # 1. Déterminer quel type de jeu (course ou passe) a la plus grande valeur dans chaque situation
    print("Préparation des données pour l'entraînement...")

    # Agréger par situation et type de jeu pour trouver la valeur moyenne
    situation_cols = ["position", "down", "yards_to_go"]
    value_by_situation = df.groupby(situation_cols + ["play_type"])["play_value"].mean().reset_index()

    # Pivoter pour avoir une colonne pour chaque type de jeu
    pivot_df = value_by_situation.pivot_table(
        index=situation_cols,
        columns="play_type",
        values="play_value"
    ).reset_index()

    # Remplir les valeurs manquantes avec la moyenne globale du type de jeu
    if "run" not in pivot_df.columns:
        pivot_df["run"] = df[df["play_type"] == "run"]["play_value"].mean()
    else:
        pivot_df["run"] = pivot_df["run"].fillna(df[df["play_type"] == "run"]["play_value"].mean())

    if "pass" not in pivot_df.columns:
        pivot_df["pass"] = df[df["play_type"] == "pass"]["play_value"].mean()
    else:
        pivot_df["pass"] = pivot_df["pass"].fillna(df[df["play_type"] == "pass"]["play_value"].mean())

    # Déterminer le meilleur type de jeu pour chaque situation
    pivot_df["best_play"] = np.where(pivot_df["run"] >= pivot_df["pass"], "run", "pass")

    # Créer le jeu de données d'entraînement
    X = pivot_df[situation_cols].copy()
    y = pivot_df["best_play"]

    # Ajouter des caractéristiques supplémentaires
    X.loc[:, "position_squared"] = X["position"] ** 2

    # Éviter la division par zéro
    X.loc[:, "yards_to_position_ratio"] = X["yards_to_go"] / X["position"].replace(0, 0.001)

    # Remplacer les valeurs infinies ou trop grandes par NaN puis par une valeur raisonnable
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())

    # Normaliser les caractéristiques numériques
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Diviser en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print(f"Jeu d'entraînement: {len(X_train)} exemples")
    print(f"Jeu de test: {len(X_test)} exemples")
    print(f"Distribution des classes (train): {y_train.value_counts(normalize=True).round(3) * 100}")

    return X_train, X_test, y_train, y_test, X.columns.tolist()

def train_model(X_train, y_train, feature_names, tune_hyperparameters=True):
    """
    Entraîne un modèle de forêt aléatoire pour prédire le meilleur type de jeu.

    Args:
        X_train: Caractéristiques d'entraînement
        y_train: Étiquettes d'entraînement
        feature_names: Noms des caractéristiques
        tune_hyperparameters: Effectuer une recherche d'hyperparamètres

    Returns:
        object: Modèle entraîné
    """
    print("Entraînement du modèle...")

    if tune_hyperparameters:
        print("Recherche des meilleurs hyperparamètres...")

        # Définir la grille des hyperparamètres à tester
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # Modèle de base
        base_model = RandomForestClassifier(random_state=42)

        # Recherche par validation croisée
        grid_search = GridSearchCV(base_model, param_grid, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        # Afficher les meilleurs paramètres
        print(f"Meilleurs paramètres: {grid_search.best_params_}")

        # Utiliser le meilleur modèle
        model = grid_search.best_estimator_
    else:
        # Utiliser des hyperparamètres par défaut
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        model.fit(X_train, y_train)

    # Calculer l'importance des caractéristiques
    feature_importance = model.feature_importances_

    # Créer un DataFrame pour l'importance des caractéristiques
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)

    print("\nImportance des caractéristiques:")
    print(importance_df)

    return model

def evaluate_model(model, X_test, y_test):
    """
    Évalue les performances du modèle sur l'ensemble de test.

    Args:
        model: Modèle entraîné
        X_test: Caractéristiques de test
        y_test: Étiquettes de test

    Returns:
        dict: Métriques d'évaluation
    """
    print("\nÉvaluation du modèle...")

    # Prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Calculer l'exactitude
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Exactitude: {accuracy:.4f}")

    # Rapport de classification détaillé
    class_report = classification_report(y_test, y_pred)
    print("Rapport de classification:")
    print(class_report)

    # Matrice de confusion
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Visualiser la matrice de confusion
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Course", "Passe"],
                yticklabels=["Course", "Passe"])
    plt.xlabel('Prédit')
    plt.ylabel('Réel')
    plt.title('Matrice de confusion')
    plt.savefig(f"{viz_dir}/confusion_matrix_module3.png")
    plt.close()

    # Regrouper les métriques d'évaluation
    evaluation_metrics = {
        "accuracy": accuracy,
        "classification_report": class_report,
        "confusion_matrix": conf_matrix
    }

    return evaluation_metrics


def analyze_model_decisions(model, feature_names):
    """
    Analyse les décisions du modèle dans différentes situations de jeu.
    """
    print("\nAnalyse des décisions du modèle...")

    # Créer une grille de situations
    positions = np.arange(20, 100, 10)  # Positions de 20 à 90 yards
    downs = [1, 2, 3, 4]  # Les 4 downs
    yards_to_go_values = [1, 3, 5, 8, 10, 15]  # Différentes distances à franchir

    # Au lieu de créer un DataFrame vide et d'y ajouter des lignes,
    # créons d'abord toutes les données puis un DataFrame en une seule fois
    dummy_data = []
    decision_data = []

    # Générer toutes les combinaisons pour le scaler
    for position in positions:
        for down in downs:
            for yards_to_go in yards_to_go_values:
                features = {
                    "position": position,
                    "down": down,
                    "yards_to_go": yards_to_go,
                    "position_squared": position ** 2,
                    "yards_to_position_ratio": yards_to_go / max(position, 0.001)
                }
                dummy_data.append(features)

    # Créer le DataFrame complet d'un coup
    dummy_df = pd.DataFrame(dummy_data)

    # S'assurer que les colonnes sont dans le bon ordre
    dummy_df = dummy_df[feature_names]

    # Créer un scaler pour normaliser les caractéristiques
    scaler = StandardScaler()

    # Entraîner le scaler sur toutes les données fictives
    scaler.fit(dummy_df)

    # Maintenant, générer les prédictions pour chaque situation
    for position in positions:
        for down in downs:
            for yards_to_go in yards_to_go_values:
                # Créer un exemple avec les caractéristiques comme dictionnaire
                features = {
                    "position": position,
                    "down": down,
                    "yards_to_go": yards_to_go,
                    "position_squared": position ** 2,
                    "yards_to_position_ratio": yards_to_go / max(position, 0.001)
                }

                # Convertir en DataFrame avec les bonnes colonnes
                features_df = pd.DataFrame([features])[feature_names]

                # Normaliser avec le scaler pré-entraîné
                X_scaled = scaler.transform(features_df)

                # Créer un DataFrame pour la prédiction avec les bons noms de colonnes
                X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

                # Prédire le type de jeu
                prediction = model.predict(X_scaled_df)[0]

                # Calculer la probabilité
                proba = model.predict_proba(X_scaled_df)[0]
                confidence = proba[0] if prediction == "run" else proba[1]

                # Stocker la décision
                decision_data.append({
                    "position": position,
                    "down": down,
                    "yards_to_go": yards_to_go,
                    "prediction": prediction,
                    "confidence": confidence
                })

    # Convertir en DataFrame
    decision_df = pd.DataFrame(decision_data)

    # Le reste de la fonction reste inchangé
    # Créer une carte de chaleur des décisions pour chaque down
    for down in downs:
        down_df = decision_df[decision_df["down"] == down].copy()

        # Créer un pivot pour la heatmap
        pivot_df = down_df.pivot(
            index="yards_to_go",
            columns="position",
            values="prediction"
        )

        fig, ax = plt.subplots(figsize=(12, 8))
        cmap = plt.cm.RdBu
        heatmap = ax.pcolor(pivot_df.columns, pivot_df.index,
                            np.where(pivot_df == "run", 1, 0), cmap=cmap, vmin=0,
                            vmax=1)

        plt.colorbar(heatmap, ticks=[0, 0.5, 1],
                     label="Type de jeu (0=Passe, 1=Course)")
        plt.title(f"Décisions du modèle pour le {down}ème down")
        plt.xlabel("Position sur le terrain (yards)")
        plt.ylabel("Yards à franchir")

        # Ajouter des annotations
        for i, yards in enumerate(pivot_df.index):
            for j, pos in enumerate(pivot_df.columns):
                play_type = pivot_df.iloc[i, j]
                color = "white" if play_type == "run" else "black"
                confidence = down_df[
                    (down_df["position"] == pos) & (down_df["yards_to_go"] == yards)][
                    "confidence"].values[0]
                text = f"{play_type}\n({confidence:.2f})"
                plt.text(pos, yards, text, ha="center", va="center", color=color)

        plt.grid(True, linestyle="--", alpha=0.7)
        plt.savefig(f"{viz_dir}/decisions_down_{down}_module3.png", bbox_inches="tight")
        plt.close()

    # Sauvegarder les décisions pour une analyse ultérieure
    decision_df.to_csv(f"{output_dir}/model_decisions.csv", index=False)
    print("Analyse des décisions terminée")


def permutation_feature_importance(model, X_test, y_test, feature_names):
    """
    Calcule l'importance des caractéristiques par permutation.

    Args:
        model: Modèle entraîné
        X_test: Caractéristiques de test
        y_test: Étiquettes de test
        feature_names: Noms des caractéristiques
    """
    print("\nCalcul de l'importance des caractéristiques par permutation...")

    # Calculer l'importance par permutation
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

    # Créer un DataFrame pour l'importance des caractéristiques
    perm_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': perm_importance.importances_mean,
        'Std': perm_importance.importances_std
    }).sort_values('Importance', ascending=False)

    print("Importance des caractéristiques par permutation:")
    print(perm_importance_df)

    # Visualiser l'importance des caractéristiques
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=perm_importance_df)
    plt.title('Importance des caractéristiques par permutation')
    plt.xlabel('Importance')
    plt.ylabel('Caractéristique')
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/feature_importance_module3.png")
    plt.close()

    return perm_importance_df

def save_model(model, feature_names):
    """
    Sauvegarde le modèle entraîné et les informations associées.

    Args:
        model: Modèle entraîné
        feature_names: Noms des caractéristiques utilisées par le modèle
    """

    # Sauvegarder le modèle
    joblib.dump(model, f"{output_dir}/football_strategy_model.joblib")

    # Sauvegarder les noms des caractéristiques
    with open(f"{output_dir}/feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)

    print("Modèle et informations associées sauvegardés dans le dossier.")

def main():
    """
    Fonction principale pour l'entraînement et l'évaluation du modèle.
    """
    # Charger les données d'entraînement
    train_data = load_training_data()

    # Préparer les données pour l'entraînement
    X_train, X_test, y_train, y_test, feature_names = prepare_data_for_training(train_data)

    # Entraîner le modèle
    model = train_model(X_train, y_train, feature_names, tune_hyperparameters=True)

    # Évaluer le modèle
    evaluation_metrics = evaluate_model(model, X_test, y_test)

    # Analyser les décisions du modèle
    analyze_model_decisions(model, feature_names)

    # Calculer l'importance des caractéristiques par permutation
    permutation_feature_importance(model, X_test, y_test, feature_names)

    # Sauvegarder le modèle
    save_model(model, feature_names)

    print("Entraînement et évaluation du modèle terminés avec succès!")

    return model, evaluation_metrics

if __name__ == "__main__":
    main()