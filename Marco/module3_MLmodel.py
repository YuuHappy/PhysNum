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

"""
Améliorations recommandées pour le module 3 (Apprentissage Automatique)
"""


# 1. PRÉPARATION DES DONNÉES
def prepare_data_for_training(plays_df, drives_df):
    """
    Prépare les données d'entraînement de manière plus efficace.

    Args:
        plays_df: DataFrame contenant les détails des jeux simulés
        drives_df: DataFrame contenant les résultats des drives

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    # 1. Créer des caractéristiques (features) plus riches et contextuelles

    # Fusionner les données des jeux avec les résultats des drives
    merged_df = plays_df.copy()

    # Ajouter des caractéristiques dérivées plus sophistiquées
    merged_df['yards_to_go_squared'] = merged_df['yards_to_go'] ** 2
    merged_df['position_squared'] = merged_df['position'] ** 2

    # Distance à la end zone
    merged_df['yards_to_endzone'] = 100 - merged_df['position']

    # Ratio yards à franchir / yards à l'en-but
    merged_df['yards_to_go_ratio'] = merged_df['yards_to_go'] / merged_df[
        'yards_to_endzone'].replace(0, 0.1)

    # Zone du terrain (catégorielle)
    merged_df['field_zone'] = pd.cut(
        merged_df['position'],
        bins=[0, 20, 40, 60, 80, 100],
        labels=['own_red_zone', 'own_zone', 'middle_field', 'opponent_zone', 'red_zone']
    )

    # Convertir la zone du terrain en variables dummy
    field_zone_dummies = pd.get_dummies(merged_df['field_zone'], prefix='zone')
    merged_df = pd.concat([merged_df, field_zone_dummies], axis=1)

    # Catégoriser yards_to_go
    merged_df['distance_category'] = pd.cut(
        merged_df['yards_to_go'],
        bins=[0, 2, 5, 10, 20, 100],
        labels=['very_short', 'short', 'medium', 'long', 'very_long']
    )

    # Convertir la catégorie de distance en variables dummy
    distance_dummies = pd.get_dummies(merged_df['distance_category'], prefix='distance')
    merged_df = pd.concat([merged_df, distance_dummies], axis=1)

    # Interactions entre caractéristiques importantes
    merged_df['down_yards_interaction'] = merged_df['down'] * merged_df['yards_to_go']
    merged_df['position_down_interaction'] = merged_df['position'] * merged_df['down']

    # One-hot encoding du down
    down_dummies = pd.get_dummies(merged_df['down'], prefix='down')
    merged_df = pd.concat([merged_df, down_dummies], axis=1)

    # 2. Créer une étiquette (label) plus nuancée pour l'apprentissage

    # Créer une valeur de jeu plus sophistiquée qui reflète mieux la réalité du football
    merged_df['expected_points_added'] = 0.0

    # Points attendus en fonction de la position sur le terrain
    position_ep_map = {
        (0, 10): -2.0,  # Près de sa propre end zone
        (10, 20): -1.0,  # Dans sa propre red zone
        (20, 40): -0.2,  # Dans sa moitié de terrain
        (40, 50): 0.0,  # Milieu de terrain
        (50, 70): 0.5,  # Dans la moitié adverse
        (70, 90): 1.5,  # Zone de field goal
        (90, 99): 3.0,  # Red zone adverse
        (99, 100): 7.0  # Sur la ligne d'en-but
    }

    # Attribuer les valeurs d'expected points en fonction de la position
    for (lower, upper), value in position_ep_map.items():
        mask = (merged_df['position'] >= lower) & (merged_df['position'] < upper)
        merged_df.loc[mask, 'field_position_value'] = value

    # Valoriser les jeux qui ont mené à des touchdowns ou field goals
    merged_df.loc[merged_df['drive_outcome'] == 'TOUCHDOWN', 'play_value'] = 1.0
    merged_df.loc[
        merged_df['drive_outcome'] == 'FIELD_GOAL_SUCCESS', 'play_value'] = 0.7

    # Valoriser les jeux qui ont obtenu un premier essai
    first_down_plays = (merged_df['yards_gained'] >= merged_df['yards_to_go']) & (
                merged_df['down'] < 4)
    merged_df.loc[
        first_down_plays & (merged_df['play_value'] < 0.5), 'play_value'] = 0.5

    # 3. Sélection et préparation des caractéristiques finales

    # Caractéristiques de base
    features = [
        'down', 'yards_to_go', 'position',
        'yards_to_endzone', 'yards_to_go_ratio',
        'yards_to_go_squared', 'position_squared',
        'down_yards_interaction', 'position_down_interaction'
    ]

    # Ajouter les variables dummy
    for col in merged_df.columns:
        if col.startswith('zone_') or col.startswith('distance_') or col.startswith(
                'down_'):
            features.append(col)

    # Préparer X et y
    X = merged_df[features]
    y = merged_df['play_type'].map(
        {'run': 0, 'pass': 1})  # Classification binaire: 0 pour course, 1 pour passe

    # Normaliser les caractéristiques numériques
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Division en ensembles d'entraînement et de test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Jeu d'entraînement: {len(X_train)} exemples")
    print(f"Jeu de test: {len(X_test)} exemples")
    print(
        f"Distribution des classes (train): {y_train.value_counts(normalize=True).round(3) * 100}")

    return X_train, X_test, y_train, y_test, X.columns.tolist()


# 2. MODÈLE D'APPRENTISSAGE AMÉLIORÉ
def train_model(X_train, y_train, feature_names, tune_hyperparameters=True):
    """
    Entraîne un modèle plus sophistiqué avec une meilleure sélection d'hyperparamètres.
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score
    import numpy as np

    print("Entraînement du modèle...")

    # 1. Utiliser plusieurs modèles candidats
    models = {
        'random_forest': RandomForestClassifier(random_state=42),
        'gradient_boosting': GradientBoostingClassifier(random_state=42)
    }

    best_model = None
    best_score = 0
    best_params = {}
    best_model_name = ""

    if tune_hyperparameters:
        print("Recherche des meilleurs hyperparamètres...")

        # Définir des métriques multiples pour l'évaluation
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f1': make_scorer(f1_score),
            'roc_auc': make_scorer(roc_auc_score)
        }

        # Grilles de paramètres pour chaque modèle
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': [None, 'balanced']
            },
            'gradient_boosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        }

        # Validation croisée stratifiée
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Tester chaque modèle avec sa grille de paramètres
        for model_name, model in models.items():
            print(f"\nOptimisation de {model_name}...")

            grid_search = GridSearchCV(
                model,
                param_grids[model_name],
                cv=cv,
                scoring='f1',  # Utiliser F1 comme métrique principale
                n_jobs=-1,  # Utiliser tous les cœurs disponibles
                verbose=1
            )

            grid_search.fit(X_train, y_train)

            # Comparer avec le meilleur modèle actuel
            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                best_model_name = model_name

        print(f"\nMeilleur modèle: {best_model_name}")
        print(f"Meilleurs paramètres: {best_params}")
        print(f"Meilleur score (F1): {best_score:.4f}")

    else:
        # Si pas d'optimisation, utiliser RandomForest avec paramètres par défaut
        best_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        )
        best_model.fit(X_train, y_train)

    # Calculer l'importance des caractéristiques
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = best_model.feature_importances_

        # Créer un DataFrame pour l'importance des caractéristiques
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)

        print("\nCaractéristiques les plus importantes:")
        print(importance_df.head(10))  # Afficher les 10 plus importantes

    return best_model


# 3. ÉVALUATION DU MODÈLE AMÉLIORÉE
def evaluate_model(model, X_test, y_test, feature_names):
    """
    Évaluation plus complète du modèle avec visualisations améliorées.
    """
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        roc_curve, auc, precision_recall_curve, average_precision_score
    )
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd

    print("\nÉvaluation du modèle...")

    # Prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,
             1]  # Probabilité de la classe positive (passe)

    # 1. Métriques de base
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Exactitude: {accuracy:.4f}")

    # Classification report détaillé
    class_report = classification_report(y_test, y_pred,
                                         target_names=['Course', 'Passe'])
    print("Rapport de classification:")
    print(class_report)

    # 2. Visualisations avancées

    # Matrice de confusion améliorée
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                xticklabels=['Course', 'Passe'],
                yticklabels=['Course', 'Passe'])
    plt.xlabel('Prédit', fontsize=12)
    plt.ylabel('Réel', fontsize=12)
    plt.title('Matrice de confusion (normalisée)', fontsize=14)
    plt.tight_layout()
    plt.savefig("Visuals/confusion_matrix_normalized_module3.png")
    plt.close()

    # Courbe ROC
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'Courbe ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de faux positifs', fontsize=12)
    plt.ylabel('Taux de vrais positifs', fontsize=12)
    plt.title('Courbe ROC (Receiver Operating Characteristic)', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("Visuals/roc_curve_module3.png")
    plt.close()

    # Courbe Précision-Rappel
    plt.figure(figsize=(10, 8))
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)

    plt.plot(recall, precision, color='blue', lw=2,
             label=f'Courbe Précision-Rappel (AP = {avg_precision:.3f})')
    plt.axhline(y=sum(y_test) / len(y_test), color='red', linestyle='--',
                label=f'Ligne de base (Prévalence = {sum(y_test) / len(y_test):.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Rappel', fontsize=12)
    plt.ylabel('Précision', fontsize=12)
    plt.title('Courbe Précision-Rappel', fontsize=14)
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("Visuals/precision_recall_curve_module3.png")
    plt.close()

    # 3. Analyse de l'importance des caractéristiques
    if hasattr(model, 'feature_importances_'):
        # Visualisation de l'importance des caractéristiques
        plt.figure(figsize=(12, 10))
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

        # Prendre les 15 caractéristiques les plus importantes
        top_features = importance_df.head(15)

        sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
        plt.title('15 caractéristiques les plus importantes', fontsize=14)
        plt.xlabel('Importance relative', fontsize=12)
        plt.ylabel('Caractéristique', fontsize=12)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig("Visuals/feature_importance_module3.png")
        plt.close()

    # 4. Analyse des erreurs par situation de jeu

    # Créer un DataFrame pour l'analyse des erreurs
    error_analysis = pd.DataFrame({
        'down': X_test['down'] if 'down' in X_test.columns else X_test['down_1'] * 1 +
                                                                X_test['down_2'] * 2 +
                                                                X_test['down_3'] * 3 +
                                                                X_test['down_4'] * 4,
        'yards_to_go': X_test['yards_to_go'],
        'position': X_test['position'],
        'true_play': y_test,
        'pred_play': y_pred,
        'probability': y_prob,
        'correct': y_test == y_pred
    })

    # Taux d'erreur par down
    plt.figure(figsize=(10, 6))
    error_by_down = error_analysis.groupby('down')['correct'].mean()
    ax = error_by_down.plot(kind='bar', color='green')
    plt.title('Précision du modèle par down', fontsize=14)
    plt.xlabel('Down', fontsize=12)
    plt.ylabel('Précision', fontsize=12)
    plt.ylim([0.5, 1.0])  # Commencer à 0.5 pour mieux voir les différences
    plt.grid(axis='y', alpha=0.3)

    # Ajouter les valeurs
    for i, v in enumerate(error_by_down):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig("Visuals/accuracy_by_down_module3.png")
    plt.close()

    # 5. Distribution des probabilités prédites
    plt.figure(figsize=(10, 6))
    sns.histplot(y_prob, bins=50, kde=True)
    plt.axvline(0.5, color='red', linestyle='--', label='Seuil de décision')
    plt.title('Distribution des probabilités prédites', fontsize=14)
    plt.xlabel('Probabilité prédite de passe', fontsize=12)
    plt.ylabel('Nombre de prédictions', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("Visuals/prediction_distribution_module3.png")
    plt.close()

    # Préparation des métriques à retourner
    evaluation_metrics = {
        "accuracy": accuracy,
        "classification_report": class_report,
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
        "avg_precision": avg_precision
    }

    return evaluation_metrics


# 4. ANALYSE DES DÉCISIONS DU MODÈLE
def analyze_model_decisions(model, feature_names):
    """
    Analyse plus sophistiquée des décisions du modèle dans différentes situations de jeu.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler

    print("\nAnalyse des décisions du modèle...")

    # Créer une grille de situations de jeu pour l'analyse
    downs = [1, 2, 3, 4]
    yards_to_go_values = [1, 2, 3, 5, 7, 10, 15]
    position_values = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    # Préparer les données pour la prédiction
    situation_data = []

    for down in downs:
        for yards_to_go in yards_to_go_values:
            for position in position_values:
                # Créer une situation de jeu
                situation = {
                    'down': down,
                    'yards_to_go': yards_to_go,
                    'position': position,
                    'yards_to_endzone': 100 - position,
                    'yards_to_go_ratio': yards_to_go / (
                                100 - position) if position < 100 else 0,
                    'yards_to_go_squared': yards_to_go ** 2,
                    'position_squared': position ** 2,
                    'down_yards_interaction': down * yards_to_go,
                    'position_down_interaction': position * down
                }

                # Ajouter les variables dummy pour down
                for d in downs:
                    situation[f'down_{d}'] = 1 if down == d else 0

                # Calculer les zones de terrain
                zones = ['own_red_zone', 'own_zone', 'middle_field', 'opponent_zone',
                         'red_zone']
                position_bins = [0, 20, 40, 60, 80, 100]
                zone_idx = 0
                for i in range(len(position_bins) - 1):
                    if position_bins[i] <= position < position_bins[i + 1]:
                        zone_idx = i
                        break

                for i, zone in enumerate(zones):
                    situation[f'zone_{zone}'] = 1 if i == zone_idx else 0

                # Calculer les catégories de distance
                distance_cats = ['very_short', 'short', 'medium', 'long', 'very_long']
                distance_bins = [0, 2, 5, 10, 20, 100]
                distance_idx = 0
                for i in range(len(distance_bins) - 1):
                    if distance_bins[i] <= yards_to_go < distance_bins[i + 1]:
                        distance_idx = i
                        break

                for i, cat in enumerate(distance_cats):
                    situation[f'distance_{cat}'] = 1 if i == distance_idx else 0

                situation_data.append(situation)

    # Convertir en DataFrame
    situations_df = pd.DataFrame(situation_data)

    # Assurer que toutes les colonnes nécessaires sont présentes
    for feature in feature_names:
        if feature not in situations_df.columns:
            situations_df[feature] = 0

    # Ne conserver que les colonnes utilisées par le modèle
    X_situations = situations_df[feature_names]

    # Normaliser les données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_situations)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

    # Prédire le type de jeu pour chaque situation
    run_pass_probs = model.predict_proba(X_scaled_df)[:, 1]  # Probabilité de passe
    situations_df['pass_probability'] = run_pass_probs
    situations_df['predicted_play'] = ['pass' if p > 0.5 else 'run' for p in
                                       run_pass_probs]

    # Créer des heatmaps pour visualiser les décisions par down
    for down in downs:
        plt.figure(figsize=(14, 10))
        down_data = situations_df[situations_df['down'] == down].copy()

        # Créer un pivot pour la heatmap
        heatmap_data = down_data.pivot_table(
            index='yards_to_go',
            columns='position',
            values='pass_probability'
        )

        # Créer la heatmap
        ax = sns.heatmap(
            heatmap_data,
            cmap='RdBu_r',  # Rouge pour courses, bleu pour passes
            annot=True,
            fmt='.2f',
            linewidths=0.5,
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Probabilité de passe'}
        )

        plt.title(f'Probabilité de passe par situation - {down}e down', fontsize=16)
        plt.xlabel('Position sur le terrain (yards)', fontsize=12)
        plt.ylabel('Yards à franchir', fontsize=12)
        plt.tight_layout()

        # Ajouter une ligne pour la décision (0.5)
        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                if not np.isnan(heatmap_data.iloc[i, j]):
                    if heatmap_data.iloc[i, j] > 0.5:
                        plt.text(j + 0.5, i + 0.5, 'P',
                                 ha='center', va='center', color='white',
                                 fontweight='bold')
                    else:
                        plt.text(j + 0.5, i + 0.5, 'C',
                                 ha='center', va='center', color='black',
                                 fontweight='bold')

        plt.savefig(f"Visuals/decision_map_down_{down}_module3.png")
        plt.close()

    # Analyse de sensibilité pour les variables clés
    key_features = ['down', 'yards_to_go', 'position']

    for feature in key_features:
        plt.figure(figsize=(12, 8))

        # Regrouper par la caractéristique et calculer la probabilité moyenne de passe
        feature_effect = situations_df.groupby(feature)['pass_probability'].mean()

        # Tracer la courbe
        ax = feature_effect.plot(marker='o', linestyle='-', linewidth=2)
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7,
                    label='Seuil de décision')

        plt.title(f'Effet de {feature} sur la probabilité de passe', fontsize=14)
        plt.xlabel(feature, fontsize=12)
        plt.ylabel('Probabilité moyenne de passe', fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend()

        # Ajouter les valeurs
        for i, v in enumerate(feature_effect):
            ax.text(feature_effect.index[i], v + 0.01, f'{v:.2f}', ha='center')

        plt.tight_layout()
        plt.savefig(f"Visuals/sensitivity_{feature}_module3.png")
        plt.close()

    return situations_df


# 5. ANALYSE STRATÉGIQUE AVANCÉE
def strategic_analysis(model, feature_names, plays_df, drives_df):
    """
    Analyse plus approfondie des stratégies optimales générées par le modèle.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler

    print("\nAnalyse stratégique avancée...")

    # Comparer les stratégies du modèle avec celles observées dans les données

    # 1. Créer un DataFrame avec les prédictions du modèle sur les données réelles
    # Sélectionner les caractéristiques utilisées par le modèle
    X_real = plays_df[feature_names].copy() if all(
        f in plays_df.columns for f in feature_names) else None

    if X_real is not None:
        # Normaliser
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_real)

        # Prédire
        y_pred_proba = model.predict_proba(X_scaled)[:, 1]
        plays_df['model_pass_probability'] = y_pred_proba
        plays_df['model_play_choice'] = np.where(y_pred_proba >= 0.5, 'pass', 'run')

        # 2. Comparaison des choix par down et distance
        # Regrouper par down et distance
        strategy_comparison = plays_df.groupby(['down', 'distance_category']).agg({
            'play_type': lambda x: (x == 'pass').mean(),  # Pourcentage réel de passes
            'model_play_choice': lambda x: (x == 'pass').mean()
            # Pourcentage prédit de passes
        }).reset_index()

        strategy_comparison.columns = ['down', 'distance_category', 'actual_pass_pct',
                                       'model_pass_pct']
        strategy_comparison['difference'] = strategy_comparison['model_pass_pct'] - \
                                            strategy_comparison['actual_pass_pct']

        # Visualisation des différences
        plt.figure(figsize=(14, 10))

        for down in sorted(strategy_comparison['down'].unique()):
            down_data = strategy_comparison[strategy_comparison['down'] == down]

            plt.subplot(2, 2, down)

            # Tracer les barres groupées
            x = np.arange(len(down_data))
            width = 0.35

            plt.bar(x - width / 2, down_data['actual_pass_pct'] * 100,
                    width, label='Réel', color='blue', alpha=0.7)
            plt.bar(x + width / 2, down_data['model_pass_pct'] * 100,
                    width, label='Modèle', color='green', alpha=0.7)

            plt.xlabel('Catégorie de distance', fontsize=10)
            plt.ylabel('Pourcentage de passes (%)', fontsize=10)
            plt.title(f'Down {down}', fontsize=12)
            plt.xticks(x, down_data['distance_category'], rotation=45)
            plt.ylim(0, 100)
            plt.grid(alpha=0.3)

            if down == 1:
                plt.legend()

        plt.suptitle('Comparaison des stratégies réelles vs modèle', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig("Visuals/strategy_comparison_module3.png")
        plt.close()

        # 3. Analyse d'efficacité des stratégies
        # Combiner les choix du modèle avec les résultats des jeux

        # Calculer l'efficacité moyenne par type de jeu et situation
        efficiency_analysis = plays_df.groupby(
            ['down', 'distance_category', 'play_type']).agg({
            'yards_gained': 'mean',
            'play_value': 'mean'
        }).reset_index()

        # Identifier les meilleures stratégies par situation
        best_strategies = efficiency_analysis.loc[
            efficiency_analysis.groupby(['down', 'distance_category'])[
                'play_value'].idxmax()
        ].copy()

        best_strategies.columns = ['down', 'distance_category', 'optimal_play',
                                   'avg_yards', 'avg_value']

        # Fusionner avec les prédictions du modèle
        strategy_comparison = strategy_comparison.merge(
            best_strategies[['down', 'distance_category', 'optimal_play']],
            on=['down', 'distance_category'],
            how='left'
        )

        # Calculer si le modèle choisit la stratégie optimale
        strategy_comparison['model_optimal'] = np.where(
            (strategy_comparison['model_pass_pct'] >= 0.5) & (
                        strategy_comparison['optimal_play'] == 'pass') |
            (strategy_comparison['model_pass_pct'] < 0.5) & (
                        strategy_comparison['optimal_play'] == 'run'),
            True, False
        )

        # Visualiser l'alignement du modèle avec les stratégies optimales
        plt.figure(figsize=(12, 8))

        sns.heatmap(
            strategy_comparison.pivot(index='down', columns='distance_category',
                                      values='model_optimal'),
            cmap=['red', 'green'],
            annot=True,
            fmt='',
            cbar=False
        )

        plt.title('Alignement du modèle avec les stratégies optimales', fontsize=14)
        plt.xlabel('Catégorie de distance', fontsize=12)
        plt.ylabel('Down', fontsize=12)
        plt.tight_layout()
        plt.savefig("Visuals/model_alignment_with_optimal_module3.png")
        plt.close()

        # 4. Impact des stratégies sur les résultats des drives
        # Calculer le pourcentage de jeux où le modèle a choisi différemment de la réalité
        plays_df['model_differs'] = plays_df['play_type'] != plays_df[
            'model_play_choice']

        # Agréger par drive
        drive_strategy_impact = plays_df.groupby('drive_id').agg({
            'model_differs': 'mean',
            # Pourcentage de jeux où le modèle a choisi différemment
            'yards_gained': 'sum',  # Total des yards gagnés dans le drive
        }).reset_index()

        # Fusionner avec les résultats des drives
        drive_strategy_impact = drive_strategy_impact.merge(
            drives_df[['drive_id', 'outcome', 'points']],
            on='drive_id',
            how='left'
        )

        # Créer des buckets basés sur le pourcentage de différences
        drive_strategy_impact['diff_bucket'] = pd.cut(
            drive_strategy_impact['model_differs'],
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=['0-25%', '25-50%', '50-75%', '75-100%']
        )

        # Analyser l'impact sur les points par drive
        plt.figure(figsize=(10, 6))

        points_by_diff = drive_strategy_impact.groupby('diff_bucket')['points'].mean()

        ax = points_by_diff.plot(kind='bar', color='purple')
        plt.title('Impact des différences stratégiques sur les points par drive',
                  fontsize=14)
        plt.xlabel('Pourcentage de jeux où le modèle diffère', fontsize=12)
        plt.ylabel('Points moyens par drive', fontsize=12)
        plt.grid(axis='y', alpha=0.3)

        # Ajouter les valeurs
        for i, v in enumerate(points_by_diff):
            ax.text(i, v + 0.1, f'{v:.2f}', ha='center')

        plt.tight_layout()
        plt.savefig("Visuals/strategy_impact_on_points_module3.png")
        plt.close()

        # 5. Matrice de confusion avancée par situation
        # Créer des matrices de confusion pour des situations spécifiques
        key_situations = [
            {'down': 3, 'distance_category': 'short', 'label': '3e et courte'},
            {'down': 3, 'distance_category': 'long', 'label': '3e et longue'},
            {'down': 4, 'distance_category': 'very_short', 'label': '4e et très courte'}
        ]

        for situation in key_situations:
            # Filtrer les jeux pour cette situation
            mask = (plays_df['down'] == situation['down']) & (
                        plays_df['distance_category'] == situation['distance_category'])
            situation_plays = plays_df[mask]

            if len(situation_plays) > 10:  # S'assurer qu'il y a assez de données
                # Créer la matrice de confusion
                conf_matrix = pd.crosstab(
                    situation_plays['play_type'],
                    situation_plays['model_play_choice'],
                    normalize='index'
                ) * 100  # En pourcentage

                plt.figure(figsize=(8, 6))
                sns.heatmap(conf_matrix, annot=True, fmt='.1f', cmap='Blues',
                            xticklabels=['Course', 'Passe'],
                            yticklabels=['Course', 'Passe'])
                plt.xlabel('Prédit par le modèle', fontsize=12)
                plt.ylabel('Réel', fontsize=12)
                plt.title(f'Matrice de confusion - {situation["label"]}', fontsize=14)
                plt.tight_layout()
                plt.savefig(
                    f"Visuals/confusion_matrix_{situation['down']}_{situation['distance_category']}_module3.png")
                plt.close()

    return {
        'strategy_comparison': strategy_comparison if X_real is not None else None,
        'drive_impact': drive_strategy_impact if X_real is not None else None
    }


# 6. SAUVEGARDE ET DÉPLOIEMENT DU MODÈLE AMÉLIORÉ
def save_model(model, feature_names):
    """
    Sauvegarde le modèle et les informations associées avec des métadonnées plus complètes.
    """
    import joblib
    import pickle
    import json
    import datetime
    import os

    # Créer le répertoire de sortie si nécessaire
    os.makedirs("OutputFiles", exist_ok=True)

    # 1. Sauvegarder le modèle
    model_path = "OutputFiles/football_strategy_model.joblib"
    joblib.dump(model, model_path)

    # 2. Sauvegarder les noms des caractéristiques
    with open("OutputFiles/feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)

    # 3. Sauvegarder les métadonnées du modèle
    model_info = {
        "model_type": type(model).__name__,
        "feature_count": len(feature_names),
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hyperparameters": model.get_params(),
    }

    with open("OutputFiles/model_metadata.json", "w") as f:
        json.dump(model_info, f, indent=4)

    # 4. Créer un script d'exemple pour utiliser le modèle
    example_script = """
# Exemple d'utilisation du modèle de prédiction de stratégie de football
import joblib
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Charger le modèle et les noms des caractéristiques
model = joblib.load('OutputFiles/football_strategy_model.joblib')
with open('OutputFiles/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Fonction pour prédire le type de jeu optimal
def predict_play_type(situation):
    # Créer un DataFrame avec les caractéristiques nécessaires
    data = pd.DataFrame([situation])

    # Vérifier et compléter les caractéristiques manquantes
    for feature in feature_names:
        if feature not in data.columns:
            data[feature] = 0

    # Sélectionner uniquement les caractéristiques utilisées par le modèle
    X = data[feature_names]

    # Normaliser les données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Prédire le type de jeu
    pass_probability = model.predict_proba(X_scaled)[0, 1]
    play_type = 'pass' if pass_probability >= 0.5 else 'run'

    return {
        'play_type': play_type,
        'pass_probability': pass_probability
    }

# Exemple d'utilisation
situation = {
    'down': 3,
    'yards_to_go': 8,
    'position': 75,
    # Ajouter d'autres caractéristiques selon besoin
}

prediction = predict_play_type(situation)
print(f"Type de jeu recommandé: {prediction['play_type']}")
print(f"Probabilité de passe: {prediction['pass_probability']:.2f}")
"""

    with open("OutputFiles/model_usage_example.py", "w") as f:
        f.write(example_script)

    print("\nModèle et informations associées sauvegardés dans le dossier OutputFiles.")
    print(f"- Modèle: {model_path}")
    print(f"- Caractéristiques: OutputFiles/feature_names.pkl")
    print(f"- Métadonnées: OutputFiles/model_metadata.json")
    print(f"- Exemple d'utilisation: OutputFiles/model_usage_example.py")

    return True


# Fonction principale améliorée
def main():
    """
    Fonction principale pour l'entraînement et l'évaluation du modèle avec des analyses plus avancées.
    """
    import os
    import time
    import pickle
    import pandas as pd

    # Mesurer le temps d'exécution
    start_time = time.time()

    # Créer les répertoires de sortie si nécessaires
    os.makedirs("Visuals", exist_ok=True)
    os.makedirs("OutputFiles", exist_ok=True)

    print("=" * 80)
    print("MODULE 3 - APPRENTISSAGE AUTOMATIQUE")
    print("=" * 80)

    # 1. Charger les données d'entraînement
    train_data_path = "OutputFiles/simulation_training_data.csv"
    drives_data_path = "OutputFiles/drives_summary.csv"

    if not os.path.exists(train_data_path):
        print(f"ERREUR: Les données d'entraînement n'existent pas: {train_data_path}")
        print("Veuillez d'abord exécuter le module 2 (module2_monte_carlo.py).")
        return

    print("Chargement des données d'entraînement...")
    plays_df = pd.read_csv(train_data_path)

    # Charger les données des drives si disponibles
    drives_df = None
    if os.path.exists(drives_data_path):
        drives_df = pd.read_csv(drives_data_path)
        print(f"Données chargées: {len(plays_df)} jeux dans {len(drives_df)} drives")
    else:
        print(
            f"Données chargées: {len(plays_df)} jeux (données des drives non disponibles)")

    # 2. Préparer les données pour l'entraînement
    print("\nPréparation des données...")
    if drives_df is not None:
        X_train, X_test, y_train, y_test, feature_names = prepare_data_for_training(
            plays_df, drives_df)
    else:
        # Fallback si les données des drives ne sont pas disponibles
        X_train, X_test, y_train, y_test, feature_names = prepare_data_for_training(
            plays_df, None)

    # 3. Entraîner le modèle
    print("\nEntraînement du modèle...")
    model = train_model(X_train, y_train, feature_names, tune_hyperparameters=True)

    # 4. Évaluer le modèle
    evaluation_metrics = evaluate_model(model, X_test, y_test, feature_names)

    # 5. Analyser les décisions du modèle
    situations_df = analyze_model_decisions(model, feature_names)

    # 6. Analyse stratégique avancée
    strategic_results = strategic_analysis(model, feature_names, plays_df, drives_df)

    # 7. Sauvegarder le modèle
    save_model(model, feature_names)

    # Rapport final
    execution_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("RÉSUMÉ DU MODÈLE D'APPRENTISSAGE AUTOMATIQUE")
    print("=" * 80)
    print(f"Exactitude globale: {evaluation_metrics['accuracy']:.4f}")
    print(f"AUC ROC: {evaluation_metrics['roc_auc']:.4f}")
    print(f"Précision moyenne: {evaluation_metrics['avg_precision']:.4f}")
    print(f"\nTemps d'exécution: {execution_time:.2f} secondes")
    print("\nEntraînement et analyse du modèle terminés avec succès!")
    print("Les visualisations sont disponibles dans le dossier 'Visuals'")
    print("Le modèle et ses métadonnées sont sauvegardés dans le dossier 'OutputFiles'")

    return model, evaluation_metrics


if __name__ == "__main__":
    main()