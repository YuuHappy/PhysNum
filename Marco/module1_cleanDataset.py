"""
Script pour charger et préparer les données NFL pour la simulation de football.
Ce script est conçu pour être exécuté en premier afin de préparer les données
qui seront utilisées par la simulation principale.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

viz_dir = "Visuals"
output_dir = "OutputFiles"


def load_and_clean_nfl_data(file_path):
    """
    Charge et nettoie les données NFL.

    Args:
        file_path: Chemin vers le fichier CSV des données NFL

    Returns:
        pd.DataFrame: Données nettoyées
    """
    print(f"Chargement des données depuis {file_path}...")
    df = pd.read_csv(file_path)

    print("Informations sur le dataset original:")
    print(f"Nombre total de jeux: {len(df)}")
    print(f"Nombre de matchs uniques: {df['gameId'].nunique()}")

    # Identifier les types de jeux (passe/course)
    df['play_type'] = 'other'

    # Identifier les passes
    pass_mask = df['PassResult'].notna()
    df.loc[pass_mask, 'play_type'] = 'pass'

    # Identifier les courses (ni passe ni jeu spécial)
    run_mask = (~pass_mask) & (~df['isSTPlay']) & (df['PlayResult'].notna())
    df.loc[run_mask, 'play_type'] = 'run'


    # Filtrer pour ne garder que les jeux de type "pass" et "run" pour les downs 1-4
    filtered_df = df[(df['play_type'].isin(['pass', 'run'])) & (df['down'].isin([1, 2, 3, 4]))].copy()

    # Statistiques sur les types de jeux identifiés
    play_type_counts = filtered_df['play_type'].value_counts()
    print("\nRépartition des types de jeux identifiés:")
    print(play_type_counts)

    print(f"Nombre de jeux après filtrage (uniquement pass/run et downs 1-4): {len(filtered_df)}")

    # Calculer la position sur le terrain (de 0 à 100)
    filtered_df.loc[:, 'field_position'] = np.where(
        filtered_df['yardlineSide'] == filtered_df['possessionTeam'],
        filtered_df['yardlineNumber'],
        100 - filtered_df['yardlineNumber']
    )

    # Vérifier les valeurs manquantes
    missing_values = filtered_df[['down', 'yardsToGo', 'field_position', 'play_type', 'PlayResult']].isna().sum()
    print("\nValeurs manquantes dans les colonnes principales:")
    print(missing_values)

    # Supprimer les lignes avec des valeurs manquantes dans les colonnes essentielles
    clean_df = filtered_df.dropna(subset=['down', 'yardsToGo', 'field_position', 'play_type', 'PlayResult'])
    print(f"Nombre de jeux après suppression des valeurs manquantes: {len(clean_df)}")

    return clean_df


def create_distribution_visualizations(df):
    """
    Crée des visualisations des distributions principales.
    """
    # Créer le répertoire s'il n'existe pas
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Distribution des yards gagnés par type de jeu
    plt.figure(figsize=(12, 6))

    # Filtrer les valeurs extrêmes pour une meilleure visualisation
    run_results = df[df['play_type'] == 'run']['PlayResult']
    run_results = run_results[(run_results >= -10) & (run_results <= 20)]

    pass_results = df[df['play_type'] == 'pass']['PlayResult']
    pass_results = pass_results[(pass_results >= -10) & (pass_results <= 30)]

    plt.hist(run_results, bins=30, alpha=0.5, label='Course', color='green')
    plt.hist(pass_results, bins=30, alpha=0.5, label='Passe', color='blue')

    plt.xlabel('Yards gagnés')
    plt.ylabel('Fréquence')
    plt.title('Distribution des yards gagnés par type de jeu')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{viz_dir}/yards_distribution_module1.png")
    plt.close()

    # 2. Boîtes à moustaches des yards gagnés par down et type de jeu
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df, x='down', y='PlayResult', hue='play_type',palette=['green', 'blue'])
    plt.title('Yards gagnés par down et type de jeu')
    plt.ylim(-10, 20)  # Limiter pour une meilleure lisibilité
    plt.savefig(f"{viz_dir}/yards_by_down_boxplot_module1.png")
    plt.close()

    # 3. Proportion de courses/passes par down
    downs_count = df.groupby(['down', 'play_type']).size().unstack()
    downs_pct = downs_count.div(downs_count.sum(axis=1), axis=0) * 100

    plt.figure(figsize=(10, 6))
    downs_pct.plot(kind='bar', stacked=True, color=['green', 'blue'])
    plt.title('Proportion de courses/passes par down')
    plt.xlabel('Down')
    plt.ylabel('Pourcentage')
    plt.xticks(rotation=0)

    # Ajouter les pourcentages sur les barres
    for i, down in enumerate(downs_pct.index):
        # Pour les courses (en bas)
        pct_run = downs_pct.iloc[i, 0]  # Supposons que 'run' est la première colonne
        plt.text(i, pct_run / 2, f"{pct_run:.1f}%", ha='center', color='white', fontweight='bold')

        # Pour les passes (en haut)
        pct_pass = downs_pct.iloc[i, 1]  # Supposons que 'pass' est la deuxième colonne
        plt.text(i, pct_run + pct_pass / 2, f"{pct_pass:.1f}%", ha='center', color='white', fontweight='bold')

    plt.savefig(f"{viz_dir}/play_type_by_down_module1.png")
    plt.close()

    # 4. Proportion de courses/passes par position sur le terrain
    # Créer des catégories pour la position sur le terrain
    df['field_position_category'] = pd.cut(
        df['field_position'],
        bins=[0, 20, 50, 80, 100],
        labels=['0-20', '20-50', '50-80', '80-100']
    )

    position_count = df.groupby(['field_position_category', 'play_type'], observed=False).size().unstack()
    position_pct = position_count.div(position_count.sum(axis=1), axis=0) * 100

    plt.figure(figsize=(10, 6))
    position_pct.plot(kind='bar', stacked=True, color=['green', 'blue'])
    plt.title('Proportion de courses/passes par position sur le terrain')
    plt.xlabel('Position sur le terrain (yards)')
    plt.ylabel('Pourcentage')
    plt.xticks(rotation=0)

    # Ajouter les pourcentages sur les barres
    for i, pos in enumerate(position_pct.index):
        # Pour les courses (en bas)
        pct_run = position_pct.iloc[i, 0]  # Supposons que 'run' est la première colonne
        plt.text(i, pct_run / 2, f"{pct_run:.1f}%", ha='center', color='white', fontweight='bold')

        # Pour les passes (en haut)
        pct_pass = position_pct.iloc[i, 1]  # Supposons que 'pass' est la deuxième colonne
        plt.text(i, pct_run + pct_pass / 2, f"{pct_pass:.1f}%", ha='center', color='white', fontweight='bold')

    plt.savefig(f"{viz_dir}/play_type_by_position_module1.png")
    plt.close()

    # 5. Yards moyens gagnés par catégorie de yards à franchir
    df['yardage_category'] = pd.cut(df['yardsToGo'],
                                    bins=[0, 3, 6, 10, 20, 100],
                                    labels=['1-3', '4-6', '7-10', '11-20', '20+'])

    yardage_results = df.groupby(['yardage_category', 'play_type'], observed=False)['PlayResult'].mean().unstack()

    plt.figure(figsize=(10, 6))
    yardage_results.plot(kind='bar', color=['green', 'blue'])
    plt.title('Yards moyens gagnés par catégorie de yards à franchir')
    plt.xlabel('Yards à franchir')
    plt.ylabel('Yards moyens gagnés')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=0)
    plt.savefig(f"{viz_dir}/yards_by_distance_module1.png")
    plt.close()


def extract_simulation_parameters(df):
    """
    Extrait les paramètres nécessaires pour alimenter la simulation.

    Args:
        df: DataFrame des données NFL nettoyées

    Returns:
        dict: Paramètres pour la simulation
    """
    # 1. Paramètres de base pour les résultats des jeux
    params = {
        'run_mean': df[df['play_type'] == 'run']['PlayResult'].mean(),
        'run_std': df[df['play_type'] == 'run']['PlayResult'].std(),
        'pass_mean': df[df['play_type'] == 'pass']['PlayResult'].mean(),
        'pass_std': df[df['play_type'] == 'pass']['PlayResult'].std(),
    }

    # 2. Taux d'interception
    if 'PassResult' in df.columns:
        params['interception_rate'] = df[df['play_type'] == 'pass']['PassResult'].eq('IN').mean()
    else:
        params['interception_rate'] = 0.03  # Valeur par défaut

    # 3. Taux de fumble (approximation)
    fumble_mask = df['playDescription'].str.contains('FUMBLE', case=False, na=False)
    params['fumble_rate'] = fumble_mask[df['play_type'] == 'run'].mean()

    # 4. Taux de complétion de passe
    if 'PassResult' in df.columns:
        completion_mask = df[df['play_type'] == 'pass']['PassResult'].eq('C')
        params['completion_rate'] = completion_mask.mean()
    else:
        params['completion_rate'] = 0.65  # Valeur par défaut

    # 5. Paramètres spécifiques par down
    down_stats = df.groupby(['down', 'play_type'])['PlayResult'].agg(['mean', 'std']).reset_index()
    params['down_adjustments'] = {}

    for _, row in down_stats.iterrows():
        down = row['down']
        play_type = row['play_type']
        if down not in params['down_adjustments']:
            params['down_adjustments'][down] = {}

        params['down_adjustments'][down][play_type] = {
            'mean_adjust': row['mean'] - params[f'{play_type}_mean'],
            'std_adjust': row['std'] - params[f'{play_type}_std']
        }

    # 6. Catégories pour la position sur le terrain
    df['field_position_category'] = pd.cut(
        df['field_position'],
        bins=[0, 20, 50, 80, 100],
        labels=['0-20', '20-50', '50-80', '80-100']
    )

    # 7. Paramètres spécifiques par position sur le terrain
    field_stats = df.groupby(['field_position_category', 'play_type'], observed=False)['PlayResult'].agg(['mean', 'std']).reset_index()
    params['field_adjustments'] = {}

    for _, row in field_stats.iterrows():
        position = row['field_position_category']
        play_type = row['play_type']
        if position not in params['field_adjustments']:
            params['field_adjustments'][position] = {}

        params['field_adjustments'][position][play_type] = {
            'mean_adjust': row['mean'] - params[f'{play_type}_mean'],
            'std_adjust': row['std'] - params[f'{play_type}_std']  # Ajout de l'ajustement d'écart-type
        }

    # 8. Catégories pour les yards à franchir
    df['yardage_category'] = pd.cut(df['yardsToGo'],
                                    bins=[0, 3, 6, 10, 20, 100],
                                    labels=['1-3', '4-6', '7-10', '11-20', '20+'])

    # 9. Distribution des choix de jeu avec informations supplémentaires
    # Créer une table de probabilités plus précise
    play_choice_data = []

    # Pour chaque combinaison de down, yards à franchir et position
    for down in range(1, 5):
        for yard_cat in ['1-3', '4-6', '7-10', '11-20', '20+']:
            for pos_cat in ['0-20', '20-50', '50-80', '80-100']:
                # Filtrer les données
                filtered = df[(df['down'] == down) &
                              (df['yardage_category'] == yard_cat) &
                              (df['field_position_category'] == pos_cat)]

                if len(filtered) > 0:
                    # Calculer les probabilités
                    run_prob = (filtered['play_type'] == 'run').mean()
                    pass_prob = 1 - run_prob

                    # Ajouter les statistiques de résultat
                    run_yards_mean = filtered[filtered['play_type'] == 'run'][
                        'PlayResult'].mean() if len(
                        filtered[filtered['play_type'] == 'run']) > 0 else np.nan
                    pass_yards_mean = filtered[filtered['play_type'] == 'pass'][
                        'PlayResult'].mean() if len(
                        filtered[filtered['play_type'] == 'pass']) > 0 else np.nan

                    play_choice_data.append({
                        'down': down,
                        'yardage_category': yard_cat,
                        'field_position_category': pos_cat,
                        'run': run_prob,
                        'pass': pass_prob,
                        'run_yards_mean': run_yards_mean,
                        'pass_yards_mean': pass_yards_mean,
                        'num_plays': len(filtered)
                    })

    play_choice_stats = pd.DataFrame(play_choice_data)
    params['play_choice_stats'] = play_choice_stats

    # 10. Probabilité de first down par type de jeu et yards à franchir
    first_down_stats = []

    for play_type in ['run', 'pass']:
        for yard_cat in ['1-3', '4-6', '7-10', '11-20', '20+']:
            filtered = df[(df['play_type'] == play_type) &
                          (df['yardage_category'] == yard_cat)]

            if len(filtered) > 0:
                # Calculer si le jeu a atteint un premier down
                first_down_prob = (filtered['PlayResult'] >= filtered['yardsToGo']).mean()

                first_down_stats.append({
                    'play_type': play_type,
                    'yardage_category': yard_cat,
                    'first_down_probability': first_down_prob,
                    'num_plays': len(filtered)
                })

    first_down_df = pd.DataFrame(first_down_stats)
    params['first_down_stats'] = first_down_df

    # 11. Afficher un résumé des paramètres extraits
    print("\nParamètres extraits pour la simulation:")
    print(f"Yards moyens en course: {params['run_mean']:.2f} (écart-type: {params['run_std']:.2f})")
    print(f"Yards moyens en passe: {params['pass_mean']:.2f} (écart-type: {params['pass_std']:.2f})")
    print(f"Taux d'interception: {params['interception_rate']:.4f}")
    print(f"Taux de fumble : {params['fumble_rate']:.4f}")
    print(f"Taux de complétion de passe: {params['completion_rate']:.4f}")

    # Visualisation pour la probabilité de premier down
    if 'first_down_stats' in params:
        plt.figure(figsize=(10, 6))
        first_down_pivot = params['first_down_stats'].pivot(
            index='yardage_category',
            columns='play_type',
            values='first_down_probability'
        )
        first_down_pivot.plot(kind='bar', color=['green', 'blue'])
        plt.title('Probabilité d\'obtenir un premier down par type de jeu et yards à franchir')
        plt.xlabel('Yards à franchir')
        plt.ylabel('Probabilité')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=0)
        plt.savefig(f"{viz_dir}/first_down_probability_module1.png")
        plt.close()

    return params


def calculate_scoring_stats(df):
    """
    Calcule des statistiques liées aux scores pour comparer avec nos simulations.

    Args:
        df: DataFrame des données NFL nettoyées

    Returns:
        dict: Statistiques liées aux scores
    """
    # Identifier les drives - nous utilisons une approximation simple
    # Nous considérons qu'un nouveau drive commence quand:
    # - le jeu est le premier down (down=1)
    # - et la position change significativement

    # Trier par match, puis par temps de jeu
    df_sorted = df.sort_values(['gameId', 'quarter', 'GameClock']).copy()

    # Créer une colonne qui indique le début d'un nouveau drive
    df_sorted['new_drive'] = (
            # Nouveau match
            (df_sorted['gameId'] != df_sorted['gameId'].shift(1)) |
            # Changement d'équipe en possession
            (df_sorted['possessionTeam'] != df_sorted['possessionTeam'].shift(1)) |
            # Nouveau quart-temps
            (df_sorted['quarter'] != df_sorted['quarter'].shift(1)) |
            # Premier down après un turnover (interception ou fumble)
            ((df_sorted['down'] == 1) &
             df_sorted['playDescription'].shift(1).str.contains('INTERCEPTED|FUMBLE', case=False, na=False)) |
            # Premier down après un punt
            ((df_sorted['down'] == 1) &
             df_sorted['playDescription'].shift(1).str.contains('PUNT', case=False, na=False)) |
            # Premier down après un field goal (réussi ou manqué)
            ((df_sorted['down'] == 1) &
             df_sorted['playDescription'].shift(1).str.contains('FIELD GOAL', case=False, na=False)))

    # Gérer le premier jeu (toujours un nouveau drive)
    df_sorted.loc[0, 'new_drive'] = True

    # Numéroter les drives
    df_sorted['drive_id'] = df_sorted['new_drive'].cumsum()

    # Calculer les statistiques par drive
    drive_stats = {}

    df_sorted['touchdown'] = (
            ((df_sorted['HomeScoreBeforePlay'].shift(1) - df_sorted['HomeScoreBeforePlay']) > 4) |
            ((df_sorted['VisitorScoreBeforePlay'].shift(1) - df_sorted['VisitorScoreBeforePlay']) > 4))

    df_sorted['field_goal'] = (
            ((df_sorted['HomeScoreBeforePlay'].shift(1) - df_sorted['HomeScoreBeforePlay']) == 3) |
            ((df_sorted['VisitorScoreBeforePlay'].shift(1) - df_sorted['VisitorScoreBeforePlay']) == 3))

    # Agréger par drive
    drive_data = df_sorted.groupby('drive_id').agg({
        'PlayResult': 'sum',         # Total des yards gagnés
        'touchdown': 'max',          # Si le drive s'est terminé par un touchdown
        'field_goal': 'max',         # Si le drive s'est terminé par un field goal
        'play_type': 'count'         # Nombre de jeux dans le drive
    }).reset_index()

    # Renommer les colonnes
    drive_data = drive_data.rename(columns={
        'PlayResult': 'total_yards',
        'play_type': 'num_plays'
    })


    # Calculer les statistiques finales
    drive_stats = {}
    drive_stats['avg_yards_per_drive'] = drive_data['total_yards'].mean()
    drive_stats['touchdown_rate'] = drive_data['touchdown'].mean()
    drive_stats['field_goal_rate'] = drive_data['field_goal'].mean()
    drive_stats['avg_plays_per_drive'] = drive_data['num_plays'].mean()

    # Nombre total de drives et scoring drives
    drive_stats['total_drives'] = len(drive_data)
    drive_stats['total_touchdown_drives'] = drive_data['touchdown'].sum()
    drive_stats['total_field_goal_drives'] = drive_data['field_goal'].sum()

    # Points par drive (7 points par touchdown, 3 points par field goal)
    drive_stats['points_per_drive'] = (drive_data['touchdown'].mean() * 7) + (drive_data['field_goal'].mean() * 3)

    print("\nStatistiques des drives réels NFL:")
    print(f"Nombre total de drives: {drive_stats['total_drives']}")
    print(f"Yards moyens par drive: {drive_stats['avg_yards_per_drive']:.2f}")
    print(f"Taux de touchdown: {drive_stats['touchdown_rate']:.4f} ({drive_stats['total_touchdown_drives']} drives)")
    print(f"Taux de field goal: {drive_stats['field_goal_rate']:.4f} ({drive_stats['total_field_goal_drives']} drives)")
    print(f"Jeux moyens par drive: {drive_stats['avg_plays_per_drive']:.2f}")
    print(f"Points moyens par drive: {drive_stats['points_per_drive']:.2f}")

    return drive_stats


def main(file_path):
    """
    Fonction principale pour traiter les données NFL.

    Args:
        file_path: Chemin vers le fichier CSV des données NFL

    Returns:
        tuple: (DataFrame nettoyé, paramètres de simulation, statistiques des drives)
    """
    # 1. Charger et nettoyer les données
    clean_df = load_and_clean_nfl_data(file_path)

    # 2. Créer des visualisations
    create_distribution_visualizations(clean_df)

    # 3. Extraire les paramètres pour la simulation
    params = extract_simulation_parameters(clean_df)

    # 4. Calculer les statistiques des drives réels pour comparaison
    drive_stats = calculate_scoring_stats(clean_df)

    # 5. Sauvegarder les données nettoyées et les paramètres
    clean_df.to_csv(f"{output_dir}/nfl_clean_data.csv", index=False)
    print("Données nettoyées sauvegardées dans nfl_clean_data.csv")

    with open(f"{output_dir}/nfl_simulation_params.pkl", "wb") as f:
        pickle.dump(params, f)
    print("Paramètres de simulation sauvegardés dans nfl_simulation_params.pkl")

    with open(f"{output_dir}/nfl_drive_stats.pkl", "wb") as f:
        pickle.dump(drive_stats, f)
    print("Statistiques des drives sauvegardées dans nfl_drive_stats.pkl")

    return clean_df, params, drive_stats


if __name__ == "__main__":
    # Remplacer par le chemin vers votre fichier de données
    data_file = "../Data/plays.csv"
    main(data_file)

    import pickle

    # Charger le fichier de paramètres
    with open("OutputFiles/nfl_simulation_params.pkl", "rb") as f:
        params = pickle.load(f)

    # Accéder aux statistiques
    play_choice_stats = params['play_choice_stats']
    first_down_stats = params['first_down_stats']

    # Afficher les statistiques
    print("Play Choice Stats:")
    print(play_choice_stats)
    print("\nFirst Down Stats:")
    print(first_down_stats)

