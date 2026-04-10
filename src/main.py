"""Script principal pour exécuter le pipeline ML complet pour deux datasets."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

try:
    from src import config, data_loader, preprocess, train, evaluate, persistence
except ModuleNotFoundError:
    import config
    import data_loader
    import preprocess
    import train
    import evaluate
    import persistence


OUTPUT_DIR = Path('ml_artifacts')
OUTPUT_DIR.mkdir(exist_ok=True)

DATASET_1_PATH = '../data/UNICEF-CME_DF_2021_WQ-1.0-download (1).csv'
DATASET_2_PATH = '../data/UNICEF-CME_CAUSE_OF_DEATH-1.0-download (1).csv'

TARGET_DATASET_1 = 'OBS_VALUE'
TARGET_DATASET_2 = 'OBS_VALUE'

DATASET_2_FEATURES = ['Cause of death', 'Sex', 'TIME_PERIOD']


def run_pipeline_dataset_1():
    """Exécute le pipeline complet pour le Dataset 1 (mortalité infantile principale)."""
    print("=" * 70)
    print("CHARGEMENT DES DONNÉES - DATASET 1 (CME_DF)")
    print("=" * 70)
    
    df = data_loader.load_data(DATASET_1_PATH)
    print(f"Forme initiale: {df.shape}")
    
    df = data_loader.get_tunisia_data(df)
    print(f"Forme après filtrage Tunisie: {df.shape}")
    
    print("\n" + "=" * 70)
    print("PRÉTRAITEMENT DES DONNÉES - DATASET 1")
    print("=" * 70)
    
    df = preprocess.clean_data(df)
    print(f"Forme après nettoyage: {df.shape}")
    
    df = preprocess.add_derived_features(df)
    print(f"Forme après features dérivées: {df.shape}")
    
    summary = data_loader.get_summary(df)
    print(f"Shape final: {summary['shape']}")
    
    print("\n" + "=" * 70)
    print("CONFIGURATION A (5 features) - Dataset 1")
    print("=" * 70)
    
    features_a = preprocess.get_features_config(df, 'A')
    print(f"Features: {features_a}")
    
    X_train_a, X_test_a, y_train_a, y_test_a = preprocess.prepare_data(
        df, features_a, TARGET_DATASET_1
    )
    
    preprocessor_a = train.build_preprocessor(features_a, config.NUMERIC_COLS)
    pipelines_a = train.train_models(X_train_a, y_train_a, preprocessor_a, config.MODELS)
    results_a = train.evaluate_models(pipelines_a, X_test_a, y_test_a)
    print(results_a.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("CONFIGURATION B (12 features) - Dataset 1")
    print("=" * 70)
    
    features_b = preprocess.get_features_config(df, 'B')
    print(f"Features: {features_b}")
    
    X_train_b, X_test_b, y_train_b, y_test_b = preprocess.prepare_data(
        df, features_b, TARGET_DATASET_1
    )
    
    preprocessor_b = train.build_preprocessor(features_b, config.NUMERIC_COLS)
    pipelines_b = train.train_models(X_train_b, y_train_b, preprocessor_b, config.MODELS)
    results_b = train.evaluate_models(pipelines_b, X_test_b, y_test_b)
    print(results_b.to_string(index=False))
    
    return {
        'results_a': results_a,
        'results_b': results_b,
        'features_a': features_a,
        'features_b': features_b,
        'X_train_a': X_train_a,
        'X_test_a': X_test_a,
        'y_train_a': y_train_a,
        'y_test_a': y_test_a,
        'X_train_b': X_train_b,
        'X_test_b': X_test_b,
        'y_train_b': y_train_b,
        'y_test_b': y_test_b,
        'preprocessor_a': preprocessor_a,
        'preprocessor_b': preprocessor_b,
        'pipelines_a': pipelines_a,
        'pipelines_b': pipelines_b
    }


def load_and_preprocess_dataset_2():
    """Charge et prétraite le Dataset 2 (cause of death)."""
    print("\n" + "=" * 70)
    print("CHARGEMENT DES DONNÉES - DATASET 2 (CAUSE OF DEATH)")
    print("=" * 70)
    
    df = pd.read_csv(DATASET_2_PATH, encoding='utf-8')
    print(f"Forme initiale: {df.shape}")
    print(f"Colonnes: {df.columns.tolist()}")
    
    df_clean = df.copy()
    
    cols_100_missing = df_clean.columns[df_clean.isnull().all()].tolist()
    df_clean = df_clean.drop(columns=cols_100_missing)
    
    if 'TIME_PERIOD' in df_clean.columns:
        df_clean['YEAR_EXTRACT'] = df_clean['TIME_PERIOD'].apply(_extract_year)
        df_clean = df_clean.drop(columns=['TIME_PERIOD'])
    
    df_clean = df_clean.sort_values('YEAR_EXTRACT').reset_index(drop=True)
    
    print(f"Forme après prétraitement: {df_clean.shape}")
    print(f"Features disponibles: {df_clean.columns.tolist()}")
    
    return df_clean


def _extract_year(value):
    """Extrait l'année d'une valeur TIME_PERIOD."""
    if pd.isna(value):
        return np.nan
    
    if isinstance(value, (int, float)):
        return float(value)
    
    value_str = str(value)
    
    if '-' in value_str:
        parts = value_str.split('-')
        try:
            return float(parts[0])
        except ValueError:
            return np.nan
    
    try:
        return float(value_str)
    except ValueError:
        return np.nan


def run_pipeline_dataset_2(df: pd.DataFrame):
    """Exécute le pipeline complet pour le Dataset 2 (cause of death)."""
    print("\n" + "=" * 70)
    print("CONFIGURATION - Dataset 2 (Cause of Death)")
    print("=" * 70)
    
    available_features = [f for f in DATASET_2_FEATURES if f in df.columns]
    print(f"Features: {available_features}")
    
    X_train_2, X_test_2, y_train_2, y_test_2 = preprocess.prepare_data(
        df, available_features, TARGET_DATASET_2
    )
    
    preprocessor_2 = train.build_preprocessor(available_features, set())
    pipelines_2 = train.train_models(X_train_2, y_train_2, preprocessor_2, config.MODELS)
    results_2 = train.evaluate_models(pipelines_2, X_test_2, y_test_2)
    print(results_2.to_string(index=False))
    
    return {
        'results': results_2,
        'features': available_features,
        'X_train': X_train_2,
        'X_test': X_test_2,
        'y_train': y_train_2,
        'y_test': y_test_2,
        'preprocessor': preprocessor_2,
        'pipelines': pipelines_2
    }


def compare_datasets(results_1_a, results_1_b, results_2):
    """Compare les résultats entre les deux datasets."""
    print("\n" + "=" * 70)
    print("COMPARAISON DES RÉSULTATS - TOUS DATASETS")
    print("=" * 70)
    
    results_1_a['Dataset'] = 'Dataset 1 - Config A'
    results_1_a['Config'] = 'A'
    results_1_b['Dataset'] = 'Dataset 1 - Config B'
    results_1_b['Config'] = 'B'
    results_2['Dataset'] = 'Dataset 2 - Cause of Death'
    results_2['Config'] = 'Cause'
    
    all_results = pd.concat([results_1_a, results_1_b, results_2], ignore_index=True)
    print(all_results.to_string(index=False))
    
    best_idx = all_results['RMSE'].idxmin()
    best_model = all_results.loc[best_idx, 'Model']
    best_dataset = all_results.loc[best_idx, 'Dataset']
    best_rmse = all_results.loc[best_idx, 'RMSE']
    
    print(f"\nMeilleur modèle全局: {best_model} ({best_dataset}) avec RMSE = {best_rmse:.4f}")
    
    return all_results


def optimize_best_model(dataset_1_results, dataset_2_results, results_1, results_2):
    """Optimise le meilleur modèle pour chaque dataset."""
    print("\n" + "=" * 70)
    print("OPTIMISATION - GRIDSEARCHCV")
    print("=" * 70)
    
    results_1_a = results_1['results_a']
    results_1_b = results_1['results_b']
    
    all_results_1 = pd.concat([results_1_a, results_1_b], ignore_index=True)
    best_idx_1 = all_results_1['RMSE'].idxmin()
    best_model_1 = all_results_1.loc[best_idx_1, 'Model']
    best_config_1 = all_results_1.loc[best_idx_1, 'Config']
    
    print(f"\nDataset 1 - Meilleur modèle: {best_model_1} (Config {best_config_1})")
    
    if best_config_1 == 'A':
        X_train_best_1 = results_1['X_train_a']
        X_test_best_1 = results_1['X_test_a']
        y_train_best_1 = results_1['y_train_a']
        y_test_best_1 = results_1['y_test_a']
        preprocessor_best_1 = results_1['preprocessor_a']
    else:
        X_train_best_1 = results_1['X_train_b']
        X_test_best_1 = results_1['X_test_b']
        y_train_best_1 = results_1['y_train_b']
        y_test_best_1 = results_1['y_test_b']
        preprocessor_best_1 = results_1['preprocessor_b']
    
    param_grids = {
        'Decision Tree': {
            'model__max_depth': [3, 5, 7, 10, None],
            'model__min_samples_split': [2, 5, 10]
        },
        'Random Forest': {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [5, 10, None]
        },
        'Gradient Boosting': {
            'model__n_estimators': [50, 100],
            'model__max_depth': [3, 5],
            'model__learning_rate': [0.05, 0.1]
        }
    }
    
    if best_model_1 in param_grids:
        param_grid_1 = param_grids[best_model_1]
        best_pipeline_1, best_params_1 = train.train_best_model(
            X_train_best_1, y_train_best_1, preprocessor_best_1, best_model_1, param_grid_1
        )
        print(f"Meilleurs paramètres: {best_params_1}")
    else:
        best_pipeline_1 = (results_1['pipelines_a'] if best_config_1 == 'A' 
                          else results_1['pipelines_b'])[best_model_1]
        best_params_1 = {}
    
    final_results_1 = train.evaluate_models({best_model_1: best_pipeline_1}, X_test_best_1, y_test_best_1)
    print(f"Dataset 1 - Résultats finaux:")
    print(final_results_1.to_string(index=False))
    
    print(f"\nDataset 2 - Meilleur modèle: {dataset_2_results['results'].loc[dataset_2_results['results']['RMSE'].idxmin(), 'Model']}")
    
    best_idx_2 = dataset_2_results['results']['RMSE'].idxmin()
    best_model_2 = dataset_2_results['results'].loc[best_idx_2, 'Model']
    X_train_best_2 = dataset_2_results['X_train']
    X_test_best_2 = dataset_2_results['X_test']
    y_train_best_2 = dataset_2_results['y_train']
    y_test_best_2 = dataset_2_results['y_test']
    preprocessor_best_2 = dataset_2_results['preprocessor']
    
    if best_model_2 in param_grids:
        param_grid_2 = param_grids[best_model_2]
        best_pipeline_2, best_params_2 = train.train_best_model(
            X_train_best_2, y_train_best_2, preprocessor_best_2, best_model_2, param_grid_2
        )
        print(f"Meilleurs paramètres: {best_params_2}")
    else:
        best_pipeline_2 = dataset_2_results['pipelines'][best_model_2]
        best_params_2 = {}
    
    final_results_2 = train.evaluate_models({best_model_2: best_pipeline_2}, X_test_best_2, y_test_best_2)
    print(f"Dataset 2 - Résultats finaux:")
    print(final_results_2.to_string(index=False))
    
    return {
        'best_pipeline_1': best_pipeline_1,
        'best_model_1': best_model_1,
        'best_config_1': best_config_1,
        'best_params_1': best_params_1,
        'X_train_best_1': X_train_best_1,
        'X_test_best_1': X_test_best_1,
        'y_train_best_1': y_train_best_1,
        'y_test_best_1': y_test_best_1,
        'best_pipeline_2': best_pipeline_2,
        'best_model_2': best_model_2,
        'best_params_2': best_params_2,
        'X_train_best_2': X_train_best_2,
        'X_test_best_2': X_test_best_2,
        'y_train_best_2': y_train_best_2,
        'y_test_best_2': y_test_best_2
    }


def cross_validate_and_visualize(opt_results):
    """Effectue la validation croisée et génère les visualisations."""
    print("\n" + "=" * 70)
    print("VALIDATION CROISÉE 5-FOLD")
    print("=" * 70)
    
    cv_results_1 = evaluate.cross_validate(
        opt_results['best_pipeline_1'], 
        opt_results['X_train_best_1'], 
        opt_results['y_train_best_1'], 
        cv=5
    )
    print(f"Dataset 1 - RMSE moyen: {cv_results_1['mean']:.4f} (+/- {cv_results_1['std']:.4f})")
    
    cv_results_2 = evaluate.cross_validate(
        opt_results['best_pipeline_2'], 
        opt_results['X_train_best_2'], 
        opt_results['y_train_best_2'], 
        cv=5
    )
    print(f"Dataset 2 - RMSE moyen: {cv_results_2['mean']:.4f} (+/- {cv_results_2['std']:.4f})")
    
    print("\n" + "=" * 70)
    print("VISUALISATIONS")
    print("=" * 70)
    
    try:
        y_pred_1 = evaluate.get_predictions(opt_results['best_pipeline_1'], opt_results['X_test_best_1'])
        
        fig_actual_1 = plt.figure(figsize=(8, 6))
        evaluate.plot_actual_vs_predicted(opt_results['y_test_best_1'].values, y_pred_1)
        persistence.save_plot(fig_actual_1, str(OUTPUT_DIR / 'dataset1_actual_vs_predicted.png'))
        
        fig_resid_1 = plt.figure(figsize=(8, 6))
        evaluate.plot_residuals(opt_results['y_test_best_1'].values, y_pred_1)
        persistence.save_plot(fig_resid_1, str(OUTPUT_DIR / 'dataset1_residuals.png'))
    except Exception as e:
        print(f"Erreur visualization Dataset 1: {e}")
    
    try:
        y_pred_2 = evaluate.get_predictions(opt_results['best_pipeline_2'], opt_results['X_test_best_2'])
        
        fig_actual_2 = plt.figure(figsize=(8, 6))
        evaluate.plot_actual_vs_predicted(opt_results['y_test_best_2'].values, y_pred_2)
        persistence.save_plot(fig_actual_2, str(OUTPUT_DIR / 'dataset2_actual_vs_predicted.png'))
        
        fig_resid_2 = plt.figure(figsize=(8, 6))
        evaluate.plot_residuals(opt_results['y_test_best_2'].values, y_pred_2)
        persistence.save_plot(fig_resid_2, str(OUTPUT_DIR / 'dataset2_residuals.png'))
    except Exception as e:
        print(f"Erreur visualization Dataset 2: {e}")
    
    return {'cv_results_1': cv_results_1, 'cv_results_2': cv_results_2}


def save_models_and_results(opt_results, all_comparison, cv_results):
    """Sauvegarde les modèles et résultats."""
    print("\n" + "=" * 70)
    print("SAUVEGARDE DES MODÈLES")
    print("=" * 70)
    
    model_path_1 = OUTPUT_DIR / 'best_model_dataset1.joblib'
    persistence.save_model(opt_results['best_pipeline_1'], str(model_path_1))
    print(f"Modèle Dataset 1 sauvegardé: {model_path_1}")
    
    model_path_2 = OUTPUT_DIR / 'best_model_dataset2.joblib'
    persistence.save_model(opt_results['best_pipeline_2'], str(model_path_2))
    print(f"Modèle Dataset 2 sauvegardé: {model_path_2}")
    
    metadata = {
        'dataset_1': {
            'model_name': opt_results['best_model_1'],
            'config': opt_results['best_config_1'],
            'best_params': opt_results['best_params_1'],
            'cv_results': cv_results['cv_results_1']
        },
        'dataset_2': {
            'model_name': opt_results['best_model_2'],
            'best_params': opt_results['best_params_2'],
            'cv_results': cv_results['cv_results_2']
        }
    }
    
    metadata_path = OUTPUT_DIR / 'metadata.json'
    persistence.save_metadata(metadata, str(metadata_path))
    
    results_path = OUTPUT_DIR / 'comparison_results.csv'
    persistence.save_results(all_comparison, str(results_path))
    
    print(f"Métadonnées sauvegardées: {metadata_path}")
    print(f"Résultats sauvegardés: {results_path}")
    
    print("\n" + "=" * 70)
    print("PIPELINE TERMINÉ AVEC SUCCÈS")
    print("=" * 70)


def run_pipeline():
    """Exécute le pipeline complet pour les deux datasets."""
    results_1 = run_pipeline_dataset_1()
    
    df_dataset_2 = load_and_preprocess_dataset_2()
    results_2 = run_pipeline_dataset_2(df_dataset_2)
    
    all_comparison = compare_datasets(
        results_1['results_a'], 
        results_1['results_b'], 
        results_2['results']
    )
    
    opt_results = optimize_best_model(results_1, results_2, results_1, results_2)
    
    cv_results = cross_validate_and_visualize(opt_results)
    
    save_models_and_results(opt_results, all_comparison, cv_results)


if __name__ == '__main__':
    run_pipeline()