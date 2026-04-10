"""Data preprocessing utilities for the infant mortality project."""

import pandas as pd
import numpy as np
from typing import List
from sklearn.model_selection import train_test_split

try:
    from src import config
except ModuleNotFoundError:
    import config


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les donnees en supprimant les colonnes non pertinentes.
    
    Applique aussi le feature engineering de base (YEAR_EXTRACT, REF_DECADE).
    
    Args:
        df: DataFrame avec les donnees brutes.
        
    Returns:
        DataFrame nettoye avec features derivees.
    """
    df = df.copy()
    
    for col in config.COLUMNS_TO_DROP:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    cols_100_missing = df.columns[df.isnull().all()].tolist()
    df = df.drop(columns=cols_100_missing)
    
    df = add_derived_features(df)
    
    date_col = 'REF_DATE' if 'REF_DATE' in df.columns else 'YEAR_EXTRACT'
    if date_col in df.columns:
        df = df.sort_values(date_col).reset_index(drop=True)
    
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des features dérivées.
    
    Args:
        df: DataFrame avec les données nettoyées.
        
    Returns:
        DataFrame avec les features dérivées.
    """
    df = df.copy()
    
    if 'TIME_PERIOD' in df.columns:
        df['YEAR_EXTRACT'] = df['TIME_PERIOD'].apply(_extract_year)
        df = df.drop(columns=['TIME_PERIOD'])
    
    if 'REF_DATE' in df.columns:
        df['REF_DECADE'] = (df['REF_DATE'] // 10) * 10
    
    return df


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


def get_features_config(df: pd.DataFrame, config_name: str) -> List[str]:
    """Retourne la liste des features selon la configuration.
    
    Args:
        df: DataFrame avec les données.
        config_name: Nom de la configuration ('A', 'B', ou 'C').
        
    Returns:
        Liste des noms de features.
    """
    mapping = {
        'A': config.CONFIG_A_FEATURES,
        'B': config.CONFIG_B_FEATURES,
        'C': config.CONFIG_C_FEATURES
    }
    
    feature_list = mapping.get(config_name.upper(), config.CONFIG_B_FEATURES)
    
    available_features = [f for f in feature_list if f in df.columns]
    
    return available_features


def prepare_data(
    df: pd.DataFrame, 
    features: List[str], 
    target: str = 'OBS_VALUE'
) -> tuple:
    """Prépare les données pour l'entraînement.
    
    Args:
        df: DataFrame avec les données.
        features: Liste des features à utiliser.
        target: Nom de la colonne cible.
        
    Returns:
        Tuple (X_train, X_test, y_train, y_test).
    """
    df_model = df.copy()
    
    for col in features:
        if col not in df_model.columns:
            raise ValueError(f"Feature {col} non trouvée dans les données")
    
    if target not in df_model.columns:
        raise ValueError(f"Cible {target} non trouvée dans les données")
    
    df_model = df_model.dropna(subset=features + [target])
    
    X = df_model[features]
    y = df_model[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE,
        shuffle=False
    )
    
    return X_train, X_test, y_train, y_test