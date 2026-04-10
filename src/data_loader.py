"""Data loading utilities for the infant mortality project."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

try:
    from src import config
except ModuleNotFoundError:
    import config


def load_wealth_data(path: Optional[str] = None) -> pd.DataFrame:
    """Charge uniquement les données wealth (UNICEF-CME_DF_2021_WQ).
    
    Args:
        path: Chemin vers le fichier CSV wealth. Si None, utilise config.DATA_PATH.
        
    Returns:
        DataFrame avec les données wealth chargées.
    """
    if path is None:
        path = config.DATA_PATH
    
    df = pd.read_csv(path, encoding='utf-8')
    
    df = _deduplicate_columns(df)
    
    df = _convert_numerics(df)
    
    return df


def load_data(path: Optional[str] = None) -> pd.DataFrame:
    """Charge les données wealth (méthode rétrocompatible).
    """
    return load_wealth_data(path)


def _deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les colonnes en double (label columns).
    
    Le fichier CSV contient des paires de colonnes: code + label.
    On supprime les colonnes label (colonnes avec des espaces).
    """
    cols_to_drop = []
    for col in df.columns:
        if ' ' in col and col.replace(' ', '_') in df.columns:
            cols_to_drop.append(col)
    
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    return df


def _convert_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit les colonnes numériques."""
    numeric_cols = ['REF_DATE', 'SERIES_YEAR', 'LOWER_BOUND', 'UPPER_BOUND', 
                   'INTERVAL', 'YEAR_EXTRACT', 'REF_DECADE']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def get_tunisia_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filtre les données pour la Tunisie.
    
    Si les données sont déjà filtrées ou si REF_AREA n'existe pas,
    retourne le DataFrame tel quel.
    
    Args:
        df: DataFrame avec les données.
        
    Returns:
        DataFrame filtré pour la Tunisie.
    """
    if 'REF_AREA' not in df.columns:
        return df
    
    unique_areas = df['REF_AREA'].unique()
    
    if 'TUN' in unique_areas:
        return df[df['REF_AREA'] == 'TUN'].copy()
    
    return df


def get_summary(df: pd.DataFrame) -> dict:
    """ Retourne un résumé des données.
    
    Args:
        df: DataFrame avec les données.
        
    Returns:
        Dictionnaire avec shape, missing values et dtypes.
    """
    return {
        'shape': df.shape,
        'missing': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.to_dict()
    }