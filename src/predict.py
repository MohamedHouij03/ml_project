"""Prediction utilities for the infant mortality project."""

import joblib
import json
import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path


def load_model(path: str) -> object:
    """Charge un modèle depuis un fichier joblib.
    
    Args:
        path: Chemin vers le fichier modèle.
        
    Returns:
        Pipeline du modèle chargé.
    """
    return joblib.load(path)


def load_metadata(path: str) -> dict:
    """Charge les métadonnées depuis un fichier JSON.
    
    Args:
        path: Chemin vers le fichier JSON.
        
    Returns:
        Dictionnaire des métadonnées.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def predict(model_pipeline: object, X_new: pd.DataFrame) -> np.ndarray:
    """Fait des prédictions sur de nouvelles données.
    
    Args:
        model_pipeline: Pipeline du modèle entraîné.
        X_new: DataFrame des nouvelles données.
        
    Returns:
        Array des prédictions.
    """
    return model_pipeline.predict(X_new)


def predict_with_context(
    model_pipeline: object, 
    X_new: pd.DataFrame
) -> pd.DataFrame:
    """Fait des prédictions avec le contexte des données d'entrée.
    
    Args:
        model_pipeline: Pipeline du modèle entraîné.
        X_new: DataFrame des nouvelles données.
        
    Returns:
        DataFrame avec prédictions et contexte.
    """
    predictions = predict(model_pipeline, X_new)
    
    result = X_new.copy()
    result['Predicted'] = predictions
    
    if 'Unit of measure' in X_new.columns:
        units = X_new['Unit of measure'].values
        result['Interpretation'] = [
            format_prediction(pred, unit) 
            for pred, unit in zip(predictions, units)
        ]
    
    return result


def format_prediction(value: float, unit: str) -> str:
    """Formate la prédiction selon l'unité de mesure.
    
    Args:
        value: Valeur prédite.
        unit: Unité de mesure.
        
    Returns:
        Chaîne formatée.
    """
    if pd.isna(value):
        return "N/A"
    
    unit_mapping = {
        'D_PER_1000_B': f"{value:.2f} décès pour 1000 naissances vivantes",
        'PER_1000': f"{value:.2f} pour 1000",
        'NR': f"{value:.2f}"
    }
    
    return unit_mapping.get(unit, f"{value:.2f}")