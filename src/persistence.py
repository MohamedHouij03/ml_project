"""Persistence utilities for saving and loading model artifacts."""

import joblib
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional


def save_model(pipeline, path: str):
    """Sauvegarde le modèle avec joblib.
    
    Args:
        pipeline: Pipeline à sauvegarder.
        path: Chemin vers le fichier.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def save_metadata(metadata: dict, path: str):
    """Sauvegarde les métadonnées en JSON.
    
    Args:
        metadata: Métadonnées à sauvegarder.
        path: Chemin vers le fichier.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def save_results(results_df: pd.DataFrame, path: str):
    """Sauvegarde les résultats en CSV.
    
    Args:
        results_df: DataFrame des résultats.
        path: Chemin vers le fichier.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(path, index=False)


def save_plot(fig, path: str):
    """Sauvegarde le graphique matplotlib.
    
    Args:
        fig: Figure matplotlib.
        path: Chemin vers le fichier.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def load_model(path: str):
    """Charge un modèle depuis un fichier joblib.
    
    Args:
        path: Chemin vers le fichier.
        
    Returns:
        Pipeline chargé.
    """
    return joblib.load(path)


def load_metadata(path: str) -> dict:
    """Charge les métadonnées depuis un fichier JSON.
    
    Args:
        path: Chemin vers le fichier.
        
    Returns:
        Métadonnées chargées.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)