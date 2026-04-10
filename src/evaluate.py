"""Model evaluation utilities for the infant mortality project."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold


def get_feature_importance(pipeline: Pipeline, feature_names: list) -> pd.Series:
    """Extrait les importances des features depuis RF ou GB.
    
    Args:
        pipeline: Pipeline entraîné.
        feature_names: Liste des noms de features.
        
    Returns:
        Series avec les importances des features.
    """
    model = pipeline.named_steps['model']
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        if hasattr(model, 'named_steps') and 'preprocessor' in pipeline.named_steps:
            preprocessor = pipeline.named_steps['preprocessor']
            
            if hasattr(preprocessor, 'transformers_'):
                cat_transformer = None
                for name, transformer, cols in preprocessor.transformers_:
                    if name == 'cat':
                        cat_transformer = transformer
                        break
                
                if cat_transformer is not None:
                    encoder = cat_transformer.named_steps['encoder']
                    if hasattr(encoder, 'get_feature_names_out'):
                        cat_features = encoder.get_feature_names_out(cols)
                        all_features = list(feature_names) + list(cat_features)
                    else:
                        all_features = feature_names
                else:
                    all_features = feature_names
            else:
                all_features = feature_names
        else:
            all_features = feature_names
        
        return pd.Series(importances, index=all_features[:len(importances)])
    else:
        raise ValueError("Le modèle ne supporte pas feature_importances_")


def get_predictions(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Fait des prédictions.
    
    Args:
        pipeline: Pipeline entraîné.
        X: DataFrame des features.
        
    Returns:
        Array des prédictions.
    """
    return pipeline.predict(X)


def get_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Calcule les résidus.
    
    Args:
        y_true: Valeurs réelles.
        y_pred: Valeurs prédites.
        
    Returns:
        Array des résidus.
    """
    return y_true - y_pred


def plot_actual_vs_predicted(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    save_path: Optional[str] = None
):
    """Trace le graphique actual vs predicted.
    
    Args:
        y_true: Valeurs réelles.
        y_pred: Valeurs prédites.
        save_path: Chemin pour sauvegarder le graphique.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.title('Actual vs Predicted')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_residuals(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    save_path: Optional[str] = None
):
    """Trace le graphique des résidus.
    
    Args:
        y_true: Valeurs réelles.
        y_pred: Valeurs prédites.
        save_path: Chemin pour sauvegarder le graphique.
    """
    residuals = get_residuals(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_residual_distribution(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    save_path: Optional[str] = None
):
    """Trace l'histogramme des résidus.
    
    Args:
        y_true: Valeurs réelles.
        y_pred: Valeurs prédites.
        save_path: Chemin pour sauvegarder le graphique.
    """
    residuals = get_residuals(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=30, edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_feature_importance(
    importances: pd.Series, 
    save_path: Optional[str] = None
):
    """Trace le graphique des importances des features.
    
    Args:
        importances: Series avec les importances.
        save_path: Chemin pour sauvegarder le graphique.
    """
    importances = importances.sort_values(ascending=True)
    
    plt.figure(figsize=(8, 6))
    plt.barh(importances.index, importances.values)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_error_by_feature(
    X: pd.DataFrame, 
    residuals: np.ndarray, 
    feature_name: str, 
    save_path: Optional[str] = None
):
    """Trace les résidus en fonction d'une feature.
    
    Args:
        X: DataFrame des features.
        residuals: Array des résidus.
        feature_name: Nom de la feature.
        save_path: Chemin pour sauvegarder le graphique.
    """
    if feature_name not in X.columns:
        raise ValueError(f"Feature {feature_name} non trouvée")
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X[feature_name], residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel(feature_name)
    plt.ylabel('Residuals')
    plt.title(f'Residuals vs {feature_name}')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()


def generate_error_report(
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    y_pred: np.ndarray, 
    top_n: int = 10
) -> pd.DataFrame:
    """Génère un rapport des pires prédictions.
    
    Args:
        X_test: DataFrame de test.
        y_test: Valeurs réelles.
        y_pred: Valeurs prédites.
        top_n: Nombre de pires prédictions à afficher.
        
    Returns:
        DataFrame avec les pires prédictions.
    """
    errors = np.abs(y_test - y_pred)
    
    df_errors = X_test.copy()
    df_errors['Actual'] = y_test.values
    df_errors['Predicted'] = y_pred
    df_errors['Error'] = errors
    df_errors['AbsError'] = errors
    
    df_errors = df_errors.sort_values('AbsError', ascending=False)
    
    return df_errors.head(top_n)


def cross_validate(
    pipeline: Pipeline, 
    X: pd.DataFrame, 
    y: pd.Series, 
    cv: int = 5
) -> dict:
    """Effectue une validation croisée k-fold.
    
    Args:
        pipeline: Pipeline à évaluer.
        X: DataFrame des features.
        y: Series de la cible.
        cv: Nombre de folds.
        
    Returns:
        Dictionnaire avec mean et std du neg RMSE.
    """
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    scores = cross_val_score(
        pipeline, X, y, 
        cv=kfold, 
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    
    return {
        'mean': -scores.mean(),
        'std': scores.std()
    }