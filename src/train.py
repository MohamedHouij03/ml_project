"""Model training utilities for the infant mortality project."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Set
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
try:
    from src import config
except ModuleNotFoundError:
    import config


def build_preprocessor(features: list, numeric_cols: Set[str]) -> ColumnTransformer:
    """Construit le préprocesseur avec ColumnTransformer.
    
    Args:
        features: Liste des noms de features.
        numeric_cols: Ensemble des colonnes numériques.
        
    Returns:
        ColumnTransformer configuré.
    """
    numeric_features = [f for f in features if f in numeric_cols]
    categorical_features = [f for f in features if f not in numeric_cols]
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    transformers = []
    if numeric_features:
        transformers.append(('num', numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(('cat', categorical_transformer, categorical_features))
    
    if not transformers:
        raise ValueError("Aucune feature trouvée")
    
    return ColumnTransformer(transformers=transformers, remainder='drop')


def temporal_split(df: pd.DataFrame, date_col: str = 'REF_DATE', test_size: float = 0.2) -> tuple:
    """Séparation temporelle train/test pour éviter data leakage.
    
    Args:
        df: DataFrame avec les données.
        date_col: Nom de la colonne de date/année.
        test_size: Proportion des données pour le test (20% par défaut).
        
    Returns:
        Tuple (X_train, X_test, y_train, y_test).
    """
    years = sorted(df[date_col].unique())
    split_idx = int(len(years) * (1 - test_size))
    split_year = years[split_idx]
    
    train_mask = df[date_col] <= split_year
    test_mask = df[date_col] > split_year
    
    return train_mask, test_mask, split_year


def train_models(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    preprocessor: ColumnTransformer,
    models: Dict[str, Any]
) -> Dict[str, Pipeline]:
    """Entraîne tous les modèles.
    
    Args:
        X_train: DataFrame d'entraînement des features.
        y_train: Series d'entraînement de la cible.
        preprocessor: ColumnTransformer pour le prétraitement.
        models: Dic

onnenaire des modèles à entraîner.
        
    Returns:
        Dictionnaire des pipelines entraînés.
    """
    trained_pipelines = {}
    
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline
    
    return trained_pipelines


def evaluate_models(
    pipelines: Dict[str, Pipeline], 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> pd.DataFrame:
    """Évalue tous les modèles.
    
    Args:
        pipelines: Dictionnaire des pipelines entraînés.
        X_test: DataFrame de test des features.
        y_test: Series de test de la cible.
        
    Returns:
        DataFrame avec les métriques d'évaluation.
    """
    results = []
    
    for name, pipeline in pipelines.items():
        y_pred = pipeline.predict(X_test)
        
        mae = np.mean(np.abs(y_test - y_pred))
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
        
        mask = y_test != 0
        if mask.any():
            mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
        else:
            mape = np.nan
        
        results.append({
            'Model': name,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        })
    
    return pd.DataFrame(results)


def train_best_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    preprocessor: ColumnTransformer,
    model_name: str,
    param_grid: Optional[Dict[str, Any]] = None
) -> tuple:
    """Entraîne le meilleur modèle avec GridSearchCV.
    
    Args:
        X_train: DataFrame d'entraînement des features.
        y_train: Series d'entraînement de la cible.
        preprocessor: ColumnTransformer pour le prétraitement.
        model_name: Nom du modèle à entraîner.
        param_grid: Grille de paramètres pour GridSearchCV.
        
    Returns:
        Tuple (best_pipeline, best_params).
    """
    if model_name not in config.MODELS:
        raise ValueError(f"Modèle {model_name} non trouvé dans config.MODELS")
    
    model = config.MODELS[model_name]
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    if param_grid is not None:
        pipeline = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=5, 
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        pipeline.fit(X_train, y_train)
        
        return pipeline.best_estimator_, pipeline.best_params_
    else:
        pipeline.fit(X_train, y_train)
        
        return pipeline, {}