"""
Script d'entrainement ML pour la prediction de mortalite infantile.

Execution directe : python build_model.py
"""

import os
import warnings
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

# Ignore les avertissements mineurs OneHotEncoder
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. CHARGER LES DONNEES
# ---------------------------------------------------------
data_path = "data/UNICEF-CME_DF_2021_WQ-1.0-download (1).csv"

print(f"Chargement du jeu de donnees : {data_path}")
dataframe = pd.read_csv(data_path)

print(f"Nombre total de lignes chargees : {len(dataframe)}")

# ---------------------------------------------------------
# 2. SELECTION DES VARIABLES
# ---------------------------------------------------------
# ---------------------------------------------------------
# 2.1 NORMALISATION DES NOMS DE COLONNES
# ---------------------------------------------------------
# Renommer les colonnes en format standard
column_rename_map = {
    "Unit of measure": "UNIT_MEASURE",
    "Series Name": "SERIES_NAME",
}

# Eviter les doublons si les noms standard existent deja
for old_name, new_name in column_rename_map.items():
    if old_name in dataframe.columns and new_name in dataframe.columns:
        dataframe = dataframe.drop(columns=[old_name])

columns_to_rename = {
    old_name: new_name
    for old_name, new_name in column_rename_map.items()
    if old_name in dataframe.columns and new_name not in dataframe.columns
}
dataframe = dataframe.rename(columns=columns_to_rename)

# Gestion de securite contre les colonnes dupliquees
if dataframe.columns.duplicated().any():
    duplicate_cols = dataframe.columns[dataframe.columns.duplicated()].tolist()
    print(f"Attention : colonnes dupliquees supprimees : {sorted(set(duplicate_cols))}")
    dataframe = dataframe.loc[:, ~dataframe.columns.duplicated()]

# ---------------------------------------------------------
# 2.2 CIBLE A PREDIRE
# ---------------------------------------------------------
target_column = "OBS_VALUE"

# ---------------------------------------------------------
# 2.3 BLOCS DE FACTEURS TESTES
# ---------------------------------------------------------
# Bloc 1: inclut LOWER_BOUND/UPPER_BOUND pour montrer le leakage
# ---------------------------------------------------------
# 2.3.1 BLOC 1 - AVEC LOWER_BOUND ET UPPER_BOUND
# ---------------------------------------------------------
features_leakage = [
    "Indicator",
    "SEX",
    "REF_DATE",
    "UNIT_MEASURE",
    "SERIES_NAME",
    "LOWER_BOUND",
    "UPPER_BOUND",
]

# ---------------------------------------------------------
# 2.3.2 BLOC 2 - FACTEURS DE BASE
# ---------------------------------------------------------
features_baseline = [
    "Indicator",
    "SEX",
    "REF_DATE",
    "UNIT_MEASURE",
    "SERIES_NAME",
    "SERIES_YEAR",
]

# ---------------------------------------------------------
# 2.3.3 BLOC 3 - AJOUT DE NOUVEAUX FACTEURS
# ---------------------------------------------------------
features_expanded = [
    "Indicator",
    "SEX",
    "REF_DATE",
    "UNIT_MEASURE",
    "SERIES_NAME",
    "OBS_STATUS",
    "SERIES_YEAR",
    "TIME_PERIOD",
    "INTERVAL",
]

# ---------------------------------------------------------
# 2.3.4 LISTE DES BLOCS A EVALUER
# ---------------------------------------------------------
scenario_blocks = [
    ("Bloc 1 - Avec LOWER_BOUND et UPPER_BOUND", features_leakage),
    ("Bloc 2 - Facteurs de base", features_baseline),
    ("Bloc 3 - Ajout de nouveaux facteurs", features_expanded),
]

all_experiment_results = {}
best_pipelines = {}
scenario_data_for_plot = {}
scenario_data_for_metadata = {}

def run_block(scenario_name, scenario_features):
    print("\n" + "=" * 70)
    print(scenario_name)
    print("Facteurs : " + ", ".join(scenario_features))

    needed_columns = scenario_features + [target_column]
    missing_columns = [col for col in needed_columns if col not in dataframe.columns]
    if missing_columns:
        print(f"[SKIP] Colonnes manquantes : {missing_columns}")
        return

    model_data = dataframe[scenario_features + [target_column]].copy()

    # Conversion numerique des colonnes numeriques disponibles
    for col in ["REF_DATE", "SERIES_YEAR", "LOWER_BOUND", "UPPER_BOUND", target_column]:
        if col in model_data.columns:
            model_data[col] = pd.to_numeric(model_data[col], errors="coerce")

    model_data = model_data.dropna(subset=[target_column])

    # Tri chrono quand REF_DATE existe
    if "REF_DATE" in model_data.columns:
        model_data = model_data.dropna(subset=["REF_DATE"])
        model_data = model_data.sort_values(by="REF_DATE")

    target_values = model_data[target_column]
    feature_data = model_data.drop(columns=[target_column])

    numeric_candidates = {"REF_DATE", "SERIES_YEAR", "LOWER_BOUND", "UPPER_BOUND"}
    numeric_features = [col for col in scenario_features if col in numeric_candidates]
    categorical_features = [col for col in scenario_features if col not in numeric_candidates]

    train_features, test_features, train_target, test_target = train_test_split(
        feature_data,
        target_values,
        test_size=0.2,
        shuffle=False,
    )

    print(f"Lignes : {len(model_data)} | Train : {len(train_features)} | Test : {len(test_features)}")

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    scenario_metrics = {}
    scenario_best_rmse = float("inf")
    scenario_best_model_name = None
    scenario_best_pipeline = None
    scenario_best_predictions = None
    scenario_best_mae = None
    scenario_best_r2 = None

    model_specs = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
    }

    for model_name, regressor in model_specs.items():
        model_pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("regressor", regressor),
            ]
        )
        model_pipeline.fit(train_features, train_target)
        predictions = model_pipeline.predict(test_features)

        rmse = float(np.sqrt(mean_squared_error(test_target, predictions)))
        mae = float(mean_absolute_error(test_target, predictions))
        r2 = float(r2_score(test_target, predictions))

        scenario_metrics[model_name] = {
            "RMSE": rmse,
            "MAE": mae,
            "TestR2": r2,
        }

        print(f"  {model_name}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

        if rmse < scenario_best_rmse:
            scenario_best_rmse = rmse
            scenario_best_mae = mae
            scenario_best_r2 = r2
            scenario_best_model_name = model_name
            scenario_best_pipeline = model_pipeline
            scenario_best_predictions = predictions

    print(f"[OK] Meilleur modele du bloc : {scenario_best_model_name}")
    print(f"  RMSE: {scenario_best_rmse:.4f} | MAE: {scenario_best_mae:.4f} | R2: {scenario_best_r2:.4f}")

    all_experiment_results[scenario_name] = {
        "features": scenario_features,
        "rows": len(model_data),
        "train_rows": len(train_features),
        "test_rows": len(test_features),
        "best_model_name": scenario_best_model_name,
        "results": scenario_metrics,
    }

    best_pipelines[scenario_name] = scenario_best_pipeline
    scenario_data_for_plot[scenario_name] = {
        "test_target": test_target,
        "best_predictions": scenario_best_predictions,
        "target_values": target_values,
        "best_model_name": scenario_best_model_name,
    }
    scenario_data_for_metadata[scenario_name] = {
        "model_data": model_data,
        "feature_columns": scenario_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "results": scenario_metrics,
        "best_model_name": scenario_best_model_name,
    }


# ---------------------------------------------------------
# 3A. BLOCK A - AVEC LOWER/UPPER BOUNDS
# ---------------------------------------------------------
run_block("Configuration fuite d'information (avec bornes)", features_leakage)

# ---------------------------------------------------------
# 3B. BLOCK B - SANS LOWER/UPPER BOUNDS (BASELINE)
# ---------------------------------------------------------
run_block("Configuration de base (5 facteurs)", features_baseline)

# ---------------------------------------------------------
# 3C. BLOCK C - FEATURES ETENDUES
# ---------------------------------------------------------
run_block("Configuration etendue (facteurs additionnels)", features_expanded)

# ---------------------------------------------------------
# 3D. COMPARAISON FINALE DES BLOCS
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("Comparaison finale des blocs (meilleur modele de chaque bloc)")
comparison_rows = []
for block_name, block_data in all_experiment_results.items():
    best_name = block_data["best_model_name"]
    best_metrics = block_data["results"][best_name]
    comparison_rows.append(
        {
            "Bloc": block_name,
            "Meilleur modele": best_name,
            "RMSE": round(best_metrics["RMSE"], 4),
            "MAE": round(best_metrics["MAE"], 4),
            "R2": round(best_metrics["TestR2"], 4),
        }
    )
comparison_df = pd.DataFrame(comparison_rows)
print(comparison_df.to_string(index=False))

# ---------------------------------------------------------
# 3.5 CHOIX DU BLOC POUR L'APPLICATION
# ---------------------------------------------------------
# On garde le bloc baseline comme configuration principale (resultat autour de R2 ~ 0.88)
app_scenario_name = "Configuration de base (5 facteurs)"
if app_scenario_name not in best_pipelines:
    raise ValueError(f"Le scenario app '{app_scenario_name}' n'a pas ete calcule.")

best_pipeline = best_pipelines[app_scenario_name]
app_context = scenario_data_for_metadata[app_scenario_name]
app_plot = scenario_data_for_plot[app_scenario_name]

print("\n" + "=" * 70)
print(f"Bloc retenu pour l'application : {app_scenario_name}")
print(f"Modele retenu : {app_context['best_model_name']}")

# ---------------------------------------------------------
# 4. SAUVEGARDE DU MODELE ET DES METADONNEES
# ---------------------------------------------------------
artifacts_dir = "ml_artifacts"
os.makedirs(artifacts_dir, exist_ok=True)

model_path = os.path.join(artifacts_dir, "best_model.joblib")
joblib.dump(best_pipeline, model_path)
print(f"[OK] Modele sauvegarde : {model_path}")

categorical_options = {
    col: sorted(app_context["model_data"][col].dropna().unique().tolist())
    for col in app_context["categorical_features"]
}
ref_date_bounds = {
    "min": float(app_context["model_data"]["REF_DATE"].min()),
    "max": float(app_context["model_data"]["REF_DATE"].max()),
}

metadata = {
    "best_model_name": app_context["best_model_name"],
    "feature_columns": app_context["feature_columns"],
    "numeric_features": app_context["numeric_features"],
    "categorical_options": categorical_options,
    "ref_date_bounds": ref_date_bounds,
    "results": app_context["results"],
    "all_experiments": all_experiment_results,
}

metadata_path = os.path.join(artifacts_dir, "metadata.json")
with open(metadata_path, "w", encoding="utf-8") as f:
    f.write(json.dumps(metadata, indent=2))
print(f"[OK] Metadonnees sauvegardees : {metadata_path}")

# ---------------------------------------------------------
# 5. VISUALISATION DU BLOC RETENU
# ---------------------------------------------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 5))
plt.scatter(app_plot["test_target"], app_plot["best_predictions"], alpha=0.5, color="blue")
plt.plot(
    [app_plot["target_values"].min(), app_plot["target_values"].max()],
    [app_plot["target_values"].min(), app_plot["target_values"].max()],
    "r--",
    lw=2,
)
plt.xlabel("OBS_VALUE reel")
plt.ylabel(f"OBS_VALUE predit ({app_plot['best_model_name']})")
plt.title("Valeurs reelles vs valeurs predites")

plt.tight_layout()
plt.show()
