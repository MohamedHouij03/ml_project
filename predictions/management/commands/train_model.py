import json
import warnings
from pathlib import Path

import django
import joblib
import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings('ignore')


class Command(BaseCommand):
    help = "Entrainer et sauvegarder le modele ML de prediction"

    def add_arguments(self, parser):
        parser.add_argument(
            '--data',
            type=str,
            default='UNICEF-CME_DF_2021_WQ-1.0-download (1).csv',
            help='Chemin vers le fichier CSV de donnees'
        )
        parser.add_argument(
            '--test-size',
            type=float,
            default=0.2,
            help='Taille du jeu de test (0-1)'
        )

    def handle(self, *args, **options):
        try:
            # Get project paths
            project_root = Path(settings.BASE_DIR)
            data_file = project_root / options['data']
            artifacts_dir = project_root / 'ml_artifacts'
            artifacts_dir.mkdir(exist_ok=True)

            # Load data
            if not data_file.exists():
                raise CommandError(
                    f"Fichier de donnees introuvable : {data_file}\n"
                    f"Verifiez que '{options['data']}' est bien dans {project_root}"
                )

            self.stdout.write(f"Chargement du jeu de donnees : {data_file}")
            df = pd.read_csv(data_file)
            self.stdout.write(self.style.SUCCESS(f"✓ {len(df)} lignes chargees"))

            # Rename columns to standard uppercase to match FIELD_LABELS and views
            rename_map = {
                "Unit of measure": "UNIT_MEASURE",
                "Series Name": "SERIES_NAME",
            }
            df = df.rename(columns=rename_map)

            # Select features
            features_to_keep = [
                "Indicator", "SEX", "REF_DATE", "UNIT_MEASURE", "SERIES_NAME"
            ]
            target_col = "OBS_VALUE"

            df_model = df[features_to_keep + [target_col]].dropna(subset=[target_col]).copy()
            self.stdout.write(f"✓ Variables selectionnees : {', '.join(features_to_keep)}")

            # Chronological split
            self.stdout.write("Tri chronologique des donnees...")
            # Ensure REF_DATE is numeric before sorting and training
            df_model["REF_DATE"] = pd.to_numeric(df_model["REF_DATE"], errors='coerce')
            df_model = df_model.dropna(subset=["REF_DATE"])
            df_model = df_model.sort_values(by="REF_DATE")

            y = df_model[target_col]
            X = df_model.drop(columns=[target_col])

            numeric_features = ["REF_DATE"]
            categorical_features = ["Indicator", "SEX", "UNIT_MEASURE", "SERIES_NAME"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=options['test_size'], shuffle=False
            )
            self.stdout.write(
                f"✓ Decoupage train/test : {len(X_train)} / {len(X_test)} lignes"
            )

            # Build preprocessor
            numeric_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])

            categorical_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
            ])

            preprocessor = ColumnTransformer(transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ], remainder="drop")

            model_specs = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
                "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
            }

            results = {}
            best_name = None
            best_pipeline = None
            best_rmse = float('inf')
            best_mae = None
            best_r2 = None

            self.stdout.write("\n" + self.style.SUCCESS("=== Entrainement des 3 modeles ==="))
            for name, regressor in model_specs.items():
                pipe = Pipeline(steps=[
                    ("preprocess", preprocessor),
                    ("regressor", regressor),
                ])
                pipe.fit(X_train, y_train)

                y_pred = pipe.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                test_r2 = r2_score(y_test, y_pred)
                results[name] = {"RMSE": rmse, "MAE": mae, "TestR2": test_r2}

                self.stdout.write(
                    f"• {name:20s} | RMSE: {rmse:8.4f} | MAE: {mae:8.4f} | R²: {test_r2:7.4f}"
                )

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_mae = mae
                    best_r2 = test_r2
                    best_name = name
                    best_pipeline = pipe

            # Save model
            model_path = artifacts_dir / "best_model.joblib"
            joblib.dump(best_pipeline, model_path)
            self.stdout.write(self.style.SUCCESS(f"\n✓ Meilleur modele sauvegarde ({best_name})"))
            self.stdout.write(f"  Emplacement : {model_path}")

            # Extract unique values for categorical features and bounds for REF_DATE
            categorical_options = {
                col: sorted(df_model[col].dropna().unique().tolist())
                for col in categorical_features
            }
            ref_date_bounds = {
                "min": float(df_model["REF_DATE"].min()),
                "max": float(df_model["REF_DATE"].max()),
            }

            # Save metadata
            metadata = {
                "best_model_name": best_name,
                "feature_columns": features_to_keep,
                "numeric_features": numeric_features,
                "categorical_options": categorical_options,
                "ref_date_bounds": ref_date_bounds,
                "results": results,
            }

            metadata_path = artifacts_dir / "metadata.json"
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            self.stdout.write(self.style.SUCCESS("✓ Metadonnees sauvegardees"))
            self.stdout.write(f"  Emplacement : {metadata_path}")

            self.stdout.write(self.style.SUCCESS("\n✓ Entrainement termine !"))
            self.stdout.write(f"\nMeilleur modele : {best_name}")
            self.stdout.write(f"  RMSE: {best_rmse:.4f}")
            self.stdout.write(f"  MAE: {best_mae:.4f}")
            self.stdout.write(f"  R²: {best_r2:.4f}")

        except Exception as e:
            raise CommandError(f"Erreur pendant l'entrainement du modele : {str(e)}")
