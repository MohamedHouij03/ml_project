import json

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


def load_data(path):
    df = pd.read_csv(path)
    rename_map = {
        "Unit of measure": "UNIT_MEASURE",
        "Series Name": "SERIES_NAME",
        "Series Year": "SERIES_YEAR",
    }
    df = df.rename(columns=rename_map)
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def evaluate(df, features, name):
    target = "OBS_VALUE"
    use_cols = [c for c in features if c in df.columns]

    for c in ["REF_DATE", "SERIES_YEAR", "OBS_VALUE", "LOWER_BOUND", "UPPER_BOUND"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    d = df[use_cols + [target]].dropna(subset=[target]).copy()
    if "REF_DATE" in d.columns:
        d = d.sort_values("REF_DATE")

    X = d[use_cols]
    y = d[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False,
    )

    numeric_features = [c for c in use_cols if c in ["REF_DATE", "SERIES_YEAR", "LOWER_BOUND", "UPPER_BOUND"]]
    categorical_features = [c for c in use_cols if c not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
    }

    results = {}
    for model_name, reg in models.items():
        pipe = Pipeline(steps=[("pre", pre), ("reg", reg)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        mae = float(mean_absolute_error(y_test, pred))
        r2 = float(r2_score(y_test, pred))
        results[model_name] = {"RMSE": rmse, "MAE": mae, "R2": r2}

    return {
        "scenario": name,
        "features": use_cols,
        "rows": len(d),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "results": results,
    }


def main():
    df = load_data("data/UNICEF-CME_DF_2021_WQ-1.0-download (1).csv")

    base_features = [
        "Indicator",
        "SEX",
        "REF_DATE",
        "UNIT_MEASURE",
        "SERIES_NAME",
    ]

    leakage_features = base_features + ["LOWER_BOUND", "UPPER_BOUND"]

    expanded_features = base_features + [
        "OBS_STATUS",
        "SERIES_YEAR",
        "TIME_PERIOD",
        "INTERVAL",
    ]

    experiments = [
        evaluate(df.copy(), leakage_features, "With LOWER_BOUND and UPPER_BOUND"),
        evaluate(df.copy(), base_features, "Current baseline factors"),
        evaluate(df.copy(), expanded_features, "Adding more candidate factors"),
    ]

    print(json.dumps(experiments, indent=2))


if __name__ == "__main__":
    main()
