"""
Génération de graphiques Matplotlib (PNG) pour la page Visualisations.
"""
from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _data_csv_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "data" / "UNICEF-CME_DF_2021_WQ-1.0-download (1).csv"
    )


def _fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#f1f5f9")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _load_dataset() -> pd.DataFrame:
    path = _data_csv_path()
    if not path.exists():
        raise FileNotFoundError("Fichier CSV introuvable.")

    df = pd.read_csv(path)
    rename_map = {
        "Unit of measure": "UNIT_MEASURE",
        "Series Name": "SERIES_NAME",
    }
    return df.rename(columns=rename_map)


def chart_indicator_frequency() -> bytes:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    try:
        df = _load_dataset()
    except Exception:
        ax.text(0.5, 0.5, "Impossible de charger le fichier CSV.", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        fig.tight_layout()
        return _fig_to_png(fig)

    if "Indicator" not in df.columns:
        ax.text(0.5, 0.5, "Colonne Indicator absente.", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        return _fig_to_png(fig)

    counts = df["Indicator"].dropna().astype(str).value_counts().head(10)
    if counts.empty:
        ax.text(0.5, 0.5, "Aucune valeur Indicator disponible.", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        return _fig_to_png(fig)

    ax.barh(counts.index[::-1], counts.values[::-1], color="#14b8a6", edgecolor="#0f172a", linewidth=0.5)
    ax.set_xlabel("Nombre de lignes dans le CSV")
    ax.set_title("Top 10 indicateurs les plus présents")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return _fig_to_png(fig)


def chart_yearly_obs_mean() -> bytes:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    try:
        df = _load_dataset()
    except Exception:
        ax.text(0.5, 0.5, "Impossible de charger le fichier CSV.", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        fig.tight_layout()
        return _fig_to_png(fig)

    needed = {"REF_DATE", "OBS_VALUE"}
    if not needed.issubset(df.columns):
        ax.text(0.5, 0.5, "Colonnes REF_DATE/OBS_VALUE absentes.", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        return _fig_to_png(fig)

    work = df[["REF_DATE", "OBS_VALUE"]].copy()
    work["REF_DATE"] = pd.to_numeric(work["REF_DATE"], errors="coerce")
    work["OBS_VALUE"] = pd.to_numeric(work["OBS_VALUE"], errors="coerce")
    work = work.dropna(subset=["REF_DATE", "OBS_VALUE"])

    if work.empty:
        ax.text(0.5, 0.5, "Aucune valeur numerique exploitable.", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        return _fig_to_png(fig)

    yearly = work.groupby("REF_DATE", as_index=False)["OBS_VALUE"].mean().sort_values("REF_DATE")
    ax.plot(yearly["REF_DATE"], yearly["OBS_VALUE"], color="#0284c7", linewidth=2.2)
    ax.scatter(yearly["REF_DATE"], yearly["OBS_VALUE"], color="#0284c7", s=18)
    ax.set_xlabel("Annee de reference")
    ax.set_ylabel("Moyenne OBS_VALUE")
    ax.set_title("Evolution annuelle moyenne (OBS_VALUE)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return _fig_to_png(fig)


def chart_sex_distribution() -> bytes:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    try:
        df = _load_dataset()
    except Exception:
        ax.text(0.5, 0.5, "Impossible de charger le fichier CSV.", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        fig.tight_layout()
        return _fig_to_png(fig)

    if "SEX" not in df.columns:
        ax.text(0.5, 0.5, "Colonne SEX absente.", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        return _fig_to_png(fig)

    labels_map = {"F": "Feminin", "M": "Masculin", "B": "Deux sexes"}
    counts = df["SEX"].dropna().astype(str).map(lambda v: labels_map.get(v, v)).value_counts()
    if counts.empty:
        ax.text(0.5, 0.5, "Aucune valeur SEX disponible.", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        return _fig_to_png(fig)

    ax.bar(counts.index, counts.values, color=["#f59e0b", "#7c3aed", "#14b8a6", "#64748b"])
    ax.set_ylabel("Nombre de lignes")
    ax.set_title("Repartition des donnees par sexe")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return _fig_to_png(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#6366f1" if "Linéaire" in L else "#f59e0b" if "Arbre" in L else "#3dd6c3" for L in labels]
    ax.barh(labels, values, color=colors, edgecolor="#334155", linewidth=0.5)
    ax.set_xlabel("R² sur le jeu de test (plus proche de 1 = meilleur)")
    ax.set_title("Comparaison des modèles — qualité d’ajustement (R²)")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return _fig_to_png(fig)


def chart_rmse_comparison() -> bytes:
    _, meta = load_ml_artifacts()
    results = meta.get("results") or {}
    rows = [
        (_MODEL_LABELS.get(k, k), float(v["RMSE"]))
        for k, v in results.items()
    ]
    rows.sort(key=lambda x: x[1], reverse=True)
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#6366f1" if "Linéaire" in L else "#f59e0b" if "Arbre" in L else "#3dd6c3" for L in labels]
    ax.barh(labels, values, color=colors, edgecolor="#334155", linewidth=0.5)
    ax.set_xlabel("RMSE (plus bas = meilleur)")
    ax.set_title("Comparaison des modèles — erreur sur le jeu de test (RMSE)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return _fig_to_png(fig)


def chart_obs_value_distribution() -> bytes:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    try:
        df = _load_dataset()
    except Exception:
        ax.text(0.5, 0.5, "Impossible de charger le fichier CSV.", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        fig.tight_layout()
        return _fig_to_png(fig)

    if "OBS_VALUE" not in df.columns:
        ax.text(0.5, 0.5, "Colonne OBS_VALUE absente.", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        return _fig_to_png(fig)

    s = pd.to_numeric(df["OBS_VALUE"], errors="coerce").dropna()
    s = s[np.isfinite(s)]
    if len(s) == 0:
        ax.text(0.5, 0.5, "Aucune valeur numerique OBS_VALUE.", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        return _fig_to_png(fig)

    ax.hist(s, bins=40, color="#3dd6c3", edgecolor="#0f172a", alpha=0.85)
    ax.set_xlabel("Valeur observée (variable cible des données)")
    ax.set_ylabel("Fréquence")
    ax.set_title("Distribution des valeurs cible dans le jeu de données")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _fig_to_png(fig)


CHART_REGISTRY = {
    "indicator_freq": ("Top indicateurs du CSV", chart_indicator_frequency),
    "yearly_obs": ("Evolution annuelle de OBS_VALUE", chart_yearly_obs_mean),
    "sex_split": ("Repartition des donnees par sexe", chart_sex_distribution),
    "obs_dist": ("Distribution de la variable cible", chart_obs_value_distribution),
}


def get_chart_png(chart_name: str) -> bytes:
    if chart_name not in CHART_REGISTRY:
        raise KeyError(chart_name)
    return CHART_REGISTRY[chart_name][1]()
