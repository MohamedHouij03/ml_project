# 🇹🇳 Prédiction de la Mortalité Infantile (UNICEF)

**🔗 Website:** https://mlproject-production-f420.up.railway.app/

## 👤 Auteur

**Mohamed Houij**

---

## 📊 Description du dataset et source

Ce projet s'appuie sur les données officielles de l'UNICEF, issues de la base **CME (Child Mortality Estimates) 2021**.
Le jeu de données (`UNICEF-CME_DF_2021.csv`) recense les taux de mortalité infantile et néonatale à travers le temps. Il contient des variables géographiques, démographiques (Sexe, Indicateur de mortalité) et temporelles (Année de référence), ainsi que des intervalles de confiance statistiques (bornes inférieures et supérieures).

## 🎯 Problématique abordée

**Comment prédire de manière fiable l'évolution du taux de mortalité infantile tout en évitant les biais de modélisation ?**

L'enjeu principal de ce projet n'est pas seulement d'appliquer un algorithme, mais de garantir l'intégrité scientifique du modèle. Nous avons notamment identifié et traité un problème de **Fuite de Données (Data Leakage)** : l'inclusion des bornes mathématiques (intervalles de confiance) faussait l'apprentissage en donnant artificiellement la réponse au modèle. Notre objectif est de construire un modèle prédictif basé uniquement sur des variables réalistes et utilisables sur le terrain.

## Structure

```
.
├── data/                    # UNICEF data
├── notebooks/               # Jupyter notebooks
├── src/                     # ML pipeline code
│   ├── config.py            # Configuration
│   ├── data_loader.py      # Data loading
│   ├── train.py             # Model training
│   └── main.py             # Pipeline
├── predictions/             # Django app
├── templates/               # HTML templates
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### Notebooks

```bash
jupyter lab notebooks/01_EDA.ipynb
jupyter lab notebooks/02_Modeling.ipynb
```

### Django

```bash
python manage.py runserver
```

Accedez sur `http://127.0.0.1:8000/`

### Script

```bash
python src/main.py
```

## Configuration

```python
# src/config.py
DATA_PATH = 'data/UNICEF-CME_DF_2021_WQ-1.0-download (1).csv'
TARGET = 'OBS_VALUE'
DATE_COLUMN = 'REF_DATE'

MODELS = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=3, min_samples_leaf=20, random_state=42),
    'Random Forest': RandomForestRegressor(max_depth=3, min_samples_leaf=20, n_estimators=20, random_state=42)
}
```

## 📈 Résumé des résultats obtenus

Nous avons testé et comparé trois algorithmes (Régression Linéaire, Arbre de Décision, et Forêt Aléatoire) sur une configuration de données assainie et enrichie (ajout d'une variable `Décennie`).

- **Régression Linéaire :** Échec de la prédiction (RMSE très élevé). Le modèle s'est avéré incapable de gérer la non-linéarité de la baisse de la mortalité infantile et a été pénalisé par le grand nombre de variables catégorielles encodées.
- **Arbre de Décision :** De meilleures performances, mais une tendance à l'instabilité et au sur-apprentissage.
- **Random Forest (Modèle Retenu) :** Il s'est imposé comme le modèle le plus robuste. Après optimisation des hyperparamètres, il a atteint un score de précision (R²) d'environ **0.88**. L'analyse de ses résidus confirme qu'il est stable et mathématiquement honnête, prouvant qu'il est possible de prédire la mortalité infantile avec une grande précision sans recourir à la triche statistique (Data Leakage).
