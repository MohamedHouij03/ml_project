# 🇹🇳 Prédiction de la Mortalité Infantile (UNICEF)

## 👤 Auteur
**[Ton Nom / Prénom]**

---

## 📊 Description du dataset et source
Ce projet s'appuie sur les données officielles de l'UNICEF, issues de la base **CME (Child Mortality Estimates) 2021**. 
Le jeu de données (`UNICEF-CME_DF_2021.csv`) recense les taux de mortalité infantile et néonatale à travers le temps. Il contient des variables géographiques, démographiques (Sexe, Indicateur de mortalité) et temporelles (Année de référence), ainsi que des intervalles de confiance statistiques (bornes inférieures et supérieures).

---

## 🎯 Problématique abordée
**Comment prédire de manière fiable l'évolution du taux de mortalité infantile tout en évitant les biais de modélisation ?**

L'enjeu principal de ce projet n'est pas seulement d'appliquer un algorithme, mais de garantir l'intégrité scientifique du modèle. Nous avons notamment identifié et traité un problème de **Fuite de Données (Data Leakage)** : l'inclusion des bornes mathématiques (intervalles de confiance) faussait l'apprentissage en donnant artificiellement la réponse au modèle. Notre objectif est de construire un modèle prédictif basé uniquement sur des variables réalistes et utilisables sur le terrain.

---

## ⚙️ Guide d'installation

Pour reproduire cet environnement et exécuter les notebooks sur votre machine locale, suivez ces étapes :

**1. Cloner le projet :**
```bash
git clone [https://github.com/votre-nom/infant-mortality-prediction.git](https://github.com/votre-nom/infant-mortality-prediction.git)
cd infant-mortality-prediction
```

**2. Créer et activer un environnement virtuel :**
* Sur Windows :
```bash
python -m venv .venv
.venv\Scripts\activate
```
* Sur macOS/Linux :
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**3. Installer les dépendances :**
```bash
pip install pandas numpy scikit-learn matplotlib jupyter
# ou via un fichier : pip install -r requirements.txt
```

**4. Lancer Jupyter Notebook :**
```bash
jupyter notebook
```
*Naviguez ensuite dans l'interface pour ouvrir les fichiers `01_EDA.ipynb`, `03_brouillon.ipynb` puis `02_Modeling.ipynb`.*

---

## 📈 Résumé des résultats obtenus

Nous avons testé et comparé trois algorithmes (Régression Linéaire, Arbre de Décision, et Forêt Aléatoire) sur une configuration de données assainie et enrichie (ajout d'une variable `Décennie`).

* **Régression Linéaire :** Échec de la prédiction (RMSE très élevé). Le modèle s'est avéré incapable de gérer la non-linéarité de la baisse de la mortalité infantile et a été pénalisé par le grand nombre de variables catégorielles encodées.
* **Arbre de Décision :** De meilleures performances, mais une tendance à l'instabilité et au sur-apprentissage.
* **Random Forest (Modèle Retenu) :** Il s'est imposé comme le modèle le plus robuste. Après optimisation des hyperparamètres, il a atteint un score de précision (R²) d'environ **0.88**. L'analyse de ses résidus confirme qu'il est stable et mathématiquement honnête, prouvant qu'il est possible de prédire la mortalité infantile avec une grande précision sans recourir à la triche statistique (Data Leakage).