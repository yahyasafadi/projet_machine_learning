# KNN — Reconnaissance de variétés de haricots

> **Mini-projet 02** — Cours de Machine Learning · INSEA · Année universitaire 2026-2027

Partie **KNN (K plus proches voisins)** du mini-projet d'équipe sur le dataset **UCI Dry Bean** (13 611 grains, 16 mesures morphologiques, 7 variétés). Les autres modèles de l'équipe (Régression logistique, Arbre de décision, Random Forest, SVM) se trouvent dans les dossiers frères à la racine du dépôt.

---

## 🎯 Résultats principaux

| Indicateur | Valeur |
|---|---:|
| Meilleurs hyperparamètres | `n_neighbors=9`, `metric="euclidean"`, `weights="distance"` |
| **F1-score macro (test)** | **0.9300** |
| Accuracy (test) | 0.9174 |
| F1-score macro (CV train) | 0.9382 |
| Gain apporté par la standardisation | **+0.22 F1 macro** |
| Baseline naïve (DummyClassifier `stratified`) | 0.1470 |

➡️ Le modèle est **~6× au-dessus du hasard**, avec un écart train/test de seulement **0.008** → bonne généralisation, pas de surapprentissage marqué.

---

## 📁 Contenu du dossier

### Livrables officiels (demandés par le prof)

| Fichier | Description |
|---|---|
| [`knn_dry_bean.ipynb`](knn_dry_bean.ipynb) | **Notebook Jupyter** avec code + narration + résultats intégrés (11 sections) |
| [`Rapport_KNN.docx`](Rapport_KNN.docx) | **Rapport** 4 pages (contexte → méthodologie → résultats → conclusion) |
| [`Presentation_KNN.pptx`](Presentation_KNN.pptx) | **Slides** 10 pages pour la soutenance orale |

### Code et données

| Fichier | Description |
|---|---|
| [`knn_dry_bean.py`](knn_dry_bean.py) | Script Python principal — tout le pipeline en une commande |
| [`Dry_Bean_Dataset.csv`](Dry_Bean_Dataset.csv) | Dataset officiel UCI (13 611 × 17) |
| [`requirements.txt`](requirements.txt) | Dépendances Python à installer |

### Documentation

| Fichier | Description |
|---|---|
| [`GUIDE_UTILISATION.md`](GUIDE_UTILISATION.md) | Comment installer, lancer, dépanner, fusionner dans le notebook commun |
| [`EXPLICATIONS_DETAILLEES.md`](EXPLICATIONS_DETAILLEES.md) | Cours pas-à-pas pour débutant en ML (19 chapitres, du concept de ML jusqu'à l'analyse d'erreurs) |

### Résultats générés (dossier [`resultats/`](resultats/))

**Étapes du protocole (règle d'or de l'équipe) :**

| Fichier | Contenu |
|---|---|
| `01_statistiques_descriptives.csv` + 3 PNG | Exploration : stats, effectifs, distributions, corrélations |
| `03_experience_standardisation.csv` / `.png` | Impact de `StandardScaler` (avec vs sans) |
| `04_resultats_gridsearch.csv` + 2 PNG | 60 combinaisons k × distance × poids |
| `05_metriques_globales.csv` | Accuracy, précision, rappel, F1 macro |
| `05_f1_par_variete.csv` / `.png` | F1 par classe |
| `05_confusions_principales.csv` | Top des confusions |
| `05_matrice_confusion.png` | Matrice de confusion (effectifs + normalisée) |

**Bonus scientifiques :**

| Fichier | Contenu |
|---|---|
| `06_baseline_dummy.csv` | Comparaison avec un `DummyClassifier` |
| `07_pca_2d.png` | Projection ACP 2D des 7 classes (82 % de variance) |
| `08_learning_curve.csv` / `.png` | Courbe d'apprentissage |
| `09_temps_prediction.csv` | Mesure du temps d'inférence |
| `10_erreurs_analyse.csv` | Analyse des grains mal classés |

---

## 🚀 Reproduction en 2 commandes

```bash
pip install -r requirements.txt
python knn_dry_bean.py --data Dry_Bean_Dataset.csv --out resultats
```

Durée totale : **25 à 40 secondes** sur un ordinateur portable récent.

Toutes les sorties (PNG, CSV) sont régénérées à l'identique grâce à `random_state=42`.

---

## 🧭 Méthodologie — Règle d'or de l'équipe

Toute l'équipe applique le même protocole pour garantir la comparabilité des 5 modèles :

- `random_state = 42` partout (split, CV, modèles)
- Split **80 % train / 20 % test stratifié** (`stratify=y`)
- Standardisation encapsulée dans un `Pipeline` sklearn (pas de fuite train → test)
- Validation croisée : `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` sur le train uniquement
- Métrique reine : **F1-score macro** (adaptée au déséquilibre des 7 classes)

### Tuning — grille testée

- `n_neighbors` : {1, 3, 5, 7, 9, 11, 15, 21, 31, 41}
- `metric` : {euclidean, manhattan, chebyshev}
- `weights` : {uniform, distance}

Soit **60 combinaisons × 5 plis = 300 entraînements**, en 24 secondes sur un CPU standard.

---

## 🔍 Principales conclusions

- **BOMBAY parfaitement reconnu** (F1 = 1.000) : ses grains, nettement plus gros que les autres, forment un îlot complètement séparé dans l'espace des features (visible sur la projection ACP).
- **Confusion DERMASON ↔ SIRA** (105 erreurs mutuelles) : ces deux variétés ont une morphologie très proche. Un modèle ne peut pas les séparer avec les seules mesures géométriques — il faudrait ajouter couleur et/ou texture.
- **Toutes les 7 classes dépassent F1 = 0.85**, aucune classe sacrifiée.
- **Plus de données n'aiderait pas** : la courbe d'apprentissage plafonne à 0.938. Les pistes d'amélioration sont donc : nouvelles features, modèles non linéaires (SVM RBF, gradient boosting), ou feature engineering.

---

## 🛠 Technologies

- **Python 3.10+**
- **scikit-learn** (KNN, Pipeline, GridSearchCV, StratifiedKFold, PCA, DummyClassifier, learning_curve)
- **pandas**, **numpy** (manipulation des données)
- **matplotlib**, **seaborn** (visualisations)
- **Jupyter** (notebook interactif)

---

## 📚 Pour aller plus loin

- **Pour relancer le code** → voir [`GUIDE_UTILISATION.md`](GUIDE_UTILISATION.md)
- **Pour comprendre chaque étape (débutant en ML)** → voir [`EXPLICATIONS_DETAILLEES.md`](EXPLICATIONS_DETAILLEES.md), qui part de "c'est quoi le machine learning" jusqu'à "comment juger si un modèle est bon ou pas"
- **Dataset source** : [UCI Machine Learning Repository — Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean)

---

