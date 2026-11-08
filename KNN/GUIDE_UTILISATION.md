# Guide d'utilisation — Partie KNN (Mini-projet 02)

Ce dossier contient **ma partie KNN** du mini-projet 02 (reconnaissance de 7 variétés de haricots secs à partir de leurs caractéristiques morphologiques). Le code respecte **à la lettre** la règle d'or de l'équipe pour qu'on puisse fusionner les scripts sans friction.

---

## 1. Ce qu'il y a dans le dossier

```
knn_dry_bean/
├── knn_dry_bean.py                  # LE script principal (une seule commande)
├── knn_dry_bean.ipynb               # notebook Jupyter avec résultats intégrés
├── Dry_Bean_Dataset.csv             # le dataset officiel UCI (13 611 × 17)
├── Rapport_KNN.docx                 # rapport 4 pages (livrable n°2 du prof)
├── Rapport_KNN.pdf                  # même rapport, en PDF
├── Presentation_KNN.pptx            # slides 10 pages pour la soutenance (livrable n°3)
├── requirements.txt                 # bibliothèques Python à installer
├── GUIDE_UTILISATION.md             # ← ce fichier
├── EXPLICATIONS_DETAILLEES.md       # cours pas-à-pas pour débutant (19 chapitres)
├── run_log.txt                      # trace complète de la dernière exécution
├── build_notebook.py / build_report.js / build_slides.js   # scripts de construction (optionnel)
└── resultats/                       # tout ce que le script principal produit
    ├── 01_statistiques_descriptives.csv
    ├── 01_effectifs_par_variete.png
    ├── 01_distributions_variables.png
    ├── 01_correlations.png
    ├── 03_experience_standardisation.csv / .png
    ├── 04_resultats_gridsearch.csv
    ├── 04_f1_selon_k_et_distance.png
    ├── 04_heatmap_grille.png
    ├── 05_metriques_globales.csv
    ├── 05_f1_par_variete.csv / .png
    ├── 05_confusions_principales.csv
    ├── 05_matrice_confusion.png
    ├── 06_baseline_dummy.csv              ← BONUS DummyClassifier
    ├── 07_pca_2d.png                      ← BONUS projection PCA 2D
    ├── 08_learning_curve.csv / .png       ← BONUS courbe d'apprentissage
    ├── 09_temps_prediction.csv            ← BONUS mesure de temps
    └── 10_erreurs_analyse.csv             ← BONUS analyse d'erreurs
```

Un fichier dont le nom commence par `0X_` correspond à l'étape X du protocole du prof.

---

## 2. Installation (une seule fois)

Dans un terminal, placez-vous dans le dossier `knn_dry_bean/` puis :

```bash
# (optionnel mais recommandé) créer un environnement isolé
python3 -m venv .venv
source .venv/bin/activate         # Windows : .venv\Scripts\activate

# installer les bibliothèques
pip install -r requirements.txt
```

Si vous n'avez pas `requirements.txt`, lancez simplement :

```bash
pip install numpy pandas scikit-learn matplotlib seaborn openpyxl scipy
```

---

## 3. Lancer le script

### a. Sur le dataset réel (ce qu'on rend au prof)

```bash
python3 knn_dry_bean.py --data Dry_Bean_Dataset.csv --out resultats
```

Si vous avez téléchargé le `.xlsx` original depuis UCI, remplacez juste l'extension :

```bash
python3 knn_dry_bean.py --data Dry_Bean_Dataset.xlsx --out resultats
```

Le `.arff` original marche aussi : `--data Dry_Bean_Dataset.arff`.

**Durée totale :** environ 25-40 secondes sur un ordinateur portable récent.

### b. Mode test (si le dataset n'est pas disponible)

```bash
python3 knn_dry_bean.py --synthetic --out resultats_test
```

Cela génère de fausses données de la même forme (16 variables, 7 classes, 13 611 lignes). **Les métriques obtenues n'ont AUCUNE valeur scientifique**, c'est uniquement pour vérifier que le code tourne.

---

## 4. Ce que le script fait, dans l'ordre

| Étape | Ce que le script fait | Fichiers produits |
|---|---|---|
| 1 | **Exploration** : charge le dataset, compte les lignes, vérifie les valeurs manquantes et les doublons, décrit les 16 variables, trace les effectifs par variété, des boîtes à moustaches et la matrice de corrélation. | `01_*.csv` et `01_*.png` |
| 2 | **Split 80/20 stratifié** avec `random_state=42` et `stratify=y`. | affichage console |
| 3 | **Expérience de standardisation** : entraîne KNN (`k=5`) **avec** puis **sans** `StandardScaler`, en validation croisée 5-fold sur le train. | `03_experience_standardisation.csv` + `.png` |
| 4 | **Tuning par `GridSearchCV`** sur 60 combinaisons (`k ∈ {1,3,5,7,9,11,15,21,31,41}`, distance ∈ {euclidean, manhattan, chebyshev}, poids ∈ {uniform, distance}), scoring = **F1 macro**, CV = `StratifiedKFold(5, shuffle=True, random_state=42)`. | `04_resultats_gridsearch.csv`, `04_f1_selon_k_et_distance.png`, `04_heatmap_grille.png` |
| 5 | **Évaluation finale (UNE SEULE FOIS)** sur le test set : accuracy, précision macro, rappel macro, F1 macro, F1 par classe, matrice de confusion (effectifs + normalisée). | `05_metriques_globales.csv`, `05_f1_par_variete.csv/.png`, `05_matrice_confusion.png`, `05_confusions_principales.csv` |

---

## 5. Résultats à annoncer à l'équipe (dataset réel)

- **Meilleurs hyperparamètres KNN :** `n_neighbors = 9`, `metric = "euclidean"`, `weights = "distance"`
- **F1 macro en validation croisée (train) :** `0.9382`
- **Accuracy sur le test :** `0.9174`
- **F1 macro sur le test :** `0.9300`
- **Impact énorme de la standardisation** : F1 macro passe de `0.714` → `0.936` en CV (+0.22). C'est la conclusion-clé à mettre dans le rapport.
- **Classe parfaitement reconnue :** BOMBAY (F1 = 1.000, car ses grains sont beaucoup plus gros que les autres).
- **Confusions les plus fréquentes :** DERMASON ↔ SIRA (54 + 51 erreurs) — deux variétés morphologiquement très proches.

---

## 6. Règle d'or : ce qui est non négociable dans le code

Si quelqu'un modifie le script, ne pas toucher à :

```python
SEED      = 42
TEST_SIZE = 0.20
N_SPLITS  = 5
```

```python
train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(...))])
scoring = "f1_macro"
```

Sinon les chiffres ne seront plus comparables entre nos modèles.

---

## 7. Fusion dans le notebook commun

Dans le notebook de l'équipe, la section KNN se résume à ces quelques lignes :

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

knn_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn",    KNeighborsClassifier()),
])

knn_grid = GridSearchCV(
    knn_pipe,
    param_grid={
        "knn__n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21, 31, 41],
        "knn__metric":      ["euclidean", "manhattan", "chebyshev"],
        "knn__weights":     ["uniform", "distance"],
    },
    cv=cv, scoring="f1_macro", n_jobs=-1,
).fit(X_train, y_train)

best_knn = knn_grid.best_estimator_
y_pred_knn = best_knn.predict(X_test)
```

Le reste (métriques, matrice de confusion) suit le même code que pour les 4 autres modèles.

---

## 8. Dépannage rapide

| Message d'erreur | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'sklearn'` | `pip install scikit-learn` |
| `FileNotFoundError: Dry_Bean_Dataset.csv` | Le script doit être lancé dans le dossier où se trouve le CSV, ou passer le chemin complet à `--data`. |
| `Format non supporte` | Utilisez un `.csv`, `.xlsx` ou `.arff`. |
| Le script est lent | Normal la première fois (25-40 s). Si ça dépasse 2 min, réduisez la grille de recherche. |
| Les images ne s'ouvrent pas | Elles sont en PNG, ouvrez-les avec n'importe quelle visionneuse. Le script tourne en mode "sans fenêtre" (matplotlib `Agg`), c'est voulu. |

---

## 9. Pour aller plus loin

Tout est détaillé pas à pas (algo KNN, standardisation, validation croisée, F1 macro, lecture de la matrice de confusion…) dans **`EXPLICATIONS_DETAILLEES.md`**, écrit pour quelqu'un qui débute en ML.
