# Explications détaillées — tout ce que fait le script, pour un débutant

Ce document explique **tout** ce que fait le script `knn_dry_bean.py`, depuis les notions les plus fondamentales du Machine Learning jusqu'à la lecture des métriques finales. Chaque terme technique est défini au moment où on l'utilise pour la première fois. On s'appuie aussi sur **les vrais chiffres obtenus sur le dataset Dry Bean**, pour que vous puissiez retrouver les résultats dans le dossier `resultats/`.

**Plan du document**

1. C'est quoi, le Machine Learning ?
2. Notre problème : reconnaître 7 variétés de haricots
3. Le dataset : à quoi il ressemble
4. La règle d'or de l'équipe (et pourquoi elle existe)
5. Étape 1 — Exploration des données
6. Étape 2 — Séparer train et test (le split 80/20 stratifié)
7. Étape 3 — Pourquoi standardiser, et le piège de la "fuite de données"
8. Étape 4 — L'algorithme KNN (k plus proches voisins), expliqué à fond
9. Étape 5 — La validation croisée (CV) et le `GridSearchCV`
10. Étape 6 — Les métriques : accuracy, précision, rappel, F1, F1 macro
11. Étape 7 — La matrice de confusion : comment la lire
12. Étape 8 — Nos résultats finaux et leur interprétation
13. Alors, le modèle est-il bon ou pas ?
14. Limites et pistes d'amélioration
15. **BONUS 1** — Le notebook Jupyter : à quoi ça sert
16. **BONUS 2** — Le `DummyClassifier` : à quoi comparer un modèle
17. **BONUS 3** — L'ACP (PCA) : voir 16 dimensions sur 2 axes
18. **BONUS 4** — La courbe d'apprentissage : a-t-on besoin de plus de données ?
19. **BONUS 5** — Temps de prédiction et analyse d'erreurs

---

## 1. C'est quoi, le Machine Learning ?

Le **Machine Learning** (ou "apprentissage automatique", ML en abrégé) est une famille de méthodes où **un programme apprend à faire une tâche en observant des exemples**, au lieu qu'on lui donne explicitement toutes les règles.

Dans la **programmation classique**, un humain écrit :
> "Si le haricot a une surface > 100 000 pixels, c'est un BOMBAY. Sinon, si sa forme est allongée et petite, c'est un HOROZ. Sinon…"

C'est très long à écrire, fragile, et incapable de s'adapter à de nouvelles situations.

Dans le **Machine Learning**, on dit au programme :
> "Voici 10 000 photos de haricots déjà étiquetées. Trouve **toi-même** les règles qui permettent de retrouver l'étiquette à partir des caractéristiques."

Le programme "apprend" alors un **modèle** — c'est-à-dire une fonction qui prend en entrée les caractéristiques d'un haricot (surface, périmètre, etc.) et renvoie en sortie une prédiction de variété.

### Vocabulaire indispensable

| Terme | Signification en langage courant |
|---|---|
| **Observation** (ou "exemple", "échantillon", "ligne") | Un individu du dataset. Chez nous : un grain de haricot. |
| **Caractéristique** (ou "feature", "variable d'entrée", "X") | Une mesure faite sur l'observation (surface, périmètre…). On en a 16. |
| **Étiquette** (ou "label", "cible", "classe", "y") | Ce qu'on veut prédire. Chez nous : le nom de la variété (SEKER, BOMBAY, …). |
| **Classification** | Problème où la cible est une **catégorie** (variété de haricot, spam vs non-spam…). Chez nous, 7 catégories → classification **multiclasse**. |
| **Régression** | Problème où la cible est un **nombre** (prix d'une maison, température…). Pas notre cas. |
| **Entraînement** (ou "apprentissage", "fit") | Phase où le modèle regarde les exemples étiquetés et ajuste ses paramètres internes. |
| **Prédiction** (ou "inférence", "predict") | Phase où le modèle, déjà entraîné, étiquette un nouvel exemple qu'il n'a jamais vu. |

Les trois grandes familles de ML :
- **Supervisé** : on connaît les étiquettes des exemples d'apprentissage. **← notre cas.**
- **Non supervisé** : on cherche des structures sans étiquettes (regroupement de clients par exemple).
- **Par renforcement** : un agent apprend par essais-erreurs à partir d'une récompense (jeux, robotique).

---

## 2. Notre problème

Un industriel a photographié des grains de haricots sur un tapis roulant. Pour chaque grain, un logiciel de vision a extrait **16 mesures géométriques** (surface, périmètre, longueur de l'axe principal, excentricité, etc.). On connaît la variété de chaque grain.

**But du projet :** construire un modèle qui, à partir des 16 mesures d'un nouveau grain, devine sa variété parmi les 7 possibles (SEKER, BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SIRA).

On ne traite pas les images directement, on travaille directement sur ces 16 chiffres.

**Ma partie :** je m'occupe du modèle **KNN** (K plus proches voisins). Les autres membres de l'équipe font la régression logistique, les arbres de décision, la random forest et le SVM. On comparera tout à la fin.

---

## 3. Le dataset Dry Bean

- **Source :** UCI Machine Learning Repository, dataset n° 602 ("Dry Bean Dataset"), publié par Koklu & Ozkan en 2020.
- **Taille :** 13 611 grains de haricots × 17 colonnes (16 caractéristiques + 1 étiquette `Class`).
- **Pas de valeurs manquantes.**

Répartition des 7 variétés dans le dataset (obtenue à l'étape 1) :

| Variété  | Effectif | Pourcentage |
|----------|---------:|------------:|
| DERMASON | 3 546    | 26.1 %      |
| SIRA     | 2 636    | 19.4 %      |
| SEKER    | 2 027    | 14.9 %      |
| HOROZ    | 1 928    | 14.2 %      |
| CALI     | 1 630    | 12.0 %      |
| BARBUNYA | 1 322    |  9.7 %      |
| BOMBAY   |   522    |  3.8 %      |

**Observation importante :** les classes sont **déséquilibrées**. BOMBAY ne représente que 3.8 % des grains, tandis que DERMASON en représente 26 %. C'est **exactement pour ça** que le prof exige le **F1 macro** comme métrique (expliqué plus loin, section 10) : il refuse qu'on ignore les petites classes.

Les 16 caractéristiques sont toutes **numériques** et décrivent la forme des grains :

- **Taille** : `Area`, `Perimeter`, `ConvexArea`, `EquivDiameter`, `MajorAxisLength`, `MinorAxisLength`
- **Forme** : `AspectRation` (rapport longueur/largeur), `Eccentricity`, `Extent`, `Solidity`, `roundness`, `Compactness`
- **Facteurs de forme** calculés : `ShapeFactor1`, `ShapeFactor2`, `ShapeFactor3`, `ShapeFactor4`

Les échelles sont **très différentes** : `Area` va de 20 420 à 254 616, alors que `Solidity` va de 0.919 à 0.995. **Ce détail est critique pour KNN** (voir section 7).

---

## 4. La règle d'or de l'équipe — pourquoi elle existe

Pour que les modèles soient comparables, chacun doit faire **exactement** les mêmes découpages, les mêmes graines aléatoires, la même métrique. Sinon, un écart de 2 points de F1 entre KNN et SVM peut venir **du modèle**, ou simplement **du hasard du split**.

La règle d'or impose :

1. **`random_state = 42`** partout (split, CV, modèle). Une graine aléatoire fixe rend les résultats **reproductibles** : refaire tourner le code donne les mêmes chiffres.
2. **Split 80/20 stratifié** : on garde la même proportion des 7 classes dans le train et dans le test.
3. **Standardisation dans un `Pipeline`** : empêche la "fuite de données" (voir section 7).
4. **CV = 5-fold stratifié sur le train uniquement** : le test est sacré, on n'y touche qu'à la toute fin.
5. **Métrique reine = F1 macro** : adaptée au déséquilibre des classes.

---

## 5. Étape 1 — Exploration des données

Avant tout modèle, on **regarde** les données. Trois choses à vérifier :

1. **Y a-t-il des valeurs manquantes ou des doublons ?**
   - Dans notre dataset : 0 valeur manquante, 68 doublons exacts (on les garde, c'est négligeable et le prof ne demande pas de les retirer).

2. **Comment sont réparties les classes ?** (voir `01_effectifs_par_variete.png`)
   - Déséquilibre marqué : DERMASON est 7× plus fréquent que BOMBAY.

3. **Quelles sont les échelles des variables ?** (voir `01_statistiques_descriptives.csv`)
   - On voit immédiatement que `Area` vaut des dizaines de milliers tandis que `Solidity` vaut autour de 1 → il faudra standardiser.

On trace aussi :
- Des **boîtes à moustaches** (`01_distributions_variables.png`) qui montrent la distribution de 4 variables par variété. On y voit déjà à l'œil nu que BOMBAY se détache (grains beaucoup plus gros) et que DERMASON et SIRA se ressemblent beaucoup.
- La **matrice de corrélation** (`01_correlations.png`) : certaines variables sont très liées (par ex. `Area`, `Perimeter`, `MajorAxisLength`). Ce n'est pas un problème pour KNN, mais c'est bon à savoir.

---

## 6. Étape 2 — Le split train/test

### Pourquoi couper le dataset en deux ?

Un modèle qui a vu un exemple pendant l'apprentissage saura presque toujours lui redonner la bonne étiquette. Ça ne prouve rien sur sa capacité à généraliser à de **nouveaux** grains.

On met donc **de côté** une partie des données (le **test set**), que le modèle ne verra **jamais** pendant l'entraînement ni pendant la recherche des hyperparamètres. Ce test set simule le futur — des grains qui arriveront sur le tapis roulant demain.

### Pourquoi 80/20 ?

C'est un compromis classique : on a assez de données pour entraîner (10 888 grains) et un test suffisamment gros pour que les métriques soient fiables (2 723 grains).

### Pourquoi "stratifié" ?

Un split aléatoire pur pourrait, par malchance, mettre très peu de BOMBAY (la classe rare) dans le test, et fausser les métriques. Le **split stratifié** garantit que chaque classe conserve ses proportions dans le train ET dans le test. Vérification sur nos chiffres :

| Classe   | Train % | Test % |
|----------|--------:|-------:|
| DERMASON | 26.06   | 26.04  |
| SIRA     | 19.37   | 19.35  |
| …        | …       | …      |
| BOMBAY   |  3.84   |  3.82  |

Les proportions sont quasi identiques. 

Code correspondant :

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)
```

---

## 7. Étape 3 — La standardisation, et le "data leakage"

### C'est quoi, standardiser ?

**Standardiser** une variable, c'est la transformer pour qu'elle ait :
- une **moyenne = 0**,
- un **écart-type = 1**.

Formule : `x_new = (x − moyenne) / écart_type`. En scikit-learn, c'est `StandardScaler`.

Avant standardisation : `Area` ≈ 50 000 ± 29 000, `Solidity` ≈ 0.99 ± 0.004.
Après standardisation : les deux variables ont une moyenne de 0 et un écart-type de 1.

### Pourquoi c'est critique pour KNN ?

KNN calcule des **distances** entre points (voir section 8). Si une variable a des valeurs énormes (comme `Area`) et une autre des valeurs minuscules (comme `Solidity`), la distance sera presque entièrement dominée par la grosse variable. `Solidity` sera ignorée **alors qu'elle contient peut-être de l'information précieuse**.

### Notre expérience (étape 3 du script)

On entraîne KNN (k=5) en CV 5-fold sur le train, dans deux configurations :

| Configuration         | F1 macro (CV) | Accuracy (CV) |
|-----------------------|--------------:|--------------:|
| **Sans** StandardScaler | 0.7138        | 0.7134        |
| **Avec** StandardScaler | 0.9355        | 0.9232        |

**Gain : +0.22 points de F1 macro**. C'est énorme. Voir `03_experience_standardisation.png`.

**Conclusion à retenir pour le rapport :** pour KNN (et SVM), standardiser n'est pas optionnel, c'est vital.

### Le piège : la "fuite de données" (data leakage)

Si on fait `StandardScaler().fit(X)` sur **tout** le dataset (train + test) avant de couper, la moyenne et l'écart-type utilisés pour standardiser le train contiennent **déjà** de l'information sur le test. On triche sans s'en rendre compte, et les chiffres sur le test deviennent trop beaux.

**La bonne pratique** est de calculer moyenne + écart-type **uniquement sur le train**, puis d'appliquer la même transformation au test. C'est exactement ce que fait un **`Pipeline`** scikit-learn :

```python
pipe = Pipeline([
    ("scaler", StandardScaler()),      # apprend moyenne/écart-type
    ("knn",    KNeighborsClassifier()) # utilise les features standardisées
])

pipe.fit(X_train, y_train)   # fit propre : uniquement sur le train
pipe.predict(X_test)         # applique la MÊME transformation au test
```

Dans la validation croisée (section 9), le `Pipeline` recalcule le scaler sur chaque sous-train des 5 folds. **Aucune information du test ne fuit jamais dans l'entraînement.** C'est la raison 4 de la règle d'or.

---

## 8. Étape 4 — L'algorithme KNN (K plus proches voisins)

### Principe en une phrase

> *"Pour deviner la variété d'un nouveau grain, je regarde les **K grains du training set qui lui ressemblent le plus**, et je lui attribue la variété la plus fréquente parmi ces K voisins."*

C'est tout. C'est un des algorithmes de ML les plus simples à comprendre.

### Exemple concret

Je reçois un nouveau grain avec `Area=50 000`, `Perimeter=850`, etc. (16 chiffres).

1. Je calcule la **distance** entre ce nouveau grain et **chacun** des 10 888 grains du train.
2. Je garde les **K=9** grains les plus proches.
3. Parmi ces 9 voisins, je compte : par exemple 6 sont DERMASON, 2 sont SIRA, 1 est SEKER.
4. Je prédis **DERMASON** (classe majoritaire parmi les voisins).

### Les 3 hyperparamètres qu'on règle

Un **hyperparamètre** est un réglage choisi par l'humain **avant** l'entraînement (contrairement aux paramètres internes, qui s'ajustent tout seuls). Pour KNN :

#### a. `n_neighbors` (= K) : combien de voisins regarder ?

- **K petit** (K=1 ou K=3) : le modèle "colle" aux données d'entraînement. Il est très sensible au bruit, on dit qu'il **surapprend** (**overfitting**).
- **K grand** (K=100 ou plus) : le modèle devient trop lisse, il noie les petites classes dans la majorité, on dit qu'il **sous-apprend** (**underfitting**).
- Il faut trouver un **compromis**. On teste : `K ∈ {1, 3, 5, 7, 9, 11, 15, 21, 31, 41}`.

#### b. `metric` : quelle distance utiliser ?

- **Euclidienne** : la distance "à vol d'oiseau", `√Σ(xᵢ − yᵢ)²`. La plus classique.
- **Manhattan** : distance "en taxi" dans une ville à grille, `Σ|xᵢ − yᵢ|`.
- **Chebyshev** : distance "roi d'échecs", `max|xᵢ − yᵢ|`.

On teste les trois.

#### c. `weights` : les voisins ont-ils tous le même poids ?

- `uniform` : les K voisins votent tous avec le même poids.
- `distance` : les voisins les plus proches comptent **plus** (poids = 1/distance). Souvent meilleur, car un voisin très proche est plus informatif qu'un voisin "limite".

### Côté "paresseux"

KNN est appelé un **modèle paresseux** ("lazy learner") : il **ne fait rien pendant l'entraînement**, il se contente de mémoriser toutes les données d'entraînement. Tout le travail se fait au moment de la prédiction (calculer 10 888 distances pour chaque nouveau grain).

**Conséquences :**
- Entraînement très rapide.
- Prédiction lente sur gros datasets.
- Besoin de garder tout le train en mémoire.

---

## 9. Étape 5 — Validation croisée et `GridSearchCV`

### Pourquoi la validation croisée ?

Comment choisir le meilleur `K` ? Mauvaise idée : "j'essaie `K=1,3,5…` et je prends celui qui donne le meilleur score **sur le test**". C'est de la triche : on utiliserait le test pour choisir le modèle, donc le score du test ne refléterait plus la performance sur de nouvelles données.

**La bonne solution : la validation croisée (cross-validation, CV).**

### K-fold stratifié expliqué avec un dessin

On divise le **train** (10 888 grains) en **5 morceaux égaux stratifiés** (chaque morceau ≈ 2 178 grains, avec les mêmes proportions de classes).

```
Pli 1  : [TEST ][train][train][train][train]  → score 1
Pli 2  : [train][TEST ][train][train][train]  → score 2
Pli 3  : [train][train][TEST ][train][train]  → score 3
Pli 4  : [train][train][train][TEST ][train]  → score 4
Pli 5  : [train][train][train][train][TEST ]  → score 5
                                               ─────────
                                               moyenne = score CV
```

À chaque "pli", on entraîne sur 4 morceaux et on évalue sur le 5ᵉ. On moyenne les 5 scores. Ça donne une estimation beaucoup plus fiable que de juger sur un seul petit morceau, et surtout **ça ne touche pas au test set**.

Code correspondant :

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

### `GridSearchCV` : essayer toutes les combinaisons

On veut tester **toutes** les combinaisons : 10 valeurs de K × 3 distances × 2 schémas de poids = **60 combinaisons**. Pour chacune, on fait la CV à 5 plis. Total : **60 × 5 = 300 entraînements**.

`GridSearchCV` fait tout ça pour nous, en parallèle (`n_jobs=-1` = tous les cœurs CPU) :

```python
grid = GridSearchCV(
    pipe,
    param_grid={
        "knn__n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21, 31, 41],
        "knn__metric":      ["euclidean", "manhattan", "chebyshev"],
        "knn__weights":     ["uniform", "distance"],
    },
    cv=cv,
    scoring="f1_macro",   # ← la métrique reine
    n_jobs=-1,
)
grid.fit(X_train, y_train)
```

### Résultat sur notre dataset

Durée : **24 s**. Vainqueur :

| `n_neighbors` | `metric`   | `weights`  | **F1 macro (CV)** |
|--------------:|------------|------------|------------------:|
| **9**         | **euclidean** | **distance** | **0.9382**        |
| 21            | euclidean  | distance   | 0.9381            |
| 11            | euclidean  | distance   | 0.9376            |
| 9             | euclidean  | uniform    | 0.9373            |
| 11            | euclidean  | uniform    | 0.9372            |

Voir `04_resultats_gridsearch.csv` pour le tableau complet, et `04_f1_selon_k_et_distance.png` pour la courbe. On remarque que :
- **La distance euclidienne gagne**.
- **Pondérer par la distance** ajoute un petit boost.
- Le F1 est quasi plat entre K=5 et K=31 : ce n'est pas un hyperparamètre très sensible.

---

## 10. Étape 6 — Les métriques expliquées simplement

Pour la **classification**, on compare `y_pred` (ce que le modèle prédit) à `y_true` (la vraie classe). Pour chaque exemple, on a soit juste, soit faux. On veut mesurer la performance globale.

### Accuracy ("taux de bonnes réponses")

$\text{Accuracy} = \dfrac{\text{nombre de prédictions correctes}}{\text{nombre total d'exemples}}$

Simple et intuitif, mais **trompeur en cas de déséquilibre**. Exemple : si 95 % des mails sont du non-spam, un modèle qui prédit toujours "non-spam" a 95 % d'accuracy… et ne sert à rien.

### Précision, Rappel, F1 — définis pour **chaque classe**

Prenons la classe **BOMBAY** comme exemple. Pour cette classe, on compte :

- **VP** (Vrais Positifs) : BOMBAY correctement prédits BOMBAY.
- **FP** (Faux Positifs) : d'autres classes prédites **à tort** comme BOMBAY.
- **FN** (Faux Négatifs) : BOMBAY réels prédits **à tort** comme autre chose.

Alors :

$$\text{Précision}_{\text{BOMBAY}} = \frac{VP}{VP + FP}$$

→ *"Parmi tous les grains que j'ai dits BOMBAY, quelle fraction l'étaient vraiment ?"*

$$\text{Rappel}_{\text{BOMBAY}} = \frac{VP}{VP + FN}$$

→ *"Parmi tous les BOMBAY réellement présents, quelle fraction ai-je retrouvée ?"*

$$F1_{\text{BOMBAY}} = 2 \cdot \frac{\text{Précision} \cdot \text{Rappel}}{\text{Précision} + \text{Rappel}}$$

→ moyenne harmonique des deux ; punit sévèrement une précision ou un rappel bas.

### Macro vs Micro vs Weighted

- **Macro-F1** : moyenne **non pondérée** des F1 de chaque classe : `(F1_SEKER + F1_BARBUNYA + … + F1_SIRA) / 7`. **Toutes les classes comptent pareil**, même les petites. → **C'est notre métrique reine.**
- **Micro-F1** : calculée globalement, revient presque à l'accuracy en multiclasse.
- **Weighted-F1** : moyenne des F1 **pondérée par l'effectif** de chaque classe → favorise les grosses classes.

### Pourquoi le prof exige Macro-F1 ?

Parce que BOMBAY (3.8 %) et DERMASON (26 %) **doivent peser autant** dans l'évaluation. Si on utilisait l'accuracy ou le weighted-F1, un modèle qui se plante systématiquement sur BOMBAY serait à peine pénalisé. Avec le macro-F1, un mauvais F1 sur BOMBAY **divise la moyenne globale**.

---

## 11. Étape 7 — La matrice de confusion

C'est **le** graphique à savoir lire. On a un tableau 7 × 7 :
- en **ligne** : la vraie classe,
- en **colonne** : la classe prédite,
- chaque **case** : combien de grains vrai=i ont été prédits j.

La **diagonale** = les prédictions correctes. Tout ce qui est en dehors = les erreurs.

Voir `resultats/05_matrice_confusion.png`. On en tire deux versions :
- **effectifs bruts** : utile pour voir où sont les erreurs en volume.
- **normalisée par ligne** : chaque ligne somme à 1, chaque case est un **rappel par classe** → utile pour comparer entre classes déséquilibrées.

Exemple de lecture (sur nos données) :
- Ligne BOMBAY, colonne BOMBAY : 104 → les 104 BOMBAY du test ont **tous** été correctement reconnus (rappel = 1.00).
- Ligne DERMASON, colonne SIRA : 54 → 54 DERMASON ont été prédits à tort comme SIRA.
- Ligne SIRA, colonne DERMASON : 51 → confusion symétrique.

**Conclusion morphologique :** DERMASON et SIRA se ressemblent beaucoup. Pour les distinguer, il faudrait des caractéristiques supplémentaires (par exemple la couleur, qu'on n'a pas ici).

---

## 12. Étape 8 — Nos résultats finaux (dataset Dry Bean, UNE évaluation sur le test)

### Métriques globales

| Métrique            | Valeur |
|---------------------|-------:|
| Accuracy            | **0.9174** |
| Précision macro     | 0.9328 |
| Rappel macro        | 0.9279 |
| **F1 macro**        | **0.9300** |
| F1 macro en CV train | 0.9382 |

### F1 par variété

| Variété  | Précision | Rappel | **F1** | Effectif test |
|----------|----------:|-------:|-------:|-------------:|
| BOMBAY   | 1.000     | 1.000  | **1.000** | 104          |
| HOROZ    | 0.958     | 0.956  | 0.957  | 386          |
| SEKER    | 0.943     | 0.946  | 0.945  | 406          |
| CALI     | 0.920     | 0.948  | 0.934  | 326          |
| DERMASON | 0.913     | 0.907  | 0.910  | 709          |
| BARBUNYA | 0.954     | 0.868  | 0.909  | 265          |
| SIRA     | 0.841     | 0.871  | 0.856  | 527          |

### Top 5 des confusions

| Vraie classe | Prédite | Nb erreurs |
|--------------|---------|-----------:|
| DERMASON     | SIRA    | 54 |
| SIRA         | DERMASON| 51 |
| BARBUNYA     | CALI    | 23 |
| SEKER        | SIRA    | 14 |
| DERMASON     | SEKER   | 12 |

### Ce qu'on en retire

1. **BOMBAY est parfaitement reconnu** (F1 = 1.000) : ses grains sont beaucoup plus gros que les autres, aucune confusion possible.
2. **SIRA est la classe la plus difficile** (F1 = 0.856) à cause de sa confusion mutuelle avec DERMASON.
3. **L'écart `F1_CV_train (0.938)` vs `F1_test (0.930)`** est petit (−0.008). Pas de surapprentissage marqué → le modèle **généralise bien** à de nouvelles données.
4. **Le gain de la standardisation (+0.22 F1 macro)** est la leçon-clé à retenir pour KNN.

---

## 13. Alors, le modèle est-il bon ou pas ?

**Oui, il est bon**, et voici comment on le justifie :

1. **Les métriques sont élevées et équilibrées.** Un F1 macro de **0.93** sur un problème à 7 classes déséquilibrées est un très bon résultat (un modèle aléatoire qui tire une classe au hasard aurait environ 1/7 ≈ 0.14 d'accuracy).
2. **Le modèle ne surapprend pas.** L'écart entre la performance en CV (0.938) et sur le test (0.930) est minime. Cela signifie que le modèle n'a pas "appris par cœur" les données d'entraînement.
3. **Les 7 classes sont toutes au-dessus de F1 = 0.85.** Même SIRA, la plus difficile, dépasse 0.85. Aucune classe n'est "sacrifiée".
4. **Les erreurs sont explicables.** Les confusions DERMASON ↔ SIRA correspondent à des variétés morphologiquement très proches. Ce ne sont pas des erreurs "stupides" du modèle — même un expert humain hésiterait sur ces cas en ne regardant que les mesures géométriques.
5. **Le modèle est reproductible et robuste.** Grâce à la graine 42 et au `Pipeline`, n'importe qui peut relancer le code et retrouver exactement les mêmes chiffres.

### Est-il excellent ? Ça dépend du contexte :
- **Pour un prototype universitaire :** oui, excellent.
- **Pour une industrie agroalimentaire** qui doit trier des millions de grains/jour : il faudrait viser > 0.98 et surtout **zéro erreur sur les classes à haute valeur commerciale**. Dans ce cas, on combinerait KNN avec d'autres modèles (ensemble learning, image réelle avec CNN, etc.).

---

## 14. Limites et pistes d'amélioration

- **KNN est lent en prédiction** sur gros volumes (il recalcule des distances contre tout le train). Pour un déploiement en production, un `KDTree` / `BallTree` (déjà utilisé en interne par scikit-learn) aide, mais un modèle "rapide" comme la Random Forest serait sans doute préférable.
- **DERMASON vs SIRA** reste le point faible. Pour l'améliorer :
  - ajouter de nouvelles caractéristiques (couleur, texture) qui n'existent pas dans ce dataset,
  - essayer d'autres algorithmes (SVM à noyau RBF, gradient boosting),
  - faire du **feature engineering** (créer de nouvelles variables comme `Perimeter²/Area`).
- **Classes déséquilibrées** : on pourrait tester un rééchantillonnage (`SMOTE`) pour booster BARBUNYA et BOMBAY, bien que la standardisation + KNN pondéré par distance s'en sorte déjà très bien ici.
- **Généralisation** : le dataset vient d'un seul tapis roulant, d'une seule caméra, d'un seul producteur. Sur des grains d'ailleurs, les performances pourraient baisser. C'est la limite de tout modèle de ML : il généralise **à la distribution d'apprentissage**, pas au-delà.

---

**En résumé :** on a entraîné un KNN proprement (pipeline, pas de fuite, CV stratifiée, tuning sur 60 combinaisons, évaluation unique sur le test). Le modèle retenu est **`k=9`, distance euclidienne, pondérée par distance**, atteint **F1 macro = 0.930 sur le test**, se généralise bien, et sera comparable sans friction aux 4 autres modèles de l'équipe grâce à la règle d'or.

---

## 15. BONUS 1 — Le notebook Jupyter : à quoi ça sert

### Script `.py` vs notebook `.ipynb` : la différence

Jusqu'ici on avait un **script** `knn_dry_bean.py`. On a aussi créé un **notebook** `knn_dry_bean.ipynb`. Les deux font la même chose, mais sont utilisés très différemment :

| Script Python (`.py`) | Notebook Jupyter (`.ipynb`) |
|---|---|
| Un seul fichier texte avec tout le code | Succession de **cellules** : code + texte + graphiques mélangés |
| Tourne en une fois, du début à la fin | On exécute **cellule par cellule**, dans l'ordre qu'on veut |
| Bon pour la **production** (livraison, automatisation) | Bon pour l'**exploration** et le **rendu pédagogique** |
| Les résultats s'affichent dans le terminal | Les résultats (texte + graphique) sont **sauvés dans le fichier lui-même** |
| On n'ouvre pas le fichier pour le "lire" | On peut l'ouvrir et le **lire comme un rapport interactif** |

### Pourquoi le prof demande un notebook

Dans le livrable n°1 ("Notebook Jupyter / Python"), il veut pouvoir ouvrir **un seul fichier** et voir en même temps :
- ton **code** (pour vérifier que c'est propre),
- tes **commentaires** en Markdown (pour comprendre ton raisonnement),
- les **résultats** affichés (tableaux, graphiques) **déjà calculés** — il n'a pas besoin de relancer quoi que ce soit.

C'est un **document vivant** : le code est à côté de son résultat.

### Structure d'une cellule

Un notebook est une liste de **cellules**, de deux types :
- **Cellule Markdown** : du texte formaté (titre, paragraphe, tableau) — comme ce que tu lis maintenant.
- **Cellule Code** : du Python qu'on peut exécuter. Son **résultat** (print, graphique, DataFrame) s'affiche **juste en dessous** après exécution.

Exemple schématique :

```
┌──────── Cellule Markdown ────────┐
│ ## 3. Split stratifié             │
│ On sépare train et test...        │
└───────────────────────────────────┘

┌──────── Cellule Code ─────────────┐
│ X_train, X_test, ... = train_test_split(...)│
│ print(X_train.shape)              │
└───────────────────────────────────┘

┌──────── Sortie (auto) ────────────┐
│ (10888, 16)                       │
└───────────────────────────────────┘
```

### Comment ouvrir et exécuter notre notebook

Deux options :

**Option A — JupyterLab / Jupyter Notebook (local, recommandé)**
```bash
pip install jupyterlab
jupyter lab knn_dry_bean.ipynb
```
Une page web s'ouvre, on peut lire et relancer chaque cellule avec `Shift + Enter`.

**Option B — Google Colab (en ligne, zéro installation)**
1. Aller sur https://colab.research.google.com
2. "File → Upload notebook" → choisir `knn_dry_bean.ipynb`
3. Uploader aussi `Dry_Bean_Dataset.csv` dans le panneau de gauche
4. Faire `Runtime → Run all`

### Comparer `.py` et `.ipynb` dans notre livrable

Les deux coexistent dans le dossier :
- `knn_dry_bean.py` : pour **relancer** en une commande, pour fusionner dans un pipeline, pour le robuste.
- `knn_dry_bean.ipynb` : pour **lire** le travail, comprendre les étapes, montrer au prof.

Les deux produisent **exactement** les mêmes chiffres (on a utilisé `random_state=42` partout).

---

## 16. BONUS 2 — Le `DummyClassifier` : à quoi comparer un modèle

### Le problème : un chiffre tout seul ne veut rien dire

On a obtenu **F1 macro = 0.93** sur le test. Est-ce "bon" ? Est-ce "excellent" ? On ne peut pas le dire sans **point de comparaison**.

Imagine un étudiant qui annonce : "J'ai eu 14/20 à l'exam". Est-ce bien ? Ça dépend :
- si la moyenne de la classe est 8 → il est **très bon**,
- si la moyenne est 17 → il est **en-dessous du lot**.

En ML, le même raisonnement s'applique. Il faut comparer notre 0.93 à un **plancher** — ce que n'importe quel modèle idiot obtiendrait.

### `DummyClassifier` : un modèle volontairement bête

`sklearn.dummy.DummyClassifier` est un classifieur qui **ignore complètement les features d'entrée** et prédit selon une règle simple. Il existe plusieurs stratégies :

| Stratégie       | Ce qu'elle fait |
|-----------------|-----------------|
| `most_frequent` | Prédit **toujours** la classe la plus fréquente du train (chez nous : DERMASON). |
| `stratified`    | Prédit une classe au hasard, en respectant les proportions du train. |
| `uniform`       | Prédit une classe au hasard uniforme (1/7 pour chaque). |

### Nos résultats concrets

| Modèle                       | Accuracy test | **F1 macro test** |
|------------------------------|--------------:|------------------:|
| DummyClassifier `most_frequent` | 0.2604        | **0.0590**        |
| DummyClassifier `stratified`    | 0.1763        | **0.1470**        |
| DummyClassifier `uniform`       | 0.1392        | **0.1332**        |
| **Notre KNN optimisé**          | **0.9174**    | **0.9300**        |

### Comment lire ces chiffres

- `most_frequent` : accuracy = 0.26 = proportion de DERMASON dans le test. F1 macro = 0.06 car il n'arrive jamais à prédire les 6 autres classes.
- `stratified` : au pire, prédit une classe au hasard pondéré → tombe juste 1 fois sur 7 environ.
- **Notre KNN à F1 = 0.93 est 6 à 15 fois meilleur** que n'importe quelle baseline.

### Pourquoi c'est important

Le `DummyClassifier` sert à :
1. **Valider qu'il y a bien du signal dans les données.** Si un KNN ne battait pas la baseline, cela voudrait dire que les 16 features sont inutiles.
2. **Donner une échelle.** 0.93 - 0.15 = 0.78 points de F1 "apportés" par le modèle. C'est énorme.
3. **Détecter les bugs.** Si ton modèle fait 0.14 alors qu'une baseline fait 0.15, c'est qu'il y a une erreur quelque part (classes inversées, pipeline mal configuré...).

> **À retenir pour le rapport :** toujours comparer un modèle à au moins une baseline simple. C'est une pratique de rigueur scientifique.

---

## 17. BONUS 3 — L'ACP (PCA) : voir 16 dimensions sur 2 axes

### Le problème : on ne sait pas dessiner en 16 dimensions

Nos grains ont 16 features. Pour **voir** où se situent les classes les unes par rapport aux autres, il faudrait un graphique à 16 axes. Un humain ne sait pas visualiser au-delà de 3.

### L'idée de l'ACP (Analyse en Composantes Principales, `PCA` en anglais)

L'ACP construit **de nouveaux axes** (appelés **composantes principales**, PC1, PC2, ...) qui sont des **combinaisons linéaires** des 16 axes originaux, et qui ont une propriété géniale :

> *"PC1 est l'axe le long duquel les données **s'étalent le plus** (celui qui capture le plus d'information). PC2 est le deuxième axe le plus étalant, **perpendiculaire** à PC1. Etc."*

En gardant seulement **les 2 premiers axes**, on obtient une projection 2D qui conserve **le maximum de variance possible** sur 2 dimensions.

### Analogie simple

Imagine un objet 3D (un ballon de rugby). Sa "vue de dessus" (2D) perd l'épaisseur, mais garde la forme allongée. C'est exactement ce que fait l'ACP, en généralisant à n dimensions.

### Variance expliquée

Chaque composante vient avec un chiffre : **le pourcentage de variance totale qu'elle capture**. Chez nous :

| Composante | Variance expliquée | Cumul |
|------------|-------------------:|------:|
| PC1        | 55.5 %             | 55.5 % |
| PC2        | 26.4 %             | **81.9 %** |

**Lecture :** en ne gardant que 2 axes sur 16, on garde quand même **82 %** de l'information. C'est excellent — ça veut dire que les 16 variables sont très **redondantes** (par ex. `Area`, `Perimeter`, `MajorAxisLength` mesurent toutes "la taille" du grain).

### Ce qu'on voit sur `07_pca_2d.png`

Sur le graphique, chaque point = un grain, coloré par sa variété. Résultats visibles **à l'œil nu** :

- **BOMBAY** (jaune ou vert selon la palette) : un **îlot complètement séparé**, loin des autres. → Conclusion : impossible à confondre, F1 = 1.00 attendu ✓
- **DERMASON et SIRA** : deux nuages qui **se chevauchent presque entièrement**. → Conclusion : confusion mutuelle garantie ✓
- **CALI et BARBUNYA** : nuages partiellement superposés. → Conclusion : confusion modérée attendue (on a vu 23 erreurs CALI → BARBUNYA) ✓

### Pourquoi c'est puissant

Avant même d'entraîner un modèle, l'ACP permet de **prédire quelles classes seront confuses**. Si deux nuages se chevauchent, aucun modèle (KNN, SVM ou autre) ne pourra parfaitement les séparer — à moins d'ajouter de nouvelles features.

### Attention : limites de l'ACP

- L'ACP est **linéaire**. Si les classes sont séparées de façon courbe, l'ACP ne le verra pas. Pour ça, on utilise `t-SNE` ou `UMAP` (non abordé ici).
- Les axes PC1/PC2 **n'ont pas d'interprétation physique directe**. Ce sont des mélanges abstraits des 16 variables.
- **On ne s'en sert JAMAIS pour entraîner le modèle final** (sauf cas spécifiques). C'est un **outil de visualisation**, point. Notre KNN tourne sur les 16 features originales.

---

## 18. BONUS 4 — La courbe d'apprentissage

### La question qu'elle répond

> *"Si j'avais eu plus de données d'entraînement, mon modèle aurait-il été meilleur ?"*

C'est **la** question qu'on te posera en soutenance. La courbe d'apprentissage donne la réponse, et pour pas cher.

### Le principe

On entraîne le modèle **plusieurs fois**, avec des quantités croissantes de données :
- 10 % du train (≈ 1 000 grains)
- 23 %, 36 %, …, 100 % du train (10 888 grains)

À chaque niveau, on calcule deux scores en **CV 5-fold** :
- **F1 train** = performance sur les grains qu'il a vus,
- **F1 validation** = performance sur les grains qu'il n'a pas vus (le "vrai" score qu'on vise).

### Comment lire la courbe (fichier `08_learning_curve.png`)

**Trois scénarios typiques :**

1. **Les deux courbes convergent et plafonnent haut.**
   → Modèle au top, pas la peine d'ajouter des données.

2. **La courbe `validation` monte encore à la fin.**
   → Plus de données ferait gagner en performance. Si on peut en obtenir, on le fait.

3. **Les deux courbes convergent et plafonnent bas.**
   → Le modèle est trop simple pour le problème (**underfitting**). Changer de modèle ou ajouter des features.

4. **Gros écart `train` >> `validation`.**
   → Surapprentissage (**overfitting**). Simplifier le modèle, régulariser, ou ajouter des données.

### Nos résultats

| Taille train | F1 train | F1 validation |
|-------------:|---------:|--------------:|
| 871          | 1.000    | 0.9242        |
| 1 990        | 1.000    | 0.9299        |
| 3 110        | 1.000    | 0.9318        |
| 4 230        | 1.000    | 0.9327        |
| 5 350        | 1.000    | 0.9347        |
| 6 470        | 1.000    | 0.9362        |
| 7 590        | 1.000    | 0.9365        |
| 8 710        | 1.000    | **0.9382**    |

### Interprétation (à écrire tel quel dans le rapport)

- **`F1 validation` plafonne doucement** autour de 0.93-0.94. Entre 5 350 et 8 710 exemples, le gain est **très faible (+0.004)**. → Plus de données **n'apporterait presque rien**. On est à la limite de ce que KNN peut tirer de ces 16 variables.
- **`F1 train` = 1.000 constant**. Attention, ce chiffre **n'est pas un bug**. Avec `weights="distance"`, le plus proche voisin d'un point du train est **lui-même** (distance = 0, poids = infini). Donc chaque point du train est **forcément** bien classé. C'est une spécificité de KNN pondéré.
- Pour voir du surapprentissage "réel" sur KNN, il faudrait désactiver la pondération et comparer. Mais le signal important est dans `F1 validation`, et il est rassurant.

**Conclusion pratique :** le modèle est à son plafond. Pour faire mieux, il faut **de nouvelles features** (couleur, texture) ou **un autre algorithme** (SVM à noyau, Random Forest, réseaux de neurones).

---

## 19. BONUS 5 — Temps de prédiction et analyse d'erreurs

### Pourquoi mesurer le temps de prédiction

Un modèle utilisé en production (tapis roulant qui trie 100 grains/seconde) a une contrainte de **latence** : il doit répondre vite. KNN est réputé **lent à l'inférence** parce qu'à chaque prédiction, il doit calculer la distance entre le nouveau grain et **tous** les grains du train.

### Nos chiffres

- **2 723 prédictions en 45.9 ms** → **0.017 ms/grain**
- Débit : **≈ 59 000 grains/seconde**

### Verdict

À notre échelle (train ~11 000 grains), **KNN est largement assez rapide** pour un usage industriel standard. Si on avait 10 millions de grains dans le train, on passerait à ~15 ms par prédiction, ce qui commencerait à poser problème. À ce moment-là, on utilise des astuces (KDTree, BallTree, approximate nearest neighbors).

> *À mentionner dans les limites du rapport : "KNN scale mal avec la taille du train, c'est son point faible structurel."*

### Analyse d'erreurs : regarder les faux-pas

On sort les **225 grains mal classés** (sur 2 723 = 8.3 % d'erreurs) et on en regarde 5 au hasard. Fichier complet : `resultats/10_erreurs_analyse.csv`.

Exemple de ce qu'on voit :

| vraie    | prédite  | Area    | Perimeter | AspectRation | Compactness |
|----------|----------|--------:|----------:|-------------:|------------:|
| SIRA     | DERMASON | 33 493  | 683       | 1.49         | 0.82        |
| SIRA     | DERMASON | 38 169  | 735       | 1.56         | 0.80        |
| SEKER    | SIRA     | 47 762  | 817       | 1.42         | 0.84        |
| BARBUNYA | CALI     | 68 999  | 1012      | 1.74         | 0.76        |
| CALI     | SEKER    | 45 666  | 790       | 1.30         | 0.88        |

### Pourquoi c'est utile pour le rapport

- Ça prouve que **les erreurs ne sont pas aléatoires** : ce sont toujours des grains à la **frontière** entre deux classes morphologiquement proches.
- Un expert agronome qui ne verrait que ces 16 mesures hésiterait aussi — ce ne sont pas des erreurs "stupides" du modèle.
- Pour corriger, il faudrait des **mesures supplémentaires** (couleur du grain, texture) qui ne sont pas dans le dataset.

### Comment faire une analyse d'erreurs plus poussée (pour aller plus loin)

1. **Grouper les erreurs par paire de classes** : quelle paire se trompe le plus ? (chez nous : DERMASON ↔ SIRA)
2. **Regarder la distribution des features sur les erreurs** : y a-t-il une plage de valeurs qui piège le modèle ?
3. **Faire un "post-mortem" avec un expert métier** : certaines confusions sont-elles acceptables ? (par ex. deux variétés qui se cuisinent pareil peuvent être tolérées.)

---

**En résumé des 5 bonus :**

| Bonus | Ce qu'il apporte |
|-------|------------------|
| 1. Notebook Jupyter | Rendre le travail **lisible** sans relancer le code |
| 2. DummyClassifier | Prouver que 0.93 F1 est bien **au-dessus du hasard** (0.15) |
| 3. PCA 2D | **Voir** pourquoi certaines classes se confondent |
| 4. Courbe d'apprentissage | Savoir si plus de données aiderait (réponse : **non**) |
| 5. Temps et erreurs | Comprendre le **coût** et le **comportement** du modèle |

Ces 5 bonus transforment une "soumission d'étudiant" en **analyse de data scientist**.
