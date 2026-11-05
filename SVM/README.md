# Dry Bean Classification: Support Vector Machines (SVM)

Machine Learning - Mini-Project 02 (INSEA 2026–2027)  
**Pôle 2 : Modèles à Marges Maximales & Géométrie (SVM Linéaire & RBF)**

---

## 1. Vue d'ensemble du Pôle

L'objectif de ce pôle est de concevoir, d'optimiser et d'évaluer des classificateurs **Support Vector Machines (SVM)** pour la reconnaissance automatique de **7 variétés de haricots secs** (*Seker, Barbunya, Bombay, Cali, Dermosan, Horoz, Sira*) à partir de 16 descripteurs morphologiques extraits d'images de graines.

Nous étudions et comparons deux formulations :
- **SVM Linéaire** : frontière de séparation plane avec optimisation du paramètre de marge souple $C$.
- **SVM Non-linéaire à Noyau Gaussien (RBF)** : projection implicite dans un espace de Hilbert de dimension infinie pour séparer les variétés aux formes géométriquement complexes, avec optimisation conjointe de $(C, \gamma)$.

---

## 2. Protocole Expérimental Obligatoire

Le pipeline suit rigoureusement le protocole partagé de l'équipe :

1. **Jeu de données :** Téléchargement direct depuis le dépôt officiel UCI Dry Bean (13 611 lignes, 16 features numériques, 0 valeurs manquantes).
2. **Découpage étanche :** 80 % Train / 20 % Test avec stratification sur les 7 variétés et graine fixée (`SEED = 42`).
3. **Pipeline & Standardisation :** Intégration systématique de `StandardScaler` dans un `Pipeline` Scikit-Learn afin d'éviter toute fuite de données (*data leakage*).
4. **Validation croisée (Train) :** Recherche par grille (`GridSearchCV`) sur 5 plis stratifiés (`StratifiedKFold`), avec pour métrique d'arbitrage le **Macro F1-score**.
5. **Évaluation finale unique (Test) :** Mesure des performances sur les 2 723 exemples du jeu de test conservé intact.
6. **Diagnostics morphologiques :** Matrice de confusion $7 \times 7$ et explication des confusions entre variétés proches.

---

## 3. Résultats Clés

### 3.1 Impact critique du Feature Scaling (Standardisation)

Les SVM calculent des distances géométriques $\|x_i - x_j\|$. Les descripteurs comme `Area` (amplitude ~ 200 000) écrasent les descripteurs sans dimension comme `Compactness` (amplitude ~ 0.5) sans normalisation :

| Configuration | Accuracy (Test) | Macro F1-score (Test) | Gain en F1 |
| :--- | :---: | :---: | :---: |
| **Sans StandardScaler (Données brutes)** | 0.6372 | 0.5983 | *Référence* |
| **Avec StandardScaler (Pipeline)** | **0.9221** | **0.9343** | **+33.6 points** |

---

### 3.2 Tableau des Performances SVM sur le Jeu de Test

| Modèle | Hyperparamètres Optimaux | Macro F1 (CV 5-folds) | Accuracy (Test) | Précision Macro | Rappel Macro | **Macro F1 (Test)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVM Linéaire** | $C = 1.0$ | 0.9355 | 0.9144 | 0.9316 | 0.9245 | **0.9275** |
| **SVM RBF (Optimal)** | $C = 5, \gamma = 0.1$ | **0.9445** | **0.9247** | **0.9376** | **0.9351** | **0.9363** |

> **Constat :** Le **SVM RBF surpasse le SVM Linéaire** en obtenant **0.9363 de Macro F1** et **0.9247 d'Accuracy**, améliorant significativement la séparation des classes complexes grâce à ses frontières non-linéaires.

---

### 3.3 Détail par Variété (SVM RBF sur le Test Set)

```
              precision    recall  f1-score   support

    BARBUNYA     0.9405    0.8943    0.9168       265
      BOMBAY     1.0000    1.0000    1.0000       104
        CALI     0.9288    0.9479    0.9382       326
    DERMASON     0.9213    0.9238    0.9225       709
       HOROZ     0.9576    0.9430    0.9502       386
       SEKER     0.9482    0.9557    0.9519       406
        SIRA     0.8665    0.8805    0.8735       527

    accuracy                         0.9247      2723
   macro avg     0.9376    0.9351    0.9363      2723
weighted avg     0.9250    0.9247    0.9248      2723
```

---

## 4. Analyse Morphologique des Confusions

1. **Reconnaissance parfaite de Bombay ($F_1 = 1.0000$) :**
   En raison de sa surface gigantesque (`Area` moyenne > 170 000 contre ~50 000 pour les autres), la variété *Bombay* est isolée dans l'espace géométrique et séparée sans la moindre ambiguïté.
2. **Chevauchement résiduel Dermason / Sira :**
   La quasi-totalité des confusions restantes se produit entre *Dermason* et *Sira* (ainsi que quelques confusions mineures entre *Barbunya* et *Cali*). Ces variétés possèdent des ratios d'aspect, compacités et facteurs de forme très proches.
3. **Pourquoi le SVM RBF est supérieur :**
   Les frontières linéaires de la régression logistique tranchent de manière rigide dans les zones d'imbrication dense. Le noyau Gaussien crée une enveloppe souple autour des amas de données, augmentant le F1-score sur *Sira* de **0.8625** à **0.8735** et sur *Dermason* de **0.9154** à **0.9225**.

---

## 5. Structure du Dossier

```
projet_machine_learning/SVM/
├── Dry_Bean_Dataset.csv          # Données téléchargées directement depuis UCI
├── svm_classification.ipynb      # Notebook complet, exécuté et documenté
├── results_svm.json              # Métriques exportées pour le tableau de synthèse d'équipe
└── README.md                     # Ce fichier de synthèse
```

---

## 6. Instructions d'Exécution

```bash
# Se placer dans le dossier SVM
cd projet_machine_learning/SVM

# Ouvrir et exécuter le notebook dans VS Code ou Jupyter
jupyter notebook svm_classification.ipynb
```
