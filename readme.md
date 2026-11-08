# Reconnaissance de Variétés de Haricots Secs (Dry Bean Classification)

> **Mini-Projet 02 — Machine Learning** | INSEA  
> Classification supervisée multiclasse à partir de caractéristiques morphologiques de graines.

---

## 📌 Présentation du Projet

Ce projet a pour objectif de développer, comparer et analyser les performances de plusieurs modèles de Machine Learning pour classifier **7 variétés de haricots secs** (*Seker, Barbunya, Bombay, Cali, Dermason, Horoz, Sira*) à partir de **16 descripteurs morphologiques et géométriques** extraits d'images de grains, sans traitement d'image direct.

### 📊 Jeu de Données (UCI Dry Bean Dataset)
- **Observations :** 13 611 graines
- **Caractéristiques :** 16 variables numériques continues (forme, taille, convexité, compacité, etc.)
- **Cible :** `Class` (7 variétés avec léger déséquilibre de classes)
- **Valeurs manquantes :** Aucune (0 missing values)

---

## ⚙️ Protocole Expérimental Commun

Pour garantir une stricte comparabilité entre tous les algorithmes :
1. **Séparation étanche :** 80% Train / 20% Test stratifié avec graine aléatoire fixée (`random_state = 42`).
2. **Prétraitement & Fuite de données :** Standardisation (`StandardScaler`) systématiquement intégrée dans un `Pipeline` Scikit-Learn (fit sur le train uniquement).
3. **Optimisation des hyperparamètres :** Validation croisée stratifiée à 5 plis (`StratifiedKFold`, 5 splits) sur le jeu d'entraînement.
4. **Métrique d'arbitrage principale :** **F1-score Macro** (donne un poids égal à chaque variété, évitant le biais vers les classes majoritaires).
5. **Évaluation finale :** Test unique sur les 2 723 observations du jeu de test.

---

## 🏆 Synthèse Comparative des Résultats

| Modèle | Hyperparamètres Optimaux | F1 Macro (CV Train) | Accuracy (Test) | **F1-Score Macro (Test)** |
| :--- | :--- | :---: | :---: | :---: |
| **Arbre de Décision** | `max_depth = 9` | 0.9211 | 90.56% | **0.9194** |
| **K-Nearest Neighbors (KNN)** | `k = 9`, `metric = euclidean`, `weights = distance` | 0.9382 | 91.74% | **0.9300** |
| **Random Forest** | `n_estimators = 200`, `max_depth = None` | 0.9363 | 91.99% | **0.9321** |
| **Régression Logistique** | `C = 100` | 0.9370 | 92.10% | **0.9329** |
| **SVM (Noyau RBF)** 🥇 | `C = 5`, `gamma = 0.1` | **0.9445** | **92.47%** | **0.9363** |

### 💡 Constats Clés :
- **Meilleur modèle global :** Le **SVM RBF** arrive en tête (**0.9363 F1 macro**), suivi de très près par la **Régression Logistique** et le **Random Forest**.
- **Impact du scaling :** Indispensable pour KNN, Régression Logistique et SVM (+22 à +33 points de F1 macro grâce au `StandardScaler`).
- **Classes faciles vs difficiles :**
  - *Bombay* et *Seker* sont identifiées de manière quasi-parfaite (F1 > 0.99).
  - La principale source de confusion réside entre *Barbunya* et *Cali*, ainsi qu'entre *Sira* et *Dermason*, en raison de fortes similarités géométriques.

---

## 📂 Organisation du Dépôt

```
machine_learning/
│
├── README.md                           # Documentation globale du projet
│
├── DecisionTree__RandomForest/         # Pôle Arbres de décision & Forêts aléatoires
│   ├── Arbre_RandomForest.ipynb        # Notebook complet d'expérimentation et d'évaluation
│   └── resultats/                      # Visualisations de l'arbre et métriques comparatives
│
├── KNN/                                # Pôle K plus proches voisins
│   ├── knn_dry_bean.ipynb              # Notebook détaillé pas-à-pas
│   ├── knn_dry_bean.py                 # Script reproductible en ligne de commande
│   ├── Rapport_KNN.docx                # Rapport rédigé du sous-projet
│   ├── Presentation_KNN.pptx           # Support de présentation
│   └── resultats/                      # Courbes d'apprentissage, matrices, analyse d'erreurs
│
├── Logistic_regression/                # Pôle Régression Logistique Multiclasse
│   ├── regression_logistique.ipynb     # Notebook d'optimisation et analyse de régularisation
│   └── README.md                       # Documentation spécifique
│
└── SVM/                                # Pôle Support Vector Machines (Linéaire & RBF)
    ├── svm_classification.ipynb        # Notebook comparant SVM linéaire vs noyau RBF
    ├── results_svm.json                # Résultats sérialisés
    └── README.md                       # Documentation spécifique
```

---

## 🚀 Installation & Exécution Rapide

### 1. Prérequis
Python 3.9 ou supérieur.

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### 2. Lancer les notebooks
Chaque dossier contient son notebook dédié exécutable directement sous Jupyter Lab / VS Code :
```bash
jupyter notebook
```
Les jeux de données sont déjà placés dans les sous-dossiers correspondants pour assurer une exécution immédiate sans dépendance externe.

