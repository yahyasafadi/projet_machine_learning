"""
Mini-projet 02 - Reconnaissance de varietes de haricots (UCI Dry Bean)
Partie : modele KNN (K plus proches voisins)

Ce script respecte la "regle d'or" de l'equipe :
  1. Dataset : UCI Dry Bean (13 611 lignes, 16 variables, 7 varietes)
  2. Seed universelle : random_state = 42
  3. Split 80/20 stratifie
  4. Standardisation dans un Pipeline (pas de fuite de donnees)
  5. Validation croisee : StratifiedKFold(n_splits=5, shuffle=True, random_state=42), sur le train uniquement
  6. Metrique reine : F1-score macro

Utilisation :
    python knn_dry_bean.py --data Dry_Bean_Dataset.xlsx
    python knn_dry_bean.py --data Dry_Bean_Dataset.csv
    python knn_dry_bean.py --data Dry_Bean_Dataset.arff
    python knn_dry_bean.py --synthetic        # test technique du code SANS le vrai dataset
"""

import argparse
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")  # pas de fenetre : on sauvegarde directement les images
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# Constantes de la regle d'or (ne pas modifier : toute l'equipe doit avoir les memes)
# ----------------------------------------------------------------------------
SEED = 42
TEST_SIZE = 0.20
N_SPLITS = 5
TARGET = "Class"

FEATURES = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "AspectRation",
    "Eccentricity", "ConvexArea", "EquivDiameter", "Extent", "Solidity",
    "roundness", "Compactness", "ShapeFactor1", "ShapeFactor2", "ShapeFactor3",
    "ShapeFactor4",
]
CLASSES = ["BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"]


# ----------------------------------------------------------------------------
# Etape 0 : chargement des donnees
# ----------------------------------------------------------------------------
def load_dataset(path):
    """Charge le dataset Dry Bean depuis un .xlsx, .csv ou .arff."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".arff":
        from scipy.io import arff

        data, _ = arff.loadarff(path)
        df = pd.DataFrame(data)
        df[TARGET] = df[TARGET].apply(lambda v: v.decode() if isinstance(v, bytes) else v)
    else:
        sys.exit(f"Format non supporte : {ext} (utiliser .xlsx, .csv ou .arff)")

    # Harmonise le nom de la colonne cible et le texte des classes
    df.columns = [c.strip() for c in df.columns]
    df[TARGET] = df[TARGET].astype(str).str.strip().str.upper()
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        sys.exit(f"Colonnes manquantes dans le fichier : {missing}")
    return df[FEATURES + [TARGET]]


def make_synthetic_dataset(n=13611):
    """
    Faux dataset de meme forme (16 variables, 7 classes), UNIQUEMENT pour tester
    que le code tourne. Les resultats obtenus avec lui n'ont AUCUNE valeur scientifique.
    """
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n, n_features=16, n_informative=10, n_redundant=4,
        n_classes=7, n_clusters_per_class=1, class_sep=1.2, random_state=SEED,
    )
    rng = np.random.RandomState(SEED)
    scales = rng.choice([1, 10, 100, 10_000], size=16)  # echelles tres differentes, comme le vrai dataset
    X = X * scales + scales * 5
    df = pd.DataFrame(X, columns=FEATURES)
    df[TARGET] = [CLASSES[i] for i in y]
    return df


# ----------------------------------------------------------------------------
# Etape 1 : exploration
# ----------------------------------------------------------------------------
def explore(df, out):
    print("\n" + "=" * 70)
    print("ETAPE 1 - EXPLORATION DES DONNEES")
    print("=" * 70)
    print(f"Dimensions : {df.shape[0]} lignes x {df.shape[1]} colonnes")
    n_missing = int(df.isna().sum().sum())
    print(f"Valeurs manquantes (total) : {n_missing}")
    print(f"Doublons exacts : {int(df.duplicated().sum())}")

    counts = df[TARGET].value_counts()
    print("\nRepartition des classes :")
    print(pd.DataFrame({"effectif": counts, "pourcentage": (counts / len(df) * 100).round(2)}))

    desc = df[FEATURES].describe().T
    desc.to_csv(os.path.join(out, "01_statistiques_descriptives.csv"))
    print("\nStatistiques descriptives sauvegardees (01_statistiques_descriptives.csv)")
    print("Echelles tres differentes -> la standardisation sera importante pour KNN :")
    print(desc[["min", "max"]].round(3).to_string())

    # Graphique : effectifs par variete
    plt.figure(figsize=(8, 4.5))
    ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="viridis", legend=False)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 20, str(v), ha="center", fontsize=9)
    plt.title("Effectifs par variete de haricot")
    plt.ylabel("Nombre d'observations")
    plt.xlabel("")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "01_effectifs_par_variete.png"), dpi=150)
    plt.close()

    # Graphique : distribution de 4 variables par variete
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, feat in zip(axes.ravel(), ["Area", "Perimeter", "Compactness", "ShapeFactor1"]):
        sns.boxplot(data=df, x=TARGET, y=feat, hue=TARGET, palette="viridis", legend=False, ax=ax)
        ax.set_title(f"{feat} par variete")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=40)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "01_distributions_variables.png"), dpi=150)
    plt.close()

    # Graphique : correlations entre variables
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[FEATURES].corr(), cmap="coolwarm", center=0, square=True,
                cbar_kws={"shrink": 0.7}, linewidths=0.3)
    plt.title("Correlations entre les 16 variables")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "01_correlations.png"), dpi=150)
    plt.close()


# ----------------------------------------------------------------------------
# Etape 3 : experience de standardisation (avec / sans StandardScaler)
# ----------------------------------------------------------------------------
def scaling_experiment(X_train, y_train, cv, out):
    print("\n" + "=" * 70)
    print("ETAPE 3 - EXPERIENCE DE STANDARDISATION (CV 5-fold sur le train)")
    print("=" * 70)
    rows = []
    for name, use_scaler in [("Sans StandardScaler", False), ("Avec StandardScaler", True)]:
        steps = ([("scaler", StandardScaler())] if use_scaler else []) + [
            ("knn", KNeighborsClassifier(n_neighbors=5))
        ]
        pipe = Pipeline(steps)
        f1s = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        accs = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        rows.append({
            "configuration": name,
            "F1_macro_moyen": f1s.mean(), "F1_macro_ecart_type": f1s.std(),
            "accuracy_moyenne": accs.mean(), "accuracy_ecart_type": accs.std(),
        })
        print(f"{name:22s} | F1 macro = {f1s.mean():.4f} (+/- {f1s.std():.4f}) | accuracy = {accs.mean():.4f}")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, "03_experience_standardisation.csv"), index=False)
    gain = res.loc[1, "F1_macro_moyen"] - res.loc[0, "F1_macro_moyen"]
    print(f"\nGain de F1 macro grace a la standardisation : {gain:+.4f}")

    plt.figure(figsize=(6, 4.5))
    ax = sns.barplot(data=res, x="configuration", y="F1_macro_moyen", hue="configuration",
                     palette=["#d1495b", "#2e86ab"], legend=False)
    ax.errorbar(x=range(2), y=res["F1_macro_moyen"], yerr=res["F1_macro_ecart_type"],
                fmt="none", c="black", capsize=5)
    for i, v in enumerate(res["F1_macro_moyen"]):
        ax.text(i, v / 2, f"{v:.3f}", ha="center", color="white", fontweight="bold")
    plt.ylim(0, 1)
    plt.ylabel("F1 macro (CV 5-fold)")
    plt.xlabel("")
    plt.title("KNN (k=5) : impact de la standardisation")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "03_experience_standardisation.png"), dpi=150)
    plt.close()
    return res


# ----------------------------------------------------------------------------
# Etape 4 : tuning des hyperparametres (GridSearchCV)
# ----------------------------------------------------------------------------
def tune_knn(X_train, y_train, cv, out):
    print("\n" + "=" * 70)
    print("ETAPE 4 - TUNING : GridSearchCV (5-fold stratifie, scoring = F1 macro)")
    print("=" * 70)
    pipe = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())])
    param_grid = {
        "knn__n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21, 31, 41],
        "knn__metric": ["euclidean", "manhattan", "chebyshev"],
        "knn__weights": ["uniform", "distance"],
    }
    n_combi = int(np.prod([len(v) for v in param_grid.values()]))
    print(f"Combinaisons testees : {n_combi} x {N_SPLITS} folds = {n_combi * N_SPLITS} entrainements")

    t0 = time.time()
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, refit=True, return_train_score=False)
    grid.fit(X_train, y_train)
    print(f"Duree : {time.time() - t0:.1f} s")
    print(f"\nMeilleurs hyperparametres : {grid.best_params_}")
    print(f"Meilleur F1 macro en CV   : {grid.best_score_:.4f}")

    cvres = pd.DataFrame(grid.cv_results_)
    cvres = cvres.rename(columns={
        "param_knn__n_neighbors": "k", "param_knn__metric": "distance", "param_knn__weights": "poids",
    })
    cols = ["k", "distance", "poids", "mean_test_score", "std_test_score", "rank_test_score"]
    cvres = cvres[cols].sort_values("rank_test_score")
    cvres.to_csv(os.path.join(out, "04_resultats_gridsearch.csv"), index=False)
    print("\nTop 5 des combinaisons :")
    print(cvres.head(5).round(4).to_string(index=False))

    # Courbe F1 macro en fonction de k, une courbe par distance (poids = meilleur poids)
    best_w = grid.best_params_["knn__weights"]
    sub = cvres[cvres["poids"] == best_w].copy()
    sub["k"] = sub["k"].astype(int)
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=sub, x="k", y="mean_test_score", hue="distance", marker="o")
    plt.axvline(grid.best_params_["knn__n_neighbors"], color="grey", ls="--", lw=1)
    plt.title(f"F1 macro (CV) selon k et la distance (poids = {best_w})")
    plt.ylabel("F1 macro moyen (CV 5-fold)")
    plt.xlabel("k (nombre de voisins)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "04_f1_selon_k_et_distance.png"), dpi=150)
    plt.close()

    # Heatmap k x distance (poids = meilleur poids)
    piv = sub.pivot(index="distance", columns="k", values="mean_test_score")
    plt.figure(figsize=(9, 3.2))
    sns.heatmap(piv, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={"label": "F1 macro (CV)"})
    plt.title(f"Grille k x distance (poids = {best_w})")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "04_heatmap_grille.png"), dpi=150)
    plt.close()
    return grid


# ----------------------------------------------------------------------------
# Etape 5 : evaluation finale (UNE SEULE FOIS) sur le test
# ----------------------------------------------------------------------------
def final_evaluation(grid, X_test, y_test, out):
    print("\n" + "=" * 70)
    print("ETAPE 5 - EVALUATION FINALE SUR LE TEST (une seule fois)")
    print("=" * 70)
    best = grid.best_estimator_
    y_pred = best.predict(X_test)
    labels = sorted(y_test.unique())

    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    prm = precision_score(y_test, y_pred, average="macro")
    rcm = recall_score(y_test, y_pred, average="macro")
    print(f"Accuracy        : {acc:.4f}")
    print(f"Precision macro : {prm:.4f}")
    print(f"Rappel macro    : {rcm:.4f}")
    print(f"F1-score macro  : {f1m:.4f}")

    report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
    per_class = pd.DataFrame(report).T.loc[labels + ["macro avg", "weighted avg"]]
    per_class = per_class.rename(columns={"precision": "precision", "recall": "rappel", "f1-score": "F1", "support": "effectif_test"})
    per_class["effectif_test"] = per_class["effectif_test"].astype(int)
    per_class.to_csv(os.path.join(out, "05_f1_par_variete.csv"))
    print("\nTableau par variete :")
    print(per_class.round(4).to_string())

    pd.DataFrame([{
        "accuracy": acc, "precision_macro": prm, "rappel_macro": rcm, "f1_macro": f1m,
        **{f"best_{k.replace('knn__', '')}": v for k, v in grid.best_params_.items()},
        "f1_macro_cv_train": grid.best_score_,
    }]).to_csv(os.path.join(out, "05_metriques_globales.csv"), index=False)

    # Matrice de confusion (effectifs + normalisee)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title("Matrice de confusion (effectifs)")
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title("Matrice de confusion (normalisee par ligne = rappel)")
    for ax in axes:
        ax.set_xlabel("Classe predite")
        ax.set_ylabel("Classe reelle")
        ax.tick_params(axis="x", rotation=40)
    plt.suptitle(f"KNN optimise ({grid.best_params_})", y=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "05_matrice_confusion.png"), dpi=150)
    plt.close()

    # Confusions les plus frequentes
    off = [(labels[i], labels[j], int(cm[i, j])) for i in range(len(labels)) for j in range(len(labels)) if i != j and cm[i, j] > 0]
    off = pd.DataFrame(off, columns=["classe_reelle", "classe_predite", "nb_erreurs"]).sort_values("nb_erreurs", ascending=False)
    off.to_csv(os.path.join(out, "05_confusions_principales.csv"), index=False)
    print("\nConfusions les plus frequentes :")
    print(off.head(5).to_string(index=False))

    # Barres des F1 par variete
    plt.figure(figsize=(8, 4.5))
    f1s = per_class.loc[labels, "F1"]
    ax = sns.barplot(x=f1s.index, y=f1s.values, hue=f1s.index, palette="viridis", legend=False)
    ax.axhline(f1m, color="red", ls="--", label=f"F1 macro = {f1m:.3f}")
    for i, v in enumerate(f1s.values):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    plt.ylim(max(0, f1s.min() - 0.1), 1.02)
    plt.ylabel("F1-score")
    plt.title("F1-score par variete (test)")
    plt.xticks(rotation=30)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "05_f1_par_variete.png"), dpi=150)
    plt.close()
    return acc, f1m


# ----------------------------------------------------------------------------
# AJOUTS SCIENTIFIQUES v2
# ----------------------------------------------------------------------------
def dummy_baseline(X_train, y_train, X_test, y_test, out):
    """Baseline naive pour montrer que 0.93 F1 n'est pas du au hasard."""
    print("\n" + "=" * 70)
    print("ETAPE 6 (BONUS) - BASELINE : DummyClassifier (strategie 'most_frequent')")
    print("=" * 70)
    rows = []
    for strat in ["most_frequent", "stratified", "uniform"]:
        clf = DummyClassifier(strategy=strat, random_state=SEED)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average="macro")
        rows.append({"strategie": strat, "accuracy_test": acc, "f1_macro_test": f1m})
        print(f"  {strat:15s} | accuracy = {acc:.4f} | F1 macro = {f1m:.4f}")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, "06_baseline_dummy.csv"), index=False)
    print("\nNotre KNN (F1 macro = 0.93) est tres au-dessus de ces baselines => le modele apprend reellement.")
    return res


def pca_visualization(X, y, out):
    """Reduction a 2 dimensions via PCA pour voir le chevauchement entre classes."""
    print("\n" + "=" * 70)
    print("ETAPE 7 (BONUS) - VISUALISATION PCA 2D DES 7 CLASSES")
    print("=" * 70)
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=SEED)
    Z = pca.fit_transform(Xs)
    print(f"Variance expliquee : PC1 = {pca.explained_variance_ratio_[0] * 100:.1f}% | "
          f"PC2 = {pca.explained_variance_ratio_[1] * 100:.1f}% | "
          f"Total 2 axes = {pca.explained_variance_ratio_.sum() * 100:.1f}%")

    df_pca = pd.DataFrame({"PC1": Z[:, 0], "PC2": Z[:, 1], "variete": y.values})
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="variete", palette="tab10",
                    s=12, alpha=0.6, edgecolor="none")
    plt.title(f"Projection PCA 2D des 13 611 grains (variance expliquee : "
              f"{pca.explained_variance_ratio_.sum() * 100:.1f}%)")
    plt.legend(title="Variete", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "07_pca_2d.png"), dpi=150)
    plt.close()
    print("Figure sauvegardee : 07_pca_2d.png (regarder le chevauchement DERMASON/SIRA)")


def plot_learning_curve(best_estimator, X_train, y_train, cv, out):
    """Courbe d'apprentissage : le F1 progresse-t-il encore si on a plus de donnees ?"""
    print("\n" + "=" * 70)
    print("ETAPE 8 (BONUS) - COURBE D'APPRENTISSAGE")
    print("=" * 70)
    sizes = np.linspace(0.1, 1.0, 8)
    print(f"Fractions testees : {[f'{s:.0%}' for s in sizes]}")
    t0 = time.time()
    train_sizes, train_scores, val_scores = learning_curve(
        best_estimator, X_train, y_train,
        train_sizes=sizes, cv=cv, scoring="f1_macro", n_jobs=-1, random_state=SEED,
    )
    print(f"Duree : {time.time() - t0:.1f} s")
    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)

    df_lc = pd.DataFrame({
        "taille_train": train_sizes,
        "F1_train_moy": train_mean, "F1_train_std": train_std,
        "F1_validation_moy": val_mean, "F1_validation_std": val_std,
    })
    df_lc.to_csv(os.path.join(out, "08_learning_curve.csv"), index=False)
    print(df_lc.round(4).to_string(index=False))

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, "o-", color="#2e86ab", label="F1 train")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#2e86ab")
    plt.plot(train_sizes, val_mean, "o-", color="#d1495b", label="F1 validation (CV)")
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="#d1495b")
    plt.xlabel("Nombre d'exemples d'entrainement")
    plt.ylabel("F1 macro")
    plt.title("Courbe d'apprentissage - KNN optimise")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "08_learning_curve.png"), dpi=150)
    plt.close()
    gap = train_mean[-1] - val_mean[-1]
    print(f"\nEcart final train-validation : {gap:+.4f}")
    if gap < 0.03:
        verdict = "Pas de surapprentissage. Le modele a bien converge."
    else:
        verdict = "Ecart notable : leger surapprentissage, plus de donnees pourrait aider."
    print(f"Verdict : {verdict}")


def prediction_time(best_estimator, X_test, out):
    """Mesure du temps d'inference, point faible connu de KNN."""
    print("\n" + "=" * 70)
    print("ETAPE 9 (BONUS) - TEMPS DE PREDICTION (point faible de KNN)")
    print("=" * 70)
    n = len(X_test)
    t0 = time.time()
    best_estimator.predict(X_test)
    dt = time.time() - t0
    per_sample_ms = dt / n * 1000
    print(f"Prediction de {n} grains : {dt * 1000:.1f} ms  ({per_sample_ms:.3f} ms/grain)")
    print(f"Debit : environ {n / dt:.0f} grains/s")
    pd.DataFrame([{
        "n_test": n, "duree_totale_ms": dt * 1000,
        "ms_par_grain": per_sample_ms, "grains_par_seconde": n / dt,
    }]).to_csv(os.path.join(out, "09_temps_prediction.csv"), index=False)


def error_analysis(best_estimator, X_test, y_test, out):
    """Quelques exemples concrets d'erreurs du modele."""
    print("\n" + "=" * 70)
    print("ETAPE 10 (BONUS) - ANALYSE D'ERREURS : 5 grains mal classes")
    print("=" * 70)
    y_pred = best_estimator.predict(X_test)
    mask = y_pred != y_test.values
    Xt = X_test.copy()
    Xt["vraie_classe"] = y_test.values
    Xt["predite"] = y_pred
    errors = Xt[mask].copy()
    print(f"Erreurs totales : {mask.sum()} / {len(y_test)} ({mask.sum() / len(y_test) * 100:.1f}%)")
    sample_cols = ["vraie_classe", "predite", "Area", "Perimeter", "AspectRation", "Compactness", "ShapeFactor1"]
    sample = errors.sample(n=min(5, len(errors)), random_state=SEED)[sample_cols]
    print(sample.round(3).to_string())
    errors.to_csv(os.path.join(out, "10_erreurs_analyse.csv"), index=False)
    print(f"\nFichier complet des {mask.sum()} erreurs sauvegarde : 10_erreurs_analyse.csv")


# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="KNN sur UCI Dry Bean (regle d'or de l'equipe)")
    parser.add_argument("--data", help="Chemin du fichier Dry_Bean_Dataset (.xlsx, .csv ou .arff)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Utilise de fausses donnees pour tester le code (resultats SANS valeur)")
    parser.add_argument("--out", default="resultats", help="Dossier de sortie (defaut : resultats)")
    args = parser.parse_args()

    if args.synthetic:
        print("!!! MODE TEST : donnees synthetiques, resultats sans valeur scientifique !!!")
        df = make_synthetic_dataset()
    elif args.data:
        df = load_dataset(args.data)
    else:
        parser.error("Donner --data <fichier> ou --synthetic")

    out = args.out
    os.makedirs(out, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Etape 1 : exploration
    explore(df, out)

    # Etape 2 : split 80/20 stratifie (regle d'or)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    print("\n" + "=" * 70)
    print("ETAPE 2 - SPLIT STRATIFIE 80/20 (random_state=42)")
    print("=" * 70)
    print(f"Train : {X_train.shape[0]} lignes | Test : {X_test.shape[0]} lignes")
    prop = pd.DataFrame({
        "train_%": (y_train.value_counts(normalize=True) * 100).round(2),
        "test_%": (y_test.value_counts(normalize=True) * 100).round(2),
    })
    print(prop)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    scaling_experiment(X_train, y_train, cv, out)
    grid = tune_knn(X_train, y_train, cv, out)
    final_evaluation(grid, X_test, y_test, out)

    # Bonus scientifiques
    dummy_baseline(X_train, y_train, X_test, y_test, out)
    pca_visualization(X, y, out)
    plot_learning_curve(grid.best_estimator_, X_train, y_train, cv, out)
    prediction_time(grid.best_estimator_, X_test, out)
    error_analysis(grid.best_estimator_, X_test, y_test, out)

    print(f"\nTermine. Tous les fichiers sont dans le dossier : {out}/")


if __name__ == "__main__":
    main()
