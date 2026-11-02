# Dry Bean Classification: Logistic Regression

Machine Learning course, Mini-Project 

## 1. Project Overview

The goal is to recognize **7 varieties of dry beans** (Seker, Barbunya, Bombay, Cali, Dermosan, Horoz, Sira) from **16 morphological features** extracted from seed images. We work directly on the tabular data, not on the images.

This is a **multiclass supervised classification** problem.

**Central question:** Which varieties does the model distinguish well, and which ones remain the hardest to recognize?

## 2. Dataset

| Item | Description |
|---|---|
| Source | Dry Bean Dataset, UCI Machine Learning Repository |
| Link | https://archive.ics.uci.edu/dataset/602/dry |
| Size | 13,611 observations |
| Features | 16 numerical variables (Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness, ShapeFactor1 to 4) |
| Target | `Class` (7 varieties) |

## 3. Method

The pipeline follows the mandatory protocol of the project:

1. **Load and explore** the data: shape, missing values, class distribution, plots.
2. **Stratified split** 80% train / 20% test, with a fixed random seed (`random_state=42`).
3. **Preprocessing:** standardization with `StandardScaler`, inside a scikit-learn `Pipeline` (no data leakage).
4. **Model:** multiclass Logistic Regression.
5. **Hyperparameter tuning:** `GridSearchCV` on the **train set only**, 5-fold cross-validation, scored with **macro F1**. Parameter searched: `C` in [0.01, 0.1, 1, 10, 100].
6. **Final evaluation:** the best model is evaluated **once** on the test set.
7. **Interpretation:** analysis of the confusion matrix and the weakest classes.

## 4. Evaluation Metrics

- Accuracy
- Precision, Recall, F1-score (**macro** and **per class**)
- **7 x 7 confusion matrix**

Macro F1 is the main metric: it gives the same weight to every class, including small ones like Bombay, so it is not hidden by the majority classes.

## 5. Repository Structure

```
.
├── Dry_Bean_Dataset.csv          # dataset (put your CSV here)
├── logreg_haricots_csv.py        # full code: Logistic Regression, A to Z
└── README.md
```

## 6. How to Run

**Requirements:** Python 3.9+

```bash
pip install pandas scikit-learn matplotlib seaborn
```

**Run the script:**

```bash
python logreg_haricots_csv.py
```

Or open the file in Jupyter / VS Code: it is split into cells (`# %%`).

> Make sure the CSV file name in the script matches your file. If the columns are not split correctly, use `pd.read_csv("file.csv", sep=";")`.

## 7. What the Script Produces

- Dataset summary (shape, missing values, class counts)
- Bar plot of the number of beans per variety
- Boxplot of a morphological feature by variety
- Best value of `C` and cross-validation macro F1
- Test metrics: accuracy, macro precision, macro recall, macro F1
- Classification report (metrics per class)
- Confusion matrix (7 x 7)

## 8. Results

Fill in after running the code.

| Metric | Value |
|---|---|
| Best `C` | |
| Macro F1 (cross-validation) | |
| Accuracy (test) | |
| Macro F1 (test) | |

**Hardest classes:** *(to complete from the confusion matrix)*

## 9. Discussion Points

- Which varieties are confused with each other, and why (similar shapes and sizes).
- Effect of standardization on Logistic Regression.
- Class imbalance (Bombay is much smaller than the others).
- Strong correlation between several features (Area, Perimeter, ConvexArea, EquivDiameter).

## 10. Limitations

- A single train/test split, so results depend on the seed.
- Only one model family (linear), so non-linear boundaries between close varieties are not captured.
- Features are pre-extracted from images, so no direct use of the images.

## 11. Author

Name: *(your name)*
Course: Machine Learning, INSEA
