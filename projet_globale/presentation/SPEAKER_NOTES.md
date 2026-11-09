# Script Oral de Soutenance — 8 à 10 Minutes Chrono
## Mini-Projet 02 : Reconnaissance de Variétés de Haricots Secs (UCI Dry Bean)
### Module de Machine Learning (S5) — INSEA Rabat
* **Équipe (DSE) :** Youssef FATIMI, Hamza BALLA, Mohamed Yahya SAFADI, Ayoub ELHANAFI
* **Sous la direction académique de :** Pr. Saidi
* **Support visuel :** Présentation Beamer 12 planches (`main.pdf`), calibrée sur le rapport officiel (`rapport_latex/main.tex`).

---

### Planche 1 : Page de Garde `[00:00 - 00:45]`
* **Action du présentateur :** Salutation solennelle au professeur et présentation des membres de l'équipe.
* **Script Oral :**
  > « Bonjour Monsieur le Professeur, chers collègues. Nous avons le plaisir de vous présenter aujourd'hui les conclusions de notre travail sur la reconnaissance automatisée de variétés de haricots secs à partir du corpus de référence *UCI Dry Bean*.
  > Notre démarche a répondu avec rigueur aux exigences du cahier des charges : concevoir un protocole expérimental étanche sans aucune fuite d'information (*Zero Data Leakage*), valider scientifiquement le rôle de la standardisation par une expérience d'ablation, et benchmarker de façon équitable cinq architectures d'apprentissage supervisé. »

---

### Planche 2 : Contexte Métier & Problématique Scientifique `[00:45 - 01:30]`
* **Action :** Exposer l'enjeu agronomique et les deux questions centrales du sujet.
* **Script Oral :**
  > « Sur le plan agronomique et commercial, mélanger involontairement des variétés distinctes de haricots altère l'homogénéité de cuisson et déprécie la valeur marchande du lot. Le tri manuel étant subjectif, fastidieux et lent, notre objectif est d'automatiser cette classification par apprentissage supervisé, sans traiter directement les images mais en exploitant leurs descripteurs morphologiques.
  > Le sujet pose deux questions fondamentales auxquelles notre étude apporte des réponses quantifiées :
  > 1. *Quels modèles distinguent le mieux les sept variétés ?*
  > 2. *Quelles classes restent les plus difficiles à reconnaître ?*
  > Pour y répondre, nous avons confronté cinq familles du cours : la Régression Logistique Softmax, les K-Plus Proches Voisins, l'arbre CART, la Forêt Aléatoire et les SVM à noyau RBF. »

---

### Planche 3 : Description du Corpus UCI Dry Bean & Déséquilibre `[01:30 - 02:30]`
* **Action :** Pointer le Tableau 1 et la Figure 1 (distribution des effectifs).
* **Script Oral :**
  > « Le jeu de données officiel comprend 13 611 grains décrits par 16 variables continues : 12 caractéristiques géométriques de dimension et 4 facteurs de forme normés, sans aucune valeur manquante.
  > Comme l'illustrent le Tableau 1 et la Figure 1, le corpus présente une asymétrie de classes marquée : la variété Dermason regroupe plus d'un quart des observations (26.05 %), tandis que Bombay n'en représente que 3.84 %, soit un ratio de prévalence de 1 à 6.8.
  > Dans ces conditions, l'Accuracy brute est un piège méthodologique qui masquerait l'échec sur les classes rares. Conformément aux consignes, nous avons donc érigé le **F1-Score Macro** comme critère cardinal d'évaluation. »

---

### Planche 4 : Distributions Morphologiques & Invariants Physiques `[02:30 - 03:30]`
* **Action :** Commenter la Figure 2 (les 6 boîtes à moustaches comparatives).
* **Script Oral :**
  > « L'exploration graphique de la Figure 2 dégage quatre invariants physiques déterminants :
  > 1. Tout d'abord, **Bombay** est un colosse géométrique isolé : avec une aire médiane de 173 000 pixels carrés (contre moins de 70 000 pour les autres), il est linéairement séparable dès le départ.
  > 2. Ensuite, **Horoz** se caractérise par une forte élongation et un ratio d'aspect extrême supérieur à 2.1.
  > 3. À l'opposé, **Seker** présente une circularité et une compacité maximales proches de 0.94.
  > 4. En revanche, observez attentivement **Dermason** et **Sira** : leurs distributions d'aire, de périmètre et de compacité se chevauchent presque totalement. C'est le foyer d'erreur majeur du problème.
  > Enfin, la multicolinéarité quasi unitaire ($r \ge 0.98$) entre les variables de taille impose une régularisation de crête $L_2$. »

---

### Planche 5 : Protocole Expérimental Partagé (Règle d'Or) `[03:30 - 04:15]`
* **Action :** Insister sur la reproductibilité et la prévention du data leakage.
* **Script Oral :**
  > « Pour garantir une comparabilité absolue entre nos modèles, notre équipe a instauré un protocole commun strict :
  > Une graine aléatoire universelle fixée à 42 ; une partition étanche 80 % train (10 888 grains) et 20 % test (2 723 grains), avec une stratification rigoureuse préservée au centième de pourcent près, comme le montre le Tableau 2.
  > Surtout, nous garantissons l'absence totale de fuite d'information (*Zero Data Leakage*) : le centrage-réduction `StandardScaler` est systématiquement encapsulé dans un `Pipeline` Scikit-Learn. Il n'est appris que sur les plis d'entraînement de la validation croisée stratifiée à 5 plis.
  > Le jeu de test reste scellé et n'est sollicité qu'une seule et unique fois lors de l'évaluation finale. »

---

### Planche 6 : Expérience d'Ablation — Impact Vital du Centrage-Réduction `[04:15 - 05:15]`
* **Action :** Pointer le Tableau 3 et confronter les modèles métriques aux arbres.
* **Script Oral :**
  > « Cette planche expose le résultat méthodologique le plus saisissant de notre travail : l'expérience d'ablation sur la standardisation.
  > Regardez le Tableau 3 : sans centrage-réduction, le SVM RBF subit une chute catastrophique de 33.60 points de F1 macro, s'effondrant de 0.9343 à 0.5983 ! Le KNN perd 22.17 points et la Régression Logistique 17.85 points.
  > L'explication théorique du cours est directe : la variable `Area` varie sur une échelle de $10^5$, écrasant les facteurs de forme de l'ordre de $10^{-3}$. Sans normalisation, le calcul des distances euclidiennes n'est qu'une mesure d'aire, rendant le noyau gaussien inopérant.
  > À l'inverse, observez l'arbre CART et la Forêt Aléatoire : l'écart est strictement nul, de 0.00 point ! Les coupures de Gini reposant sur l'ordre des valeurs, toute transformation monotone strictement croissante préserve les partitions optimales. »

---

### Planche 7 : Modélisation & Optimisation des 5 Architectures `[05:15 - 06:30]`
* **Action :** Décrire succinctement les hyperparamètres optimaux retenus par GridSearchCV.
* **Script Oral :**
  > « Chaque modèle a été étalonné par `GridSearchCV` 5-plis sur l'échantillon d'entraînement :
  > 1. Pour la **Régression Logistique Softmax**, l'optimisation retient $C^* = 100$, confirmant que la standardisation conditionne idéalement la surface de perte convexe.
  > 2. Pour le **KNN**, la configuration optimale retient $k^* = 9$, la distance euclidienne et une pondération par l'inverse de la distance ($w_i = 1/d$). Cette pondération lève le flou décisionnel dans les zones frontalières denses.
  > 3. Pour l'**Arbre CART**, contraindre la profondeur à $\texttt{max\_depth}^* = 9$ prévient le sur-apprentissage précoce des arbres libres.
  > 4. Pour la **Forêt Aléatoire**, $B = 200$ arbres décorrélés avec $\sqrt{p} = 4$ variables candidates réduisent drastiquement la variance (+1.27 point de gain face à CART).
  > 5. Pour le **SVM RBF**, l'optimum conjoint $C^* = 5$ et $\gamma^* = 0.1$ courbe harmonieusement les frontières de décision dans un espace de Hilbert. »

---

### Planche 8 : Grand Benchmark Comparatif des Performances `[06:30 - 07:30]`
* **Action :** Commenter le Tableau 4 complet du grand benchmark sur le test set.
* **Script Oral :**
  > « Voici la synthèse finale évaluée sur le jeu de test indépendant scellé (2 723 grains). Trois enseignements majeurs se dégagent :
  > 1. **Le Sacre du SVM RBF :** Il s'impose champion absolu sur tous les indicateurs : **0.9363 de F1 macro**, 92.47 % d'Accuracy, 93.66 % de précision et 93.62 % de rappel.
  > 2. **L'Efficience Remarquable de la Régression Logistique :** Elle talonne le champion à seulement 0.34 point de F1 (0.9329), tout en exécutant l'inférence en seulement **2.1 millisecondes**, soit 7 fois plus vite que le SVM et 21 fois plus vite que le KNN !
  > 3. **Le Décrochage de l'Arbre CART :** L'arbre individuel accuse un retard de 1.7 point (0.9194), limité par ses séparations orthogonales rigides. »

---

### Planche 9 : Analyse Désagrégée des Performances par Variété `[07:30 - 08:30]`
* **Action :** Analyser la Figure 3 (F1 par classe pour les 5 modèles).
* **Script Oral :**
  > « La Figure 3 détaille les performances variété par variété :
  > Sur **Bombay**, les cinq modèles réussissent un sans-faute parfait de $F_1 = \mathbf{1.0000}$ (zéro erreur).
  > Sur **Horoz** et **Seker**, la stabilité est excellente ($F_1 \ge 0.94$).
  > **Cali** et **Barbunya** atteignent 0.93 à 0.94.
  > Le véritable pivot discriminatoire se situe sur **Dermason** et **Sira** : c'est précisément sur ces deux classes difficiles que le SVM RBF fait la différence en surclassant la Régression Logistique de près d'un point (+0.91 pt sur Dermason et +0.91 pt sur Sira).
  > Enfin, comparés à notre baseline factice (DummyClassifier à 0.1472), nos modèles opèrent à plus de **6.3 fois au-dessus du hasard**. »

---

### Planche 10 : Dissection des Confusions & Limite Bayesienne 2D `[08:30 - 09:30]`
* **Action :** Commenter la Figure 4 (matrices de confusion du champion).
* **Script Oral :**
  > « La Figure 4 dissèque les 205 erreurs commises par le SVM RBF sur les 2 723 grains de test (soit seulement 7.53 % d'erreur) :
  > Près de la moitié de ces erreurs (**46.8 %**) provient exclusivement de la confusion mutuelle entre **Dermason** et **Sira** (96 confusions : 52 Dermason classés Sira et 44 Sira classés Dermason).
  > La confusion secondaire concerne **Barbunya** et **Cali** (11.7 %, soit 24 grains).
  > Zéro pollution sur Bombay.
  > Nous démontrons qu'il ne s'agit pas d'une sous-optimisation algorithmique, mais d'une **limite Bayesienne intrinsèque** : dans l'espace morphologique 2D, les distributions de Dermason et Sira se chevauchent physiquement. Aucun séparateur géométrique ne peut éliminer cette ambiguïté sans données supplémentaires. »

---

### Planche 11 : Synthèse des Réponses au Cahier des Charges & Perspectives `[09:30 - 10:15]`
* **Action :** Répondre explicitement aux questions posées et dresser les perspectives.
* **Script Oral :**
  > « Pour conclure et répondre sans ambiguïté aux questions du cahier des charges :
  > 1. *Quel modèle distingue le mieux les variétés ?* Le **SVM à noyau RBF** s'adjuge le meilleur pouvoir discriminant ($F_1 = 0.9363$), mais la **Régression Logistique** ($F_1 = 0.9329$) représente le choix industriel optimal pour un trieur haute cadence grâce à sa latence record de 2.1 ms.
  > 2. *Quelles classes restent les plus difficiles ?* **Dermason** ($F_1 = 0.8936$) et **Sira** ($F_1 = 0.8778$) constituent le verrou majeur.
  > 3. *Quel est le rôle de la standardisation ?* Indispensable pour les modèles métriques (+33.6 pts pour SVM), et strictement neutre sur les arbres.
  > En perspective, l'acquisition de canaux de couleur (RGB / HSV) ou le recours à des réseaux convolutifs (CNN) sur images brutes permettra de lever définitivement le verrou Dermason/Sira en exploitant le veinage et la texture. »

---

### Planche 12 : Conclusion & Remerciements `[10:15 - 10:30]`
* **Action :** Clôturer la présentation et inviter le jury aux questions.
* **Script Oral :**
  > « Nous vous remercions chaleureusement pour votre écoute attentive et nous sommes à votre entière disposition pour répondre à vos questions. »
