# Projet 2 — Détecter des attaques réseau avec le Machine Learning

**Auteur : SOW Cheikh Adama**

Un modèle de machine learning décide, pour chaque flux réseau, s'il est **normal (0)** ou une
**attaque (1)** (scan de ports, force brute, DoS).

## Lancer le projet
```bash
pip install -r requirements.txt
python src/train.py      # compare les modèles et enregistre models/model.joblib
python src/predict.py    # exemple de prédiction sur 10 flux
```

Utilisation de la fonction `predict` :
```python
import pandas as pd
from src.predict import predict
df = pd.read_csv("data/flux_train.csv").drop(columns=["attaque"])
predict(df)   # liste de 0 / 1, une valeur par ligne
```

## Démarche
1. **Exploration** : 3 000 flux, 10 colonnes, 90 valeurs manquantes (colonne `duration`), environ 28 % d'attaques.
2. **Prétraitement** (dans un `Pipeline`) : imputation par la médiane, mise à l'échelle (`StandardScaler`),
   encodage one-hot de `protocol`.
3. **Modèles comparés** : régression logistique, arbre de décision, forêt aléatoire
   (`class_weight="balanced"` car les classes sont déséquilibrées).
4. **Évaluation** sur un jeu de validation de 20 % (600 flux, découpage stratifié), avec la métrique
   principale **F1 de la classe « attaque »**.
5. **Choix du meilleur modèle**, ré-entraîné sur toutes les données puis enregistré dans `models/model.joblib`.

## Résultats

**Nom / Prénom :** SOW Cheikh Adama

| Modèle | F1 (attaque) | Rappel (attaque) | Précision (attaque) |
|---|---|---|---|
| Régression logistique | 0,920 | 0,898 | 0,943 |
| Arbre de décision (profondeur 6) | 0,953 | 0,964 | 0,942 |
| **Forêt aléatoire (200 arbres)** | **0,982** | **0,976** | **0,988** |

Validation croisée à 5 plis (F1 attaque, moyenne) : régression logistique 0,896 ; arbre 0,935 ; forêt 0,959.

**Modèle retenu et pourquoi :** la forêt aléatoire. Elle a le meilleur F1 sur la validation (0,982) et en
validation croisée (0,959). Les attaques se reconnaissent par des seuils et des combinaisons de variables
(beaucoup de connexions vers la même machine, `syn_rate` élevé, peu d'octets reçus), que les arbres capturent
mieux qu'un modèle linéaire. Faire la moyenne de nombreux arbres limite aussi le surapprentissage,
ce qui aide à généraliser sur des flux jamais vus. Le score en validation croisée (0,959) est une estimation
plus prudente de la performance attendue que le score d'un seul découpage (0,982).

**Analyse des erreurs (faux négatifs / faux positifs) :**

Matrice de confusion de la forêt aléatoire sur la validation (600 flux) :

| | Prédit normal | Prédit attaque |
|---|---|---|
| **Réel normal** | 431 | 2 (faux positifs) |
| **Réel attaque** | 4 (faux négatifs) | 163 |

- **4 attaques ratées (faux négatifs)** sur 167, soit environ 2,4 %. C'est le **pire cas en sécurité** :
  une attaque non détectée passe inaperçue et peut réussir, alors qu'un faux positif ne coûte
  qu'une vérification inutile.
- **2 faux positifs** : des flux normaux signalés à tort, ce qui génère de fausses alertes.
- À comparer : la régression logistique rate 17 attaques et l'arbre de décision 6.
- Piste d'amélioration : abaisser le seuil de décision pour augmenter le rappel (moins d'attaques ratées),
  au prix de plus de fausses alertes.

## Structure du dépôt
```
├── README.md
├── requirements.txt
├── data/flux_train.csv
├── src/train.py
├── src/predict.py
└── models/model.joblib
```
