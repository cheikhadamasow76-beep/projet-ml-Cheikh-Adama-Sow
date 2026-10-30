"""Entraînement d'un détecteur d'attaques réseau.

Lancement (depuis la racine du dépôt) : python src/train.py
- compare plusieurs modèles sur un jeu de validation ;
- ré-entraîne le meilleur sur toutes les données ;
- enregistre le Pipeline dans models/model.joblib.
"""
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHEMIN_DONNEES = os.path.join(RACINE, "data", "flux_train.csv")
CHEMIN_MODELE = os.path.join(RACINE, "models", "model.joblib")

CIBLE = "attaque"
COLONNES_NUM = ["duration", "dst_port", "src_bytes", "dst_bytes", "nb_packets",
                "nb_failed_logins", "nb_conn_same_host", "syn_rate"]
COLONNES_CAT = ["protocol"]
GRAINE = 42  # pour obtenir toujours les mêmes résultats


def construire_pipeline(modele):
    """Prétraitement + modèle dans un seul Pipeline."""
    pretraitement = ColumnTransformer([
        ("num", Pipeline([
            ("imputation", SimpleImputer(strategy="median")),  # valeurs manquantes
            ("echelle", StandardScaler()),                      # mise à l'échelle
        ]), COLONNES_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), COLONNES_CAT),  # protocol
    ])
    return Pipeline([("pretraitement", pretraitement), ("modele", modele)])


def main():
    df = pd.read_csv(CHEMIN_DONNEES)
    print("Taille :", df.shape)
    print("Valeurs manquantes :\n", df.isna().sum()[df.isna().sum() > 0])
    print("Répartition de la cible :\n", df[CIBLE].value_counts(normalize=True).round(3))

    X = df[COLONNES_NUM + COLONNES_CAT]
    y = df[CIBLE]

    # stratify=y garde la même proportion d'attaques dans train et validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=GRAINE)

    modeles = {
        "Régression logistique": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Arbre de décision": DecisionTreeClassifier(max_depth=6, class_weight="balanced",
                                                    random_state=GRAINE),
        "Forêt aléatoire": RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                                  random_state=GRAINE),
    }

    scores = {}
    for nom, modele in modeles.items():
        pipe = construire_pipeline(modele)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_val)
        scores[nom] = f1_score(y_val, pred, pos_label=1)
        print(f"\n===== {nom} =====")
        print(classification_report(y_val, pred, target_names=["normal", "attaque"], digits=3))
        print("Matrice de confusion [[VN FP] [FN VP]] :\n", confusion_matrix(y_val, pred))

    meilleur = max(scores, key=scores.get)
    print("\nF1 (attaque) par modèle :", {k: round(v, 3) for k, v in scores.items()})
    print("Modèle retenu :", meilleur)

    # Ré-entraînement du meilleur modèle sur TOUTES les données
    final = construire_pipeline(modeles[meilleur])
    final.fit(X, y)
    os.makedirs(os.path.dirname(CHEMIN_MODELE), exist_ok=True)
    joblib.dump(final, CHEMIN_MODELE)
    print("Modèle enregistré dans", CHEMIN_MODELE)


if __name__ == "__main__":
    main()
