"""Prédiction : charge models/model.joblib et prédit 0 (normal) ou 1 (attaque)."""
import os

import joblib
import pandas as pd

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHEMIN_MODELE = os.path.join(RACINE, "models", "model.joblib")
COLONNES = ["duration", "dst_port", "src_bytes", "dst_bytes", "nb_packets",
            "nb_failed_logins", "nb_conn_same_host", "syn_rate", "protocol"]

_modele = None  # le modèle est chargé une seule fois


def predict(df: pd.DataFrame) -> list:
    """df : mêmes colonnes que flux_train.csv, SANS 'attaque'.
    Retourne une liste de 0 (normal) ou 1 (attaque), une valeur par ligne."""
    global _modele
    if _modele is None:
        _modele = joblib.load(CHEMIN_MODELE)
    return [int(p) for p in _modele.predict(df[COLONNES])]


if __name__ == "__main__":
    donnees = pd.read_csv(os.path.join(RACINE, "data", "flux_train.csv"))
    resultat = predict(donnees.drop(columns=["attaque"]).head(10))
    print("Prédictions :", resultat)
    print("Réel        :", donnees["attaque"].head(10).tolist())
