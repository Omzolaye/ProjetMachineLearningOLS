"""Entraînement : détection d'attaques réseau (0 = normal, 1 = attaque).

Usage (depuis la racine du dépôt) : python src/train.py
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "flux_train.csv"
MODEL = ROOT / "models" / "model.joblib"
SEED = 42

NUM = ["duration", "dst_port", "src_bytes", "dst_bytes", "nb_packets",
       "nb_failed_logins", "nb_conn_same_host", "syn_rate"]
CAT = ["protocol"]


def build_pipeline(clf) -> Pipeline:
    """Prétraitement (imputation + échelle + one-hot) suivi du classifieur."""
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    return Pipeline([("prep", prep), ("clf", clf)])


def main() -> None:
    # 1. Exploration rapide
    df = pd.read_csv(DATA)
    print("Taille :", df.shape)
    print("Valeurs manquantes :\n", df.isna().sum()[df.isna().sum() > 0])
    print("Répartition attaque :\n", df["attaque"].value_counts(normalize=True).round(3))

    X, y = df.drop(columns="attaque"), df["attaque"]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    # 2. Modèles à comparer (class_weight pour les classes déséquilibrées)
    models = {
        "Régression logistique": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Arbre de décision": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=SEED),
        "Forêt aléatoire": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                  random_state=SEED, n_jobs=-1),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}
    for name, clf in models.items():
        pipe = build_pipeline(clf)
        cv_f1 = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1").mean()
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_val)
        results[name] = dict(
            f1=f1_score(y_val, pred), recall=recall_score(y_val, pred),
            precision=precision_score(y_val, pred), cv_f1=cv_f1)
        print(f"\n===== {name} =====")
        print(classification_report(y_val, pred, target_names=["normal", "attaque"], digits=3))
        tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
        print(f"Matrice de confusion : TN={tn} FP={fp} FN={fn} TP={tp}")

    print("\n===== Récapitulatif (validation) =====")
    print(pd.DataFrame(results).T.round(3))

    # 3. Meilleur modèle (F1 validation), ré-entraîné sur toutes les données
    best = max(results, key=lambda k: results[k]["f1"])
    print("\nModèle retenu :", best)
    final = build_pipeline(models[best]).fit(X, y)
    MODEL.parent.mkdir(exist_ok=True)
    joblib.dump(final, MODEL)
    print("Modèle enregistré :", MODEL)


if __name__ == "__main__":
    main()
