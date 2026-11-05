# Projet 2 — Détecter des attaques réseau avec le Machine Learning

Classification de flux réseau : **normal (0)** ou **attaque (1)** (scan de ports, force brute, DoS).

## Utilisation
```bash
pip install -r requirements.txt
python src/train.py      # compare les modèles, enregistre models/model.joblib
python src/predict.py    # test rapide de la fonction predict
```

## Démarche
- **Données** : 3000 flux, 9 variables, 27,8 % d'attaques ; 90 valeurs manquantes (`duration`).
- **Prétraitement** (dans un `Pipeline`) : imputation par la médiane, `StandardScaler`, `OneHotEncoder` sur `protocol`.
- **Validation** : split 80/20 stratifié (600 flux de validation) + validation croisée 5 plis sur la partie entraînement.
- **Déséquilibre** : `class_weight="balanced"` et métrique principale = F1 de la classe « attaque ».
- **Modèle final** : ré-entraîné sur les 3000 flux, enregistré dans `models/model.joblib`.

## Résultats

**Nom / Prénom :** LAYE SOW Omar

| Modèle | F1 (attaque) | Rappel (attaque) | Précision (attaque) | F1 CV 5 plis |
|---|---|---|---|---|
| Régression logistique | 0,920 | 0,898 | 0,943 | 0,889 |
| Arbre de décision (profondeur 6) | 0,953 | 0,964 | 0,942 | 0,934 |
| **Forêt aléatoire (300 arbres)** | **0,976** | **0,976** | **0,976** | **0,952** |

Matrices de confusion sur la validation (600 flux, 167 attaques) :

| Modèle | VN | FP | FN | VP |
|---|---|---|---|---|
| Régression logistique | 424 | 9 | 17 | 150 |
| Arbre de décision | 423 | 10 | 6 | 161 |
| Forêt aléatoire | 429 | 4 | 4 | 163 |

**Modèle retenu et pourquoi :** la forêt aléatoire, meilleure sur tous les critères (F1, rappel, précision) et aussi en validation croisée (0,952), donc pas seulement due au hasard du split. Les relations entre variables ne sont pas linéaires (ex. un nombre élevé de connexions vers la même machine *combiné* à un `syn_rate` élevé), ce que la régression logistique modélise mal, alors que les arbres capturent naturellement ces seuils et interactions. Moyenner 300 arbres réduit en plus le surapprentissage d'un arbre seul.

**Analyse des erreurs (faux négatifs / faux positifs) :**
- Forêt aléatoire : **4 faux négatifs** (attaques ratées sur 167, soit 2,4 %) et **4 faux positifs** (fausses alertes sur 433 flux normaux, soit 0,9 %).
- Un **faux négatif** est le pire cas en sécurité : une attaque passe inaperçue, donc l'intrusion, la compromission de comptes ou l'indisponibilité du service peuvent se produire sans réaction. Un faux positif coûte seulement du temps d'analyse à l'équipe SOC (fatigue d'alertes si trop nombreux).
- La régression logistique rate 17 attaques (10,2 %) : c'est le modèle le moins adapté à un IDS.
- Piste d'amélioration : si l'on veut réduire encore les faux négatifs, on peut abaisser le seuil de décision (`predict_proba` > 0,5) au prix de plus de faux positifs.
- Limite : les données semblent synthétiques ; les performances sur un trafic réel seraient probablement plus faibles.
