# Classification des images spatiales

Projet de bureau d'étude (BE) visant à classer des images satellites de très haute résolution (600×600 px, RGB) en 30 classes distinctes, à l'aide de réseaux de neurones convolutifs et du transfer learning (VGG16).

## Auteurs
ALEGRE, VALVERDE, MENZOU

## Contexte

La base de données contient des images spatiales réparties en 30 classes (aéroport, terrain vague, plage, pont, zone résidentielle, forêt, stade, etc.). L'objectif est de trouver le modèle optimal permettant de classer correctement une image dans l'une de ces 30 catégories.

Le projet couvre les étapes suivantes :
1. **Préparation des données** — répartition train/val/test (80/10/10)
2. **Modélisation** — d'un premier CNN maison (~80% accuracy) à un modèle VGG16 pré-entraîné avec freezing et data augmentation (~90% accuracy)
3. **Visualisation** — inspection des filtres et des cartes de caractéristiques des couches convolutives
4. **Analyse des résultats** — matrice de confusion, rapport de classification
5. **Déploiement** — une API de classification via Streamlit

## Structure du dépôt

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── Rapport_BE_ALEGRE_VALVERDE_MENZOU_final.ipynb   # rapport complet, analyse et résultats
├── src/
│   ├── BE_pretraitement.py   # découpage du dataset en train/val/test
│   ├── BE_lecture.py         # génération des batchs d'images (premier essai, 600x600)
│   ├── BE_model.py           # construction, entraînement et évaluation du modèle VGG16
│   └── visu_car.py           # visualisation des filtres, cartes de caractéristiques, analyse des résultats
├── app/
│   └── streamlit_sat.py      # API de classification (Streamlit)
└── models/
    ├── model.json            # architecture du modèle final
    └── best_model.h5         # poids du modèle (non inclus, voir ci-dessous)
```

## Modèle pré-entraîné

Le fichier `best_model.h5` (~60 Mo) n'est pas versionné dans ce dépôt. Vous pouvez le télécharger ici :

👉 [Lien de téléchargement](https://filesender.renater.fr/?s=download&token=f21fd843-7bc1-4c95-8d04-641eda4cb941)

Placez-le dans le dossier `models/` avant de lancer l'API ou le script de visualisation.

> ⚠️ Ce lien peut expirer. Pour une solution pérenne, envisagez [Git LFS](https://git-lfs.github.com/) ou un stockage type Google Drive / Hugging Face Hub.

## Installation

```bash
git clone <url-du-depot>
cd <nom-du-depot>
pip install -r requirements.txt
```

## Utilisation

### 1. Prétraitement des données
```bash
python src/BE_pretraitement.py
```

### 2. Entraînement du modèle
```bash
python src/BE_model.py
```

### 3. Visualisation des filtres et analyse des résultats
```bash
python src/visu_car.py
```

### 4. Lancer l'API de classification
```bash
streamlit run app/streamlit_sat.py
```

## Résultats

- Modèle CNN maison : ~80% d'accuracy
- Modèle VGG16 (freezing + data augmentation) : ~90% d'accuracy
- Classes les mieux classées : *Viaduc*
- Classes les moins bien classées : *School*

Pour le détail complet de la démarche, des choix méthodologiques et des visualisations, voir le notebook dans `notebooks/`.

## Dépendances principales

- TensorFlow / Keras
- Streamlit
- OpenCV
- scikit-learn
- seaborn / matplotlib
- split-folders

Voir `requirements.txt` pour la liste complète.
