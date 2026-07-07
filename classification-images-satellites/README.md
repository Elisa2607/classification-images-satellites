# Satellite Image Classification

Study project (BE) aiming to classify very high resolution satellite images (600×600 px, RGB) into 30 distinct classes, using convolutional neural networks and transfer learning (VGG16).

## Authors
ALEGRE, VALVERDE, MENZOU

## Context

The dataset contains satellite images spread across 30 classes (airport, bare land, beach, bridge, residential area, forest, stadium, etc.). The goal is to find the optimal model to correctly classify an image into one of these 30 categories.

The project covers the following steps:
1. **Data preparation** — train/val/test split (80/10/10)
2. **Modeling** — from a first custom CNN (~80% accuracy) to a pre-trained VGG16 model with freezing and data augmentation (~90% accuracy)
3. **Visualization** — inspection of the filters and feature maps of the convolutional layers
4. **Results analysis** — confusion matrix, classification report
5. **Deployment** — a classification API via Streamlit

## Repository structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── Rapport_BE_ALEGRE_VALVERDE_MENZOU_final.ipynb   # full report, analysis and results
├── src/
│   ├── BE_pretraitement.py   # dataset split into train/val/test
│   ├── BE_lecture.py         # image batch generation (first attempt, 600x600)
│   ├── BE_model.py           # VGG16 model building, training and evaluation
│   └── visu_car.py           # filter visualization, feature maps, results analysis
├── app/
│   └── streamlit_sat.py      # classification API (Streamlit)
└── models/
    ├── model.json            # final model architecture
    └── best_model.h5         # model weights (not included, see below)
```

## Pre-trained model

The `best_model.h5` file (~60 MB) is not versioned in this repository. You can download it here:

- [Download link](https://filesender.renater.fr/?s=download&token=f21fd843-7bc1-4c95-8d04-641eda4cb941)

Place it in the `models/` folder before running the API or the visualization script.

## Installation

```bash
git clone <repo-url>
cd <repo-name>
pip install -r requirements.txt
```

## Usage

### 1. Data preprocessing
```bash
python src/BE_pretraitement.py
```

### 2. Model training
```bash
python src/BE_model.py
```

### 3. Filter visualization and results analysis
```bash
python src/visu_car.py
```

### 4. Launch the classification API
```bash
streamlit run app/streamlit_sat.py
```

## Results

- Custom CNN model: ~80% accuracy
- VGG16 model (freezing + data augmentation): ~90% accuracy
- Best classified classes: *Viaduct*
- Worst classified classes: *School*

For the full methodology, design choices, and visualizations, see the notebook in `notebooks/`.

## Main dependencies

- TensorFlow / Keras
- Streamlit
- OpenCV
- scikit-learn
- seaborn / matplotlib
- split-folders

See `requirements.txt` for the full list.
