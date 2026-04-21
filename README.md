# Mushroom Classification – Machine Learning Project

This project trains and evaluates machine learning models to classify mushrooms as edible or poisonous using the UCI Mushroom Dataset. Three classifiers are implemented and compared: Decision Tree, Random Forest, and Naive Bayes.

---

## System Requirements

- Python 3.11
- pandas 2.0+
- numpy 2.4.4
- scikit-learn 1.8
- matplotlib 3.10.8
- seaborn 0.13.2
- streamlit 1.56.0

---

## Project Structure

```
sources/
└── mushroom-classification/
    ├── data/
    │   ├── raw/
    │   │   └── mushrooms.csv                        # Raw UCI Mushroom Dataset
    │   └── processed/
    │       └── processed_mushrooms.csv              # Generated after running EDA notebook
    ├── models/
    │   ├── decision_tree.pkl                        # Generated after running model comparison notebook
    │   ├── random_forest.pkl
    │   ├── naive_bayes.pkl
    │   └── label_encoders.pkl
    ├── notebooks/
    │   ├── 01_eda.ipynb                             # EDA and preprocessing
    │   ├── 02_decision_tree.ipynb                   # Decision Tree classifier
    │   ├── 03_random_forest.ipynb                   # Random Forest classifier
    │   ├── 04_naive_bayes.ipynb                     # Naive Bayes classifier
    │   ├── 05_model_comparison.ipynb                # Cross-model comparison
    │   └── 06_example_selection.ipynb               # Extracting examples for the app
    ├── results/
    │   └── random_forest_feature_importance.csv     # Generated after running Random Forest notebook
    ├── app.py                                       # Streamlit web application
    └── README.md
```

---

## Installation

**1. Clone or download the project**

```bash
git clone https://github.com/khushinanda29/mushroom-classification.git
cd mushroom-classification
```

**3. Install dependencies**

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit
```

---

## How to Run

Notebooks must be run in order, as each step depends on outputs from the previous one.

### Step 1 – EDA and Preprocessing (`01_eda.ipynb`)

This notebook must be run first. It loads the raw dataset, handles missing values, encodes categorical features, and saves the processed dataset.

- **Input:** `data/raw/mushrooms.csv`
- **Output:** `data/processed/processed_mushrooms.csv`

### Step 2 – Decision Tree (`02_decision_tree.ipynb`)

Trains Decision Tree classifiers using both Gini impurity and Entropy. Evaluates performance, analyzes feature importance, and investigates overfitting with different max depth values.

- **Input:** `data/processed/processed_mushrooms.csv`

### Step 3 – Random Forest (`03_random_forest.ipynb`)

Trains a Random Forest classifier with 100 trees. Evaluates performance, plots feature importance, and compares accuracy across different numbers of estimators.

- **Input:** `data/processed/processed_mushrooms.csv`
- **Output:** `results/random_forest_feature_importance.csv`

### Step 4 – Naive Bayes (`04_naive_bayes.ipynb`)

Trains a Categorical Naive Bayes classifier. Evaluates performance metrics and visualizes the most influential features based on learned log probabilities.

- **Input:** `data/processed/processed_mushrooms.csv`

### Step 5 – Model Comparison (`05_model_comparison.ipynb`)

Loads all three models and compares accuracy, precision, recall, F1-score, confusion matrices, and false negative counts side by side. Also saves all trained models.

- **Input:** `data/processed/processed_mushrooms.csv`, `data/raw/mushrooms.csv`
- **Output:** `models/decision_tree.pkl`, `models/random_forest.pkl`, `models/naive_bayes.pkl`, `models/label_encoders.pkl`

### Step 6 – Example Selection (`06_example_selection.ipynb`)

Extracts real edible and poisonous mushroom examples from the dataset, validates them through the trained Decision Tree model, and converts them to human-readable format for use in the Streamlit app.

- **Input:** `data/raw/mushrooms.csv`, `models/decision_tree.pkl`, `models/label_encoders.pkl`

### Step 7 – Streamlit App (`app.py`)

The Streamlit app provides an interactive interface for classifying mushrooms. It must be run after completing Steps 1–5, as it depends on the trained Decision Tree model and label encoders saved in the `models/` folder.

**Features:**

- Select mushroom characteristics using dropdown menus grouped by category (cap, odor, gill, stalk, and other features)
- Load a pre-validated edible or poisonous example with a single button click
- Reset all inputs back to default
- Click Predict to classify the mushroom as edible or poisonous
- View the model's confidence score as a percentage
- Explore the top 10 most important features used by the Decision Tree
- Trace the exact decision path the model took to reach its prediction

**Requirements before running:**

- `models/decision_tree.pkl` must exist (generated in Step 5)
- `models/label_encoders.pkl` must exist (generated in Step 5)

**Run the app:**

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

---

## Dataset

The raw dataset (`mushrooms.csv`) is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/mushroom). It contains 8,124 samples and 22 categorical features describing physical characteristics of mushrooms such as cap shape, odor, gill color, and habitat. The target variable indicates whether a mushroom is edible (`e`) or poisonous (`p`).

> **Note:** The raw CSV does not include a header row. Column names are assigned manually during loading, as defined in the EDA notebook.

---

## Notes

- The `veil-type` feature is dropped during preprocessing because it contains only one unique value and provides no predictive information.
- Missing values in `stalk-root` (represented as `?`) are replaced with the category label `"missing"`.
- All models use the same 80/20 stratified train/test split with `random_state=42` for reproducibility.

## Contributors

Nataly Yau, Khushi Nanda, Rose Joseph
