# Mushroom Classification Project

**Team:** Query Queens  
**Members:** Khushi Nanda, Nataly Yau, Rose Joseph  

This project applies machine learning techniques to classify mushrooms as **edible or poisonous** using the **UCI Mushroom Dataset**. The dataset contains 8,124 instances with 22 categorical features describing physical characteristics of mushrooms such as cap shape, gill color, odor, habitat, and stalk properties.

The goal of this project is to explore how different machine learning models can classify mushrooms based on their physical characteristics and to analyze which features are most important for predicting whether a mushroom is edible or poisonous.

---

# Dataset

**Source:** UCI Machine Learning Repository – Mushroom Dataset  

- Number of instances: **8,124**
- Number of features: **22 categorical attributes**

Each mushroom is labeled as:

- **e** – edible  
- **p** – poisonous  

The dataset includes attributes describing mushroom characteristics such as cap shape, gill color, bruising behavior, stalk structure, ring number, spore print color, population, and habitat.

---

# Week 1 Progress – Data Understanding and Preprocessing (Khushi)

- Loaded and explored the UCI Mushroom Dataset  
- Performed **exploratory data analysis (EDA)** on all categorical features  
- Investigated missing values in the **stalk-root** feature (`?`)  
- Treated missing values as a separate category for modeling  
- Encoded categorical features for machine learning models  
- Performed **train/test split (80/20)** with stratification to maintain class balance  
- Generated visualizations to analyze feature distributions and relationships with the target variable

---

# Week 2 Progress – Decision Tree Implementation and Analysis (Khushi)

- Implemented **Decision Tree classifiers** using two splitting criteria:
  - **Gini impurity**
  - **Entropy (Information Gain)**
- Evaluated model performance using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix
- Visualized and interpreted the **decision tree structure**
- Analyzed **tree depth and number of leaves**
- Conducted **overfitting analysis by training decision trees with different maximum depths (`max_depth=None`, `max_depth=5`, and `max_depth=10`) and comparing training vs testing accuracy**
- Computed and analyzed the **top 10 most important features** influencing mushroom classification
- Identified key predictive features such as **gill color, spore-print color, population, and gill size**

---

# Repository Structure
mushroom-classification
│
├── notebooks
│ ├── eda.ipynb
│ └── decision_tree.ipynb
│
├── results
│ ├── confusion_matrix_gini.png
│ ├── confusion_matrix_entropy.png
│ ├── top10_feature_importance.png
│ └── decision_tree_visualization.png
│
├── sources
│ └── data
│ ├── mushrooms.csv
│ └── processed_mushrooms.csv
│
├── requirements.txt
├── README.md
└── .gitignore

---

# Tools and Libraries

- Python  
- pandas  
- numpy  
- scikit-learn  
- seaborn  
- matplotlib  

---

# Next Steps

- Implement additional classification models (e.g., **Random Forest and Naive Bayes**)  
- Compare model performance across multiple algorithms  
- Analyze classification errors and further investigate feature importance  
- Prepare results for the **mid-project report and final evaluation**