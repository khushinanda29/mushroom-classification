# Mushroom Classification Project

**Team:** Query Queens  
**Members:** Khushi Nanda, Nataly Yau, Rose Joseph  

This project applies machine learning techniques to classify mushrooms as **edible or poisonous** using the **UCI Mushroom Dataset**. The dataset contains 8,124 instances with 22 categorical features describing physical characteristics of mushrooms such as cap shape, gill color, odor, habitat, and stalk properties.

The goal of this project is to compare different classification models and analyze how well they can distinguish between edible and poisonous mushrooms, as well as identify which features are most important for prediction.

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

- Loaded and explored the dataset  
- Performed **exploratory data analysis (EDA)** on all categorical features  
- Handled missing values in the **stalk-root** feature (`?`)  
- Encoded categorical variables for modeling  
- Performed **train/test split (80/20)** with stratification  
- Created visualizations to understand feature distributions  

---

# Week 2 Progress – Model Implementation and Analysis

### Decision Tree (Khushi Nanda)
- Implemented Decision Tree using:
  - **Gini impurity**
  - **Entropy (Information Gain)**
- Achieved **perfect performance (Accuracy, Precision, Recall, F1 = 1.0)**
- Generated confusion matrices showing **no misclassifications**
- Analyzed tree structure:
  - Depth = 7
  - Leaves = 20
- Conducted **overfitting analysis** using different `max_depth` values
- Extracted and interpreted **top 10 important features**

### Naive Bayes (Rose Joseph)
- Achieved:
  - Accuracy = 0.926  
  - Precision = 0.919  
  - Recall = 0.927  
  - F1-score = 0.923  
- Slightly lower performance due to **feature independence assumption**

### Random Forest (Nataly Yau)
- Achieved **perfect performance (all metrics = 1.0)**
- Demonstrated strong ability to capture complex feature relationships
- More stable due to ensemble learning

---

# Model Comparison

- **Decision Tree:** Perfect accuracy with strong interpretability  
- **Random Forest:** Perfect accuracy with improved stability and generalization  
- **Naive Bayes:** Slightly lower performance due to simplifying assumptions  

Tree-based models perform better because they can capture **relationships between features**, while Naive Bayes assumes independence.

---

# Tools and Libraries

- Python  
- pandas  
- numpy  
- scikit-learn  
- seaborn  
- matplotlib  
- streamlit   

---

# Next Steps

- Perform detailed **model comparison and evaluation**
- Analyze **misclassifications**, especially false negatives  
- Further explore **feature importance across models**
- Build a **Streamlit application** for interactive prediction
- Prepare final report and presentation

# Demo (Planned)

A simple **Streamlit app** will be developed to allow users to input mushroom characteristics and receive a prediction indicating whether the mushroom is edible or poisonous based on the trained model.