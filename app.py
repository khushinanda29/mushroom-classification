import streamlit as st
import pickle
import numpy as np

# Load models and encoders
dt = pickle.load(open('models/decision_tree.pkl', 'rb'))
rf = pickle.load(open('models/random_forest.pkl', 'rb'))
nb = pickle.load(open('models/naive_bayes.pkl', 'rb'))
label_encoders = pickle.load(open('models/label_encoders.pkl', 'rb'))

# Full label mappings
label_maps = {
    'class': {'e': 'edible', 'p': 'poisonous'},
    'cap-shape': {'b': 'bell', 'c': 'conical', 'f': 'flat', 'k': 'knobbed', 's': 'sunken', 'x': 'convex'},
    'cap-surface': {'f': 'fibrous', 'g': 'grooves', 's': 'smooth', 'y': 'scaly'},
    'cap-color': {'b': 'buff', 'c': 'cinnamon', 'e': 'red', 'g': 'gray', 'n': 'brown', 'p': 'pink', 'r': 'green', 'u': 'purple', 'w': 'white', 'y': 'yellow'},
    'bruises': {'f': 'no', 't': 'yes'},
    'odor': {'a': 'almond', 'c': 'creosote', 'f': 'foul', 'l': 'anise', 'm': 'musty', 'n': 'none', 'p': 'pungent', 's': 'spicy', 'y': 'fishy'},
    'gill-attachment': {'a': 'attached', 'f': 'free'},
    'gill-spacing': {'c': 'close', 'w': 'crowded'},
    'gill-size': {'b': 'broad', 'n': 'narrow'},
    'gill-color': {'b': 'buff', 'e': 'red', 'g': 'gray', 'h': 'chocolate', 'k': 'black', 'n': 'brown', 'o': 'orange', 'p': 'pink', 'r': 'green', 'u': 'purple', 'w': 'white', 'y': 'yellow'},
    'stalk-shape': {'e': 'enlarging', 't': 'tapering'},
    'stalk-root': {'b': 'bulbous', 'c': 'club', 'e': 'equal', 'r': 'rooted', '?': 'missing'},
    'stalk-surface-above-ring': {'f': 'fibrous', 'k': 'silky', 's': 'smooth', 'y': 'scaly'},
    'stalk-surface-below-ring': {'f': 'fibrous', 'k': 'silky', 's': 'smooth', 'y': 'scaly'},
    'stalk-color-above-ring': {'b': 'buff', 'c': 'cinnamon', 'e': 'red', 'g': 'gray', 'n': 'brown', 'o': 'orange', 'p': 'pink', 'w': 'white', 'y': 'yellow'},
    'stalk-color-below-ring': {'b': 'buff', 'c': 'cinnamon', 'e': 'red', 'g': 'gray', 'n': 'brown', 'o': 'orange', 'p': 'pink', 'w': 'white', 'y': 'yellow'},
    'veil-type': {'p': 'partial', 'u': 'universal'},
    'veil-color': {'n': 'brown', 'o': 'orange', 'w': 'white', 'y': 'yellow'},
    'ring-number': {'n': 'none', 'o': 'one', 't': 'two'},
    'ring-type': {'e': 'evanescent', 'f': 'flaring', 'l': 'large', 'n': 'none', 'p': 'pendant'},
    'spore-print-color': {'b': 'buff', 'h': 'chocolate', 'k': 'black', 'n': 'brown', 'o': 'orange', 'r': 'green', 'u': 'purple', 'w': 'white', 'y': 'yellow'},
    'population': {'a': 'abundant', 'c': 'clustered', 'n': 'numerous', 's': 'scattered', 'v': 'several', 'y': 'solitary'},
    'habitat': {'d': 'woods', 'g': 'grasses', 'l': 'leaves', 'm': 'meadows', 'p': 'paths', 'u': 'urban', 'w': 'waste'}
}

st.title("🍄 Mushroom Classification App")
st.write("Select the mushroom characteristics below to predict whether it is edible or poisonous.")

# Model selector
model_choice = st.selectbox("Select Model", ["Decision Tree", "Random Forest", "Naive Bayes"])

# Feature inputs
st.subheader("Mushroom Features")
user_input = {}
for col, le in label_encoders.items():
    if col in ["class", "veil-type"]:
        continue
    mapping = label_maps.get(col, {})
    options = le.classes_
    display_options = [mapping.get(o, o) for o in options]
    selected = st.selectbox(f"{col}", display_options)
    reverse_map = {v: k for k, v in mapping.items()}
    user_input[col] = reverse_map.get(selected, selected)

# Predict button
if st.button("Predict"):
    encoded = [label_encoders[col].transform([val])[0] for col, val in user_input.items()]
    input_array = np.array(encoded).reshape(1, -1)

    if model_choice == "Decision Tree":
        prediction = dt.predict(input_array)[0]
    elif model_choice == "Random Forest":
        prediction = rf.predict(input_array)[0]
    else:
        prediction = nb.predict(input_array)[0]

    result = label_encoders["class"].inverse_transform([prediction])[0]

    if result == "p":
        st.error("⚠️ This mushroom is POISONOUS!")
    else:
        st.success("✅ This mushroom is EDIBLE!")