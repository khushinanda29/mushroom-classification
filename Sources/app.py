import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import _tree

# --- LOAD MODEL ---
dt = pickle.load(open('Sources/models/decision_tree.pkl', 'rb'))
label_encoders = pickle.load(open('Sources/models/label_encoders.pkl', 'rb'))

# --- LABEL MAPS ---
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
    'veil-color': {'n': 'brown', 'o': 'orange', 'w': 'white', 'y': 'yellow'},
    'ring-number': {'n': 'none', 'o': 'one', 't': 'two'},
    'ring-type': {'e': 'evanescent', 'f': 'flaring', 'l': 'large', 'n': 'none', 'p': 'pendant'},
    'spore-print-color': {'b': 'buff', 'h': 'chocolate', 'k': 'black', 'n': 'brown', 'o': 'orange', 'r': 'green', 'u': 'purple', 'w': 'white', 'y': 'yellow'},
    'population': {'a': 'abundant', 'c': 'clustered', 'n': 'numerous', 's': 'scattered', 'v': 'several', 'y': 'solitary'},
    'habitat': {'d': 'woods', 'g': 'grasses', 'l': 'leaves', 'm': 'meadows', 'p': 'paths', 'u': 'urban', 'w': 'waste'}
}

# --- FEATURE ORDER (CRITICAL FIX) ---
feature_names = [
    col for col in label_encoders.keys()
    if col not in ["class", "veil-type"]
]

# --- EXAMPLES ---
edible_example = {
    "cap-shape": "convex", "cap-surface": "smooth", "cap-color": "yellow",
    "bruises": "yes", "odor": "almond", "gill-attachment": "free",
    "gill-spacing": "close", "gill-size": "broad", "gill-color": "black",
    "stalk-shape": "enlarging", "stalk-root": "club",
    "stalk-surface-above-ring": "smooth", "stalk-surface-below-ring": "smooth",
    "stalk-color-above-ring": "white", "stalk-color-below-ring": "white",
    "veil-color": "white", "ring-number": "one", "ring-type": "pendant",
    "spore-print-color": "brown", "population": "numerous", "habitat": "grasses"
}

poisonous_example = {
    "cap-shape": "convex", "cap-surface": "smooth", "cap-color": "brown",
    "bruises": "yes", "odor": "pungent", "gill-attachment": "free",
    "gill-spacing": "close", "gill-size": "narrow", "gill-color": "black",
    "stalk-shape": "enlarging", "stalk-root": "equal",
    "stalk-surface-above-ring": "smooth", "stalk-surface-below-ring": "smooth",
    "stalk-color-above-ring": "white", "stalk-color-below-ring": "white",
    "veil-color": "white", "ring-number": "one", "ring-type": "pendant",
    "spore-print-color": "black", "population": "scattered", "habitat": "urban"
}

# --- SESSION STATE ---
if "example_type" not in st.session_state:
    st.session_state.example_type = None

# --- UI ---
st.title("🍄 Mushroom Classification App")
st.info("Use example buttons or customize features to explore predictions.")

# --- BUTTONS ---
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🟢 Edible Example"):
        st.session_state.example_type = "edible"
with col2:
    if st.button("🔴 Poisonous Example"):
        st.session_state.example_type = "poisonous"
with col3:
    if st.button("🔄 Reset"):
        st.session_state.example_type = None
        st.rerun()

st.markdown("---")

# --- INPUTS ---
user_input = {}

current_example = (
    edible_example if st.session_state.example_type == "edible"
    else poisonous_example if st.session_state.example_type == "poisonous"
    else {}
)

def render_section(title, cols):
    st.markdown(f"### {title}")
    for col in cols:
        le = label_encoders[col]
        mapping = label_maps.get(col, {})

        options = le.classes_
        display_options = [mapping.get(o, o) for o in options]

        index = display_options.index(current_example[col]) if col in current_example else 0
        selected = st.selectbox(col, display_options, index=index)

        reverse_map = {v: k for k, v in mapping.items()}
        user_input[col] = reverse_map.get(selected, selected)

render_section("Cap Features", ["cap-shape", "cap-surface", "cap-color"])
render_section("Odor & Bruises", ["odor", "bruises"])
render_section("Gill Features", ["gill-attachment", "gill-spacing", "gill-size", "gill-color"])
render_section("Stalk Features", [
    "stalk-shape", "stalk-root",
    "stalk-surface-above-ring", "stalk-surface-below-ring",
    "stalk-color-above-ring", "stalk-color-below-ring"
])
render_section("Other Features", ["veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"])

st.markdown("---")

# --- DECISION PATH ---
def get_decision_path(tree, input_array):
    tree_ = tree.tree_
    node_indicator = tree.decision_path(input_array)
    leaf_id = tree.apply(input_array)

    path_text = []

    for node_id in node_indicator.indices:
        if leaf_id[0] == node_id:
            values = tree_.value[node_id][0]
            pred_idx = np.argmax(values)
            pred_class = label_encoders["class"].inverse_transform([tree.classes_[pred_idx]])[0]
            path_text.append(f"➡️ Reached leaf node {node_id} → Prediction: {pred_class.upper()}")
            continue

        feature_idx = tree_.feature[node_id]
        feature = feature_names[feature_idx]
        threshold = tree_.threshold[node_id]

        value_encoded = int(input_array[0][feature_idx])

        try:
            letter = label_encoders[feature].inverse_transform([value_encoded])[0]
        except:
            letter = str(value_encoded)

        value_readable = label_maps.get(feature, {}).get(letter, letter)

        direction = "⬅️ left" if value_encoded <= threshold else "➡️ right"

        path_text.append(
            f"**{feature}** = `{value_readable}` → {direction} (threshold: {threshold:.2f})"
        )

    return path_text

# --- PREDICT ---
if st.button("🔍 Predict"):
    encoded = [
        label_encoders[col].transform([user_input[col]])[0]
        for col in feature_names
    ]

    input_array = np.array(encoded).reshape(1, -1)

    prediction = dt.predict(input_array)[0]
    probs = dt.predict_proba(input_array)[0]
    confidence = max(probs)

    result = label_encoders["class"].inverse_transform([prediction])[0]

    st.markdown("---")

    if result == "p":
        st.error("⚠️ **Result: POISONOUS**")
    else:
        st.success("✅ **Result: EDIBLE**")

    st.markdown(f"📊 **Confidence: {confidence:.2%}**")
    st.progress(int(confidence * 100))

    # --- FEATURE IMPORTANCE ---
    st.markdown("### 🌳 Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": dt.feature_importances_
    }).sort_values(by="Importance", ascending=False).head(10)

    st.bar_chart(importance_df.set_index("Feature"))

    # --- DECISION PATH ---
    st.markdown("### 🌿 Decision Path")

    path = get_decision_path(dt, input_array)
    for step in path:
        st.write(step)

    top_feature = importance_df.iloc[0]["Feature"]
    st.info(f"🔍 Most important feature: **{top_feature}**")