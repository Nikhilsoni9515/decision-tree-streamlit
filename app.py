"""Decision Tree Classifier - Streamlit UI
Based on Decison_Tree_Sol.ipynb (health_lifestyle_dataset.csv -> disease_risk).
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

DEFAULT_CSV = "health_lifestyle_dataset.csv"
DEFAULT_TARGET = "disease_risk"

st.set_page_config(page_title="Decision Tree Classifier", page_icon="🌳", layout="wide")
st.title("🌳 Decision Tree Classifier")
st.caption("Upload a dataset, pick a target column, train a tree, and make predictions.")


# ---------------------------------------------------------------- data
@st.cache_data
def load_csv(file_or_path):
    return pd.read_csv(file_or_path)


with st.sidebar:
    st.header("1. Data")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        df = load_csv(uploaded)
    elif os.path.exists(DEFAULT_CSV):
        df = load_csv(DEFAULT_CSV)
        st.info(f"Using `{DEFAULT_CSV}`")
    else:
        st.warning("Upload a CSV to get started.")
        st.stop()

    cols = list(df.columns)
    target = st.selectbox(
        "Target column (what to predict)", cols,
        index=cols.index(DEFAULT_TARGET) if DEFAULT_TARGET in cols else len(cols) - 1,
    )
    test_size = st.slider("Test size", 0.1, 0.5, 0.33, 0.01)
    random_state = st.number_input("Random state", value=42, step=1)

    st.header("2. Model")
    mode = st.radio("Training mode", ["Manual", "RandomizedSearchCV"])
    if mode == "Manual":
        criterion = st.selectbox("criterion", ["gini", "entropy", "log_loss"])
        unlimited = st.checkbox("max_depth = None (unlimited)", value=True)
        max_depth = None if unlimited else st.slider("max_depth", 1, 20, 3)
        min_samples_split = st.slider("min_samples_split", 2, 50, 2)
        min_samples_leaf = st.slider("min_samples_leaf", 1, 50, 1)
    else:
        n_iter = st.slider("n_iter (random combinations)", 5, 50, 20)
        cv = st.slider("cv (folds)", 2, 10, 5)
        st.caption("Search space: max_depth [3,5,7,9,None], min_samples_split [2,5,10], "
                   "min_samples_leaf [1,2,4], criterion [gini, entropy]")

    train_clicked = st.button("🚀 Train model", type="primary")


# ---------------------------------------------------------------- preprocessing
def preprocess(df, target):
    """Label-encode text columns (like the notebook does for `gender`)."""
    data = df.copy()
    notes = []
    for c in data.columns:  # fill missing values so training never crashes
        if data[c].isnull().any():
            if pd.api.types.is_numeric_dtype(data[c]):
                data[c] = data[c].fillna(data[c].median())
            else:
                data[c] = data[c].fillna(data[c].mode()[0])
            notes.append(c)
    encoders = {}
    for c in data.columns:
        if not pd.api.types.is_numeric_dtype(data[c]):
            le = LabelEncoder()
            data[c] = le.fit_transform(data[c].astype(str))
            encoders[c] = le
    X = data.drop(columns=[target])
    y = data[target]
    return X, y, encoders, notes


# ---------------------------------------------------------------- training
if train_clicked:
    X, y, encoders, filled = preprocess(df, target)
    if y.nunique() > 20:
        st.error(f"`{target}` has {y.nunique()} unique values - this looks like a regression "
                 "target. Pick a categorical/label column for classification.")
        st.stop()
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(random_state), stratify=stratify)

    with st.spinner("Training..."):
        best_params = None
        if mode == "Manual":
            model = DecisionTreeClassifier(
                criterion=criterion, max_depth=max_depth,
                min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                random_state=int(random_state))
            model.fit(X_train, y_train)
        else:
            grid = {"max_depth": [3, 5, 7, 9, None], "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4], "criterion": ["gini", "entropy"]}
            search = RandomizedSearchCV(
                DecisionTreeClassifier(random_state=int(random_state)), grid,
                n_iter=n_iter, cv=cv, scoring="accuracy",
                random_state=int(random_state), n_jobs=-1)
            search.fit(X_train, y_train)
            model, best_params = search.best_estimator_, search.best_params_

    y_pred = model.predict(X_test)
    st.session_state["res"] = dict(
        model=model, X=X, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        y_pred=y_pred, encoders=encoders, target=target, best_params=best_params,
        filled=filled, raw=df.drop(columns=[target]),
    )

# ---------------------------------------------------------------- tabs
tab_data, tab_res, tab_tree, tab_pred = st.tabs(["📊 Data", "📈 Results", "🌲 Tree", "🔮 Predict"])

with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing values", int(df.isnull().sum().sum()))
    st.subheader("Top 5 rows")
    st.dataframe(df.head())
    a, b = st.columns(2)
    with a:
        st.subheader("Null values per column")
        st.dataframe(df.isnull().sum().rename("nulls"))
    with b:
        st.subheader(f"Class balance: `{target}`")
        st.bar_chart(df[target].value_counts())

res = st.session_state.get("res")

with tab_res:
    if not res:
        st.info("Set options in the sidebar and click **Train model**.")
    else:
        m, y_test, y_pred = res["model"], res["y_test"], res["y_pred"]
        enc = res["encoders"].get(res["target"])
        labels = list(enc.classes_) if enc is not None else sorted(set(y_test))
        acc = accuracy_score(y_test, y_pred)
        train_acc = accuracy_score(res["y_train"], m.predict(res["X_train"]))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test accuracy", f"{acc:.2%}")
        c2.metric("Train accuracy", f"{train_acc:.2%}")
        c3.metric("Tree depth", m.get_depth())
        c4.metric("Leaves", m.get_n_leaves())
        if train_acc - acc > 0.10:
            st.warning("Train accuracy is much higher than test accuracy - the tree may be "
                       "overfitting. Try a smaller max_depth or larger min_samples_leaf.")
        if res["best_params"]:
            st.success(f"Best parameters: {res['best_params']}")
        if res["filled"]:
            st.caption(f"Missing values were filled in: {', '.join(res['filled'])}")

        left, right = st.columns(2)
        with left:
            st.subheader("Confusion matrix")
            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay.from_predictions(
                y_test, y_pred, display_labels=labels, ax=ax, cmap="Blues")
            st.pyplot(fig)
        with right:
            st.subheader("Feature importance")
            imp = pd.Series(m.feature_importances_, index=res["X"].columns).sort_values()
            fig, ax = plt.subplots(figsize=(5, 4))
            imp.plot.barh(ax=ax, color="#2e7d32")
            ax.set_xlabel("Importance")
            st.pyplot(fig)

        st.subheader("Classification report")
        rep = classification_report(y_test, y_pred, target_names=[str(l) for l in labels],
                                    output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(3))

with tab_tree:
    if not res:
        st.info("Train a model to see the tree.")
    else:
        m = res["model"]
        enc = res["encoders"].get(res["target"])
        class_names = [str(c) for c in (enc.classes_ if enc is not None else m.classes_)]
        shown = st.slider("Levels to display", 1, max(1, m.get_depth()), min(3, max(1, m.get_depth())))
        fig, ax = plt.subplots(figsize=(max(12, 3 * shown + 6), 3 * shown + 2))
        plot_tree(m, feature_names=list(res["X"].columns), class_names=class_names,
                  filled=True, rounded=True, max_depth=shown, fontsize=8, ax=ax)
        st.pyplot(fig)

with tab_pred:
    if not res:
        st.info("Train a model first, then enter values here to get a prediction.")
    else:
        m, raw, encoders = res["model"], res["raw"], res["encoders"]
        st.subheader("Enter feature values")
        values = {}
        cols_ui = st.columns(3)
        for i, c in enumerate(res["X"].columns):
            with cols_ui[i % 3]:
                if c in encoders:
                    values[c] = st.selectbox(c, list(encoders[c].classes_), key=f"in_{c}")
                else:
                    s = raw[c].dropna()
                    lo, hi, med = float(s.min()), float(s.max()), float(s.median())
                    if pd.api.types.is_integer_dtype(s):
                        values[c] = st.number_input(c, int(lo), int(hi), int(med), step=1, key=f"in_{c}")
                    else:
                        values[c] = st.number_input(c, lo, hi, med, key=f"in_{c}")
        if st.button("Predict", type="primary"):
            row = pd.DataFrame([values])
            for c, le in encoders.items():
                if c in row:
                    row[c] = le.transform(row[c].astype(str))
            row = row[res["X"].columns]
            pred = m.predict(row)[0]
            tgt_enc = encoders.get(res["target"])
            label = tgt_enc.inverse_transform([pred])[0] if tgt_enc is not None else pred
            st.success(f"Predicted **{res['target']}**: {label}")
            names = tgt_enc.classes_ if tgt_enc is not None else m.classes_
            proba = pd.Series(m.predict_proba(row)[0], index=[str(n) for n in names], name="probability")
            st.bar_chart(proba)
