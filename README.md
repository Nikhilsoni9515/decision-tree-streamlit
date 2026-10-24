# 🌳 Decision Tree Classifier – Streamlit UI

Interactive UI for the decision tree classification workflow in `Decison_Tree_Sol.ipynb`
(`health_lifestyle_dataset.csv` → predict `disease_risk`).

## Features
- Upload any CSV (or drop `health_lifestyle_dataset.csv` next to `app.py` to load it automatically)
- Choose the target column, test size and random state
- Text columns (e.g. `gender`) are label-encoded automatically
- **Manual** training (criterion, max_depth, min_samples_split, min_samples_leaf) or **RandomizedSearchCV**
- Accuracy, confusion matrix, classification report, feature importance, tree visualisation
- Predict form that is generated from your dataset's columns

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy free
Push this folder to GitHub, then go to [share.streamlit.io](https://share.streamlit.io), pick the repo and set the main file to `app.py`.
