import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.inspection import permutation_importance

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

################################### TSCC Dataset
ls_vals = pd.read_csv("tscc_layerwise_synch.csv")
layer_cols = [str(i) for i in range(13)]
ls_vals[layer_cols] = 1 - ls_vals[layer_cols]
ls_vals = ls_vals.groupby("file")[layer_cols].mean().reset_index()

features = pd.read_csv("tscc_spss_factors.csv")
outcomes = features[['cefr', 'file']]
features = features.drop(columns=['cefr'])
feature_columns = features.columns.tolist()
feature_columns = feature_columns[13:]

ls_vals["row_idx"]  = ls_vals.groupby("file").cumcount()
features["row_idx"] = features.groupby("file").cumcount()

merged = ls_vals.merge(features, on="file", how="inner")
merged = merged.drop(columns=["row_idx_y"])
outcomes = outcomes[outcomes['file'].isin(merged['file'])]

pc_cols = ['syntactic', 'semantic', 'lexical'] 
layer_cols = [str(i) for i in range(13)]  

data = merged[['file'] + pc_cols].merge(outcomes[['file','cefr']], on='file', how='inner')
data = data.replace(r"^\s*$", np.nan, regex=True).dropna()

layer_cols = [str(i) for i in range(13)]

temp = data.merge(ls_vals[['file'] + layer_cols], on='file', how='inner')

# Setup Data
X = temp[layer_cols]
y = temp['cefr']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Importance functions
def perm_importance(model, X_test, y_test): # measures drop in f1 score when shuffling x of specific layer
    perm = permutation_importance(
        model, X_test, y_test,
        n_repeats=20, random_state=42, scoring="accuracy"
    )
    return perm.importances_mean

def weight_importance(model, X_test, y_test): #aggregate multiclass coefficient to one
    last = list(model.named_steps)[-1]
    W = model.named_steps[last].coef_

    if W.ndim == 1:
        return np.abs(W) # getting absolute weights
    return np.linalg.norm(W, axis=0)


# 5 fold CV 
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
def cv_evaluate(model, X_train, y_train, imp_func=None, param_grid=None, inner_cv=5):
    f1, acc, imp = [], [], []
    best_params = None

    # Hyperparameter tuning
    if param_grid is not None:
        gs = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="accuracy",
            cv=inner_cv,
            n_jobs=-1
        )
        gs.fit(X_train, y_train)
        best_model = gs.best_estimator_
        best_params = gs.best_params_
    else:
        best_model = model.fit(X_train, y_train)

    # Evaluation
    y_pred = best_model.predict(X_test)
    acc.append(accuracy_score(y_test, y_pred))
    f1.append(f1_score(y_test, y_pred, average="macro"))

    if imp_func is not None:
        imp.append(imp_func(best_model, X_test, y_test))

    return {
        "accuracy": np.mean(acc),
        "macro_f1": np.mean(f1),
        "importance": np.mean(imp, axis=0) if imp else None,
        "best_params": best_params
    }

# Output format-- ranking the layer importance
def make_importance_df(model_name, importances):
    df = pd.DataFrame({
        "model": model_name,
        "layer": list(X.columns),
        "importance": importances
    }).sort_values("layer").reset_index(drop=True)

    rank = df.sort_values("importance", ascending=False).reset_index(drop=True)
    rank["rank"] = np.arange(1, len(rank) + 1)
    rank["model"] = model_name
    return df, rank[["model", "rank", "layer", "importance"]]

# Model Definition
models = [
    ("RF",
     RandomForestClassifier(random_state=42),
     perm_importance,
     {
         "n_estimators": [200, 300, 400, 500],
         "max_depth": [None, 10, 30],
         "min_samples_split": [2, 5, 10],
        #  "min_samples_leaf": [1, 2, 4],
        #  "max_features": ["sqrt", "log2"]
     }),

    ("GBM",
     GradientBoostingClassifier(random_state=42),
     perm_importance,
     {
         "n_estimators": [100, 300, 500],
         "learning_rate": [0.01, 0.05, 0.1],
         "max_depth": [2, 3, 4],
        #  "subsample": [0.8, 1.0]
     }),

    ("kNN",
     Pipeline([("scaler", StandardScaler()),
               ("knn", KNeighborsClassifier())]),
     perm_importance,
     {
         "knn__n_neighbors": [3, 5, 7, 11, 15],
         "knn__weights": ["uniform", "distance"],
         "knn__p": [1, 2]  # Manhattan vs Euclidean
     }),

    ("SVM_weights",
     Pipeline([("scaler", StandardScaler()),
               ("svc", LinearSVC(max_iter=20000))]),
     perm_importance,
     {
         "svc__C": [0.01, 0.1, 1, 10]
     }),

    ("LogReg_weights",
     Pipeline([("scaler", StandardScaler()),
               ("logreg", LogisticRegression(
                   penalty="l2", solver="lbfgs", max_iter=5000
               ))]),
     perm_importance,
     {
         "logreg__C": [0.01, 0.1, 1, 10]
     }),
]


all_rankings = []
metrics = []
best_params_by_model = {}

for name, model, imp_func, grid in models:
    out = cv_evaluate(model, X_train, y_train, imp_func=imp_func, param_grid=grid, inner_cv=5)

    metrics.append({
        "model": name,
        "accuracy": out["accuracy"],
        "macro_f1": out["macro_f1"]
    })

    best_params_by_model[name] = out.get("best_params", None)

    _, rank_df = make_importance_df(name, out["importance"])
    all_rankings.append(rank_df)

# Results
rankings_df = pd.concat(all_rankings, ignore_index=True)
metrics_df = pd.DataFrame(metrics).sort_values("macro_f1", ascending=False)

print("\nModel performance (5-fold CV mean):")
print(metrics_df)

# print("\nLayer Rankings (CV-mean importance):")
# print(rankings_df.sort_values(["model", "rank"]))

print("\nBest hyperparameters:")
for name, params in best_params_by_model.items():
    if params is None:
        print(f"{name}: (no tuning)")
    else:
        print(f"{name}: {params}")

# formatted cell: "layer (importance)"
rankings_df["cell"] = rankings_df.apply(
    lambda r: f"{r['layer']} ({r['importance']:.3f})",
    axis=1
)

# pivot: rows = rank, columns = model
rank_table = rankings_df.pivot(
    index="rank",
    columns="model",
    values="cell"
).sort_index()

print("\nRank table:")
print(rank_table)

