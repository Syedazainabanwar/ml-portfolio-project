import glob
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")


def find_csv(word):
    for folder in ["~/ml-portfolio-project", "~/ml_project", "~/Downloads", "~/Desktop", "~/Documents"]:
        hits = sorted(glob.glob(os.path.expanduser(folder) + f"/*{word}*.csv"))
        if hits:
            print("Found", word, "file:", hits[0])
            return hits[0]
    raise SystemExit(f"Could not find a CSV with '{word}' in its name. Put it in the project folder.")


# ------------------------------------------------------------------ regression
print("\n=== REGRESSION ===")
df = pd.read_csv(find_csv("regression")).drop_duplicates()
y = df["campaign_revenue_usd"]
X = df.drop(columns=["campaign_revenue_usd"])
num_all = X.select_dtypes(include="number").columns.tolist()
cat_all = X.select_dtypes(exclude="number").columns.tolist()


def reg_pre(columns):
    return ColumnTransformer([
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("e", OneHotEncoder(handle_unknown="ignore"))]),
         [c for c in columns if c in cat_all]),
        ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]),
         [c for c in columns if c in num_all]),
    ])


X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=42)
probe = Pipeline([("pre", reg_pre(X.columns)), ("poly", PolynomialFeatures(2, include_bias=False)),
                  ("reg", LinearRegression())]).fit(X_tr, y_tr)
imp = pd.Series(permutation_importance(probe, X_te, y_te, n_repeats=10, random_state=42,
                                       scoring="r2").importances_mean, index=X.columns)
selected = imp[imp > 0.002].sort_values(ascending=False).index.tolist()
print("Selected features:", selected)

X_tr, X_te = X_tr[selected], X_te[selected]
kf = KFold(5, shuffle=True, random_state=42)
candidates = {
    "Linear Regression": (Pipeline([("pre", reg_pre(selected)), ("m", LinearRegression())]), {}),
    "Polynomial Regression": (Pipeline([("pre", reg_pre(selected)), ("poly", PolynomialFeatures(include_bias=False)),
                                        ("m", LinearRegression())]), {"poly__degree": [1, 2, 3]}),
    "Random Forest": (Pipeline([("pre", reg_pre(selected)), ("m", RandomForestRegressor(random_state=42))]),
                      {"m__n_estimators": [100, 200], "m__min_samples_leaf": [1, 3]}),
    "Gradient Boosting": (Pipeline([("pre", reg_pre(selected)), ("m", GradientBoostingRegressor(random_state=42))]),
                          {"m__n_estimators": [100, 200], "m__learning_rate": [0.05, 0.1], "m__max_depth": [2, 3]}),
}
best_name, best_model, best_cv = None, None, np.inf
for name, (pipe, grid) in candidates.items():
    gs = GridSearchCV(pipe, grid or {"m": [pipe.named_steps["m"]]}, scoring="neg_mean_absolute_error",
                      cv=kf, n_jobs=-1).fit(X_tr, y_tr)
    print(f"{name}: CV MAE = {-gs.best_score_:.2f}")
    if -gs.best_score_ < best_cv:
        best_name, best_model, best_cv = name, gs.best_estimator_, -gs.best_score_
pred = best_model.predict(X_te)
print(f"Selected: {best_name} | test MAE {mean_absolute_error(y_te, pred):.2f} | test R2 {r2_score(y_te, pred):.4f}")
joblib.dump(best_model, "revenue_model.joblib")
joblib.dump(selected, "revenue_features.joblib")

# -------------------------------------------------------------- classification
print("\n=== CLASSIFICATION ===")
df = pd.read_csv(find_csv("classification")).drop_duplicates()
y = df["return_risk_label"]
X = df.drop(columns=["return_risk_label"])
cont = ["cart_value_usd", "past_return_rate", "delivery_days_taken", "discount_pct_applied"]
binary = ["size_bracketing_flag", "newsletter_subscriber"]
cats = X.select_dtypes(exclude="number").columns.tolist()


def cls_pre(columns):
    return ColumnTransformer([
        ("c", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]),
         [c for c in cont if c in columns]),
        ("b", SimpleImputer(strategy="most_frequent"), [c for c in binary if c in columns]),
        ("k", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                        ("e", OneHotEncoder(handle_unknown="ignore"))]), [c for c in cats if c in columns]),
    ])


X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
probe = Pipeline([("pre", cls_pre(X.columns)),
                  ("m", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))]).fit(X_tr, y_tr)
imp = pd.Series(permutation_importance(probe, X_te, y_te, n_repeats=10, random_state=42,
                                       scoring="roc_auc").importances_mean, index=X.columns)
selected = imp[imp > 0.005].sort_values(ascending=False).index.tolist()
print("Selected features:", selected)

X_tr, X_te = X_tr[selected], X_te[selected]
skf = StratifiedKFold(5, shuffle=True, random_state=42)
candidates = {
    "Logistic Regression": (LogisticRegression(max_iter=1000), {"m__C": [0.1, 1, 10, 100, 1000]}),
    "Random Forest": (RandomForestClassifier(random_state=42, n_jobs=-1),
                      {"m__n_estimators": [100, 300], "m__min_samples_leaf": [1, 5]}),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=42),
                          {"m__n_estimators": [100, 200], "m__learning_rate": [0.05, 0.1], "m__max_depth": [2, 3]}),
}
best_name, best_model, best_cv = None, None, -1
for name, (model, grid) in candidates.items():
    gs = GridSearchCV(Pipeline([("pre", cls_pre(selected)), ("m", model)]), grid, scoring="roc_auc",
                      cv=skf, n_jobs=-1).fit(X_tr, y_tr)
    print(f"{name}: CV ROC-AUC = {gs.best_score_:.4f}")
    if gs.best_score_ > best_cv:
        best_name, best_model, best_cv = name, gs.best_estimator_, gs.best_score_
pred = best_model.predict(X_te)
proba = best_model.predict_proba(X_te)[:, 1]
print(f"Selected: {best_name} | accuracy {accuracy_score(y_te, pred):.3f} | recall {recall_score(y_te, pred):.3f} "
      f"| F1 {f1_score(y_te, pred):.3f} | ROC-AUC {roc_auc_score(y_te, proba):.3f}")
joblib.dump(best_model, "return_model.joblib")
joblib.dump(selected, "return_features.joblib")

print("\nDONE. Created 4 files: revenue_model.joblib, revenue_features.joblib, return_model.joblib, return_features.joblib")
