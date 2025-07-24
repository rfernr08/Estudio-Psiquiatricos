import polars as pl
import random
import numpy as np
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def random_tree(X, y, n_estimators=100):
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    clf.fit(X, y)
    return clf

def decision_tree(X, y, max_depth=None):
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    clf.fit(X, y)
    return clf

def xgboost_tree(X, y, n_estimators=100):
    clf = XGBClassifier(n_estimators=n_estimators, random_state=seed)
    clf.fit(X, y)
    return clf

def lightgbm_tree(X, y, n_estimators=100):
    clf = LGBMClassifier(
        n_estimators=n_estimators, 
        class_weight='balanced', 
        random_state=seed,
        verbose=-1  # Suppress warnings
    )
    clf.fit(X_train, y_train)
    return clf

def catboost_tree(X, y, n_estimators=100):
    clf = CatBoostClassifier(iterations=n_estimators, verbose=0)
    clf.fit(X, y)
    return clf

def gradient_boosting_tree(X, y, n_estimators=100):
    clf = GradientBoostingClassifier(n_estimators=n_estimators, random_state=seed)
    clf.fit(X, y)
    return clf

dataset =  "recursos/otros/diagnosticos_F20_F20.89.csv"
df = pl.read_csv(dataset, separator="|")

seed = int(time.time_ns() % (2**32))
np.random.seed(seed)
diag_colms = [col for col in df.columns if col.startswith("Diag")]

df_diag = df.select(diag_colms).fill_nan("").fill_null("")
X = df_diag.to_pandas()

# Encode categorical features
le = LabelEncoder()
for col in X.columns:
    X[col] = le.fit_transform(X[col].astype(str))

y = df.select(
    pl.col("DIAG PSQ").str.contains("F20.89").cast(pl.Int32).alias("target")
).to_pandas()["target"]

print(f"First 5 rows of X:\n{X.head()}")
print(f"First 5 rows of y:\n{y.head()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

n_estimators = 100

print("=== Random Forest Classifier ===")

clf = random_tree(X_train, y_train, n_estimators=n_estimators)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

print("=== Decision Tree Classifier ===")

clf = decision_tree(X_train, y_train, max_depth=5)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

print("=== XGBoost Classifier ===")

clf = xgboost_tree(X_train, y_train, n_estimators=n_estimators)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

print("=== CatBoost Classifier ===")

clf = catboost_tree(X_train, y_train, n_estimators=n_estimators)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

print("=== LightGBM Classifier ===")

clf = lightgbm_tree(X_train, y_train, n_estimators=n_estimators)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

print("=== Gradient Boosting Classifier ===")

clf = gradient_boosting_tree(X_train, y_train, n_estimators=n_estimators)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))



