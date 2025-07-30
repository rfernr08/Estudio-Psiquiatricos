import polars as pl
import random
import numpy as np
import time
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

dataset =  "recursos/otros/BERT/diagnosticos_F20_F20.89_sin_dups_limpio.csv"
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


param_grid_gb = {
    'n_estimators': [100, 200, 300],           # Número de árboles
    'learning_rate': [0.01, 0.05, 0.1],        # Tasa de aprendizaje
    'max_depth': [3, 5, 7],                    # Profundidad máxima de los árboles
    'min_samples_split': [2, 5, 10],           # Mínimo de muestras para dividir un nodo
    'min_samples_leaf': [1, 2, 4],             # Mínimo de muestras en una hoja
    'subsample': [0.6, 0.8, 1.0],              # Proporción de muestras usadas en cada árbol
    'max_features': ['sqrt', 'log2', None]     # Número de features consideradas para el split
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

grid_xgb = GridSearchCV(GradientBoostingClassifier(random_state=seed),
                        param_grid=param_grid_gb,
                        scoring='f1',
                        cv=5,
                        n_jobs=-1,
                        verbose=1)

grid_xgb.fit(X_train, y_train)
print("Mejores parámetros XGB:", grid_xgb.best_params_)

"""
Fitting 5 folds for each of 2187 candidates, totalling 10935 fits
Mejores parámetros XGB: {'learning_rate': 0.05, 'max_depth': 7, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 100, 'subsample': 1.0}

Fitting 5 folds for each of 2187 candidates, totalling 10935 fits
Mejores parámetros XGB: {'learning_rate': 0.05, 'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200, 'subsample': 1.0}

Fitting 5 folds for each of 2187 candidates, totalling 10935 fits
Mejores parámetros XGB: {'learning_rate': 0.1, 'max_depth': 3, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 300, 'subsample': 1.0)
"""