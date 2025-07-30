import polars as pl
import random
import numpy as np
import time
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

dataset =  "recursos/otros/BERT/diagnosticos_F20_F20.89_con_descripcion_sin_dups_limpio.csv"
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
    pl.col("DIAG PSQ").str.contains("Otros tipos de esquizofrenia").cast(pl.Int32).alias("target")
).to_pandas()["target"]


param_grid_xgb = {
    'n_estimators': [100, 300],                  # Nº de árboles
    'learning_rate': [0.01, 0.05, 0.1],          # Tasa de aprendizaje
    'max_depth': [3, 6, 9],                      # Profundidad del árbol
    'subsample': [0.7, 0.8, 1.0],                # Porcentaje de muestras por árbol
    'colsample_bytree': [0.7, 0.8, 1.0],         # Porcentaje de columnas por árbol
    'min_child_weight': [1, 3, 5],               # Nº mínimo de instancias por hoja
    'gamma': [0, 0.2, 0.5],                      # Umbral mínimo de ganancia para split
    'scale_pos_weight': [1, 2]                   # Balanceo clases (importante para F20/F20.89)
}



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

grid_xgb = GridSearchCV(XGBClassifier(scale_pos_weight=2, eval_metric='logloss', verbosity=0),
                        param_grid=param_grid_xgb,
                        scoring='f1',
                        cv=5,
                        n_jobs=-1,
                        verbose=1)

grid_xgb.fit(X_train, y_train)
print("Mejores parámetros XGB:", grid_xgb.best_params_)

"""
Fitting 5 folds for each of 2916 candidates, totalling 14580 fits
Mejores parámetros XGB: {'colsample_bytree': 0.7, 'gamma': 0.2, 'learning_rate': 0.05, 'max_depth': 9, 'min_child_weight': 1, 'n_estimators': 300, 'scale_pos_weight': 2, 'subsample': 1.0}

Fitting 5 folds for each of 2916 candidates, totalling 14580 fits
Mejores parámetros XGB: {'colsample_bytree': 0.7, 'gamma': 0.2, 'learning_rate': 0.1, 'max_depth': 6, 'min_child_weight': 1, 'n_estimators': 100, 'scale_pos_weight': 2, 'subsample': 1.0}

Fitting 5 folds for each of 2916 candidates, totalling 14580 fits
Mejores parámetros XGB: {'colsample_bytree': 0.7, 'gamma': 0.5, 'learning_rate': 0.05, 'max_depth': 9, 'min_child_weight': 1, 'n_estimators': 300, 'scale_pos_weight': 2, 'subsample': 0.7}
"""