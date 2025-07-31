import polars as pl
import random
import numpy as np
import time
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

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


param_grid_rf = {
    'n_estimators': [50, 100, 200, 300],           # Número de árboles
    'max_depth': [10, 20, 30],               # Profundidad máxima de los árboles
    'min_samples_split': [2, 5, 10],               # Mínimo de muestras para dividir un nodo
    'min_samples_leaf': [1, 2, 4],                 # Mínimo de muestras en una hoja
    'max_features': ['sqrt', 'log2', None],        # Número de features consideradas para el split
    'bootstrap': [True, False],                     # Si usar bootstrap para construir los árboles
    'criterion': ['gini', 'entropy'],               # Criterio para medir la calidad de una división
    'class_weight': ['balanced', 'balanced_subsample']  # Pesos de las clases para balancear
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

grid_rf = GridSearchCV(RandomForestClassifier(random_state=seed),
                       param_grid=param_grid_rf,
                       scoring='f1',
                       cv=5,
                       n_jobs=-1,
                       verbose=1)

grid_rf.fit(X_train, y_train)
print("Mejores parámetros RandomForest:", grid_rf.best_params_)
