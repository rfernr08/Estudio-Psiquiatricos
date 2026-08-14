import os
import time
import random
import numpy as np
import pandas as pd
import polars as pl

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.ensemble import GradientBoostingClassifier

# 1. Carga de datos
dataset_path = r"C:\Users\Usuario\Documents\Workspace\Mirage\TFG\dataset\dataset_finales\Desbalance\dataset_descripciones.csv"
df = pl.read_csv(dataset_path, separator="|")

seed = int(time.time_ns() % (2**32))
np.random.seed(seed)

# 2. Filtrar por códigos objetivo (Multiclase: F20, F21, etc.)
codes = sorted(["F20", "F21", "F22", "F23", "F25", "F29", "F60.1"])
df = df.filter(pl.col("DIAG PSQ").is_in(codes))

# 3. Separar Características (X) y Target (y)
diag_colms = [col for col in df.columns if col.startswith("Diag")]
df_diag = df.select(diag_colms).fill_null("")

# Convertir X usando OrdinalEncoder (más limpio y estándar para features que LabelEncoder)
X_pandas = df_diag.to_pandas()
encoder_X = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X = encoder_X.fit_transform(X_pandas)

# Convertir y a enteros (0 a 6)
le_y = LabelEncoder()
y = le_y.fit_transform(df.select("DIAG PSQ").to_series().to_numpy())

# 4. División Train / Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=seed, stratify=y
)

# 5. Definir la Malla de Búsqueda
param_grid_gb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.6, 0.8, 1.0],
    'max_features': ['sqrt', 'log2', None]
}

# 6. Búsqueda Aleatoria Inteligente (RandomizedSearchCV)
print("🚀 Iniciando búsqueda de hiperparámetros...")
search_gb = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(random_state=seed),
    param_distributions=param_grid_gb,
    n_iter=25,               # Prueba 25 combinaciones aleatorias en lugar de 2.187
    scoring='f1_weighted',    # Necesario para evaluación multiclase
    cv=5,
    n_jobs=-1,               # Usa todos los núcleos del procesador
    random_state=seed,
    verbose=1
)

search_gb.fit(X_train, y_train)

# 7. Resultados y Evaluación
print("\n🏆 Mejores parámetros encontrados:")
print(search_gb.best_params_)

best_model = search_gb.best_estimator_
y_pred = best_model.predict(X_test)

print("\n📊 Reporte de Clasificación en Test:")
print(classification_report(y_test, y_pred, target_names=le_y.classes_))