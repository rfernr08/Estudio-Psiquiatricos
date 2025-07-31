import shap
import pandas as pd
import polars as pl
import numpy as np
import time
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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

with open(r"code\verificar\params.json", "r") as f:
    model_params = json.load(f)

# Entrenamiento de modelos
rf_model = RandomForestClassifier(**model_params["RandomForestClassifier"])
xgb_model = XGBClassifier(**model_params["XGBClassifier"])
gb_model = GradientBoostingClassifier(**model_params["GradientBoostingClassifier"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# 🎯 Evaluación rápida
print("=== Random Forest ===")
print(classification_report(y_test, rf_model.predict(X_test)))
print("=== XGBoost ===")
print(classification_report(y_test, xgb_model.predict(X_test)))
print("=== Gradient Boosting ===")
print(classification_report(y_test, gb_model.predict(X_test)))

# === SHAP ===

# Usamos TreeExplainer para modelos de árbol
explainer_rf = shap.Explainer(rf_model, X_train)
explainer_xgb = shap.Explainer(xgb_model, X_train)
explainer_gb = shap.Explainer(gb_model, X_train)

shap_values_rf = explainer_rf(X_test)
shap_values_xgb = explainer_xgb(X_test)
shap_values_gb = explainer_gb(X_test)

# 💡 Visualizaciones

# === Random Forest ===
"""
print("\n🔎 Random Forest - Importancia de características (global):")
shap.plots.bar(shap_values_rf, max_display=10)
# === XGBoost ===
print("\n🔎 XGBoost - Importancia de características (global):")
shap.plots.bar(shap_values_xgb, max_display=10)

# === Gradient Boosting ===
print("\n🔎 Gradient Boosting - Importancia de características (global):")
shap.plots.bar(shap_values_gb, max_display=10)

# === Explicación individual para una fila ===
sample_idx = 0  # cambia esto para ver otras filas

print("\n🔍 SHAP - Waterfall plot fila individual (ej. XGBoost):")
shap.plots.waterfall(shap_values_xgb[sample_idx])
"""
print("\n🔎 Random Forest - Importancia de características (global):")
try:
    # Use summary_plot instead of bar plot for better compatibility
    shap.summary_plot(shap_values_rf, X_test, plot_type="bar", max_display=10, show=False)
    plt.title("Random Forest - Feature Importance")
    plt.savefig("feature_importance_rf_2.png")
    plt.show()
except Exception as e:
    print(f"Error plotting RF bar chart: {e}")
    # Alternative: print feature importance values
    feature_importance = np.abs(shap_values_rf.values).mean(0)
    feature_names = X_test.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    print(importance_df.head(10))



print("\n🔎 Random Forest - Importancia de características (global):")
try:
    # Use summary_plot instead of bar plot for better compatibility
    shap.summary_plot(shap_values_rf, X_test, plot_type="bar", max_display=10, show=False)
    plt.title("Random Forest - Feature Importance")
    plt.savefig("feature_importance_rf.png")
    plt.show()

except Exception as e:
    print(f"Error plotting RF bar chart: {e}")
    # Alternative: print feature importance values
    feature_importance = np.abs(shap_values_rf.values).mean(0)
    feature_names = X_test.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    print(importance_df.head(10))

# === XGBoost ===
print("\n🔎 XGBoost - Importancia de características (global):")
try:
    shap.summary_plot(shap_values_xgb, X_test, plot_type="bar", max_display=10, show=False)
    plt.title("XGBoost - Feature Importance")
    plt.savefig("feature_importance_xgb.png")
    plt.show()
    
except Exception as e:
    print(f"Error plotting XGB bar chart: {e}")
    feature_importance = np.abs(shap_values_xgb.values).mean(0)
    feature_names = X_test.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    print(importance_df.head(10))

# === Gradient Boosting ===
print("\n🔎 Gradient Boosting - Importancia de características (global):")
try:
    shap.summary_plot(shap_values_gb, X_test, plot_type="bar", max_display=10, show=False)
    plt.title("Gradient Boosting - Feature Importance")
    plt.savefig("feature_importance_gb.png")
    plt.show()
    
except Exception as e:
    print(f"Error plotting GB bar chart: {e}")
    feature_importance = np.abs(shap_values_gb.values).mean(0)
    feature_names = X_test.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    print(importance_df.head(10))

# === Explicación individual para una fila ===
sample_idx = 0  # cambia esto para ver otras filas

print("\n🔍 SHAP - Waterfall plot fila individual (ej. XGBoost):")
try:
    shap.plots.waterfall(shap_values_xgb[sample_idx], show=False)
    plt.savefig("waterfall_plot_xgb.png")
    plt.show()
    
except Exception as e:
    print(f"Error plotting waterfall: {e}")
    # Alternative: show individual SHAP values
    print(f"SHAP values for sample {sample_idx}:")
    shap_df = pd.DataFrame({
        'feature': X_test.columns,
        'shap_value': shap_values_xgb.values[sample_idx]
    }).sort_values('shap_value', key=abs, ascending=False)
    print(shap_df.head(10))
