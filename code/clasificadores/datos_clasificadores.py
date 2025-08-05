import polars as pl
import random
import numpy as np
import time
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

dataset =  "TFG/dataset/diagnosticos_binarios_combinados.csv"
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

# Cargar parámetros desde JSON
with open(r"TFG/dataset/best_params.json", "r") as f:
    all_params = json.load(f)

# Seleccionar el dataset que estás usando
dataset_type = "combinado"  # Cambia a "codigos", "descripciones" o "combinado" según tu dataset
model_params = all_params[dataset_type]

print(f"Usando parámetros para dataset: {dataset_type}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

# === XGBoost Classifier ===
best_model_xgb = XGBClassifier(**model_params["XGBClassifier"])
best_model_xgb.set_params(random_state=seed)  # Asegurar que el seed se use

best_model_xgb.fit(X_train, y_train)
y_scores_xgb = best_model_xgb.predict_proba(X_test)[:, 1]
y_pred_xgb = best_model_xgb.predict(X_test)

# Calcular métricas XGBoost
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb, average='binary', zero_division=0)
recall_xgb = recall_score(y_test, y_pred_xgb, average='binary', zero_division=0)
f1_xgb = f1_score(y_test, y_pred_xgb, average='binary', zero_division=0)
fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_test, y_scores_xgb)
roc_auc_xgb = roc_auc_score(y_test, y_scores_xgb)

print("=== XGBoost Classifier Results ===")
print(f"Accuracy: {accuracy_xgb:.4f}")
print(f"Precision: {precision_xgb:.4f}")
print(f"Recall: {recall_xgb:.4f}")
print(f"F1-Score: {f1_xgb:.4f}")
print(f"ROC AUC: {roc_auc_xgb:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_xgb))

# === Gradient Boosting Classifier ===
best_model_gb = GradientBoostingClassifier(**model_params["GradientBoostingClassifier"])
best_model_gb.set_params(random_state=seed)  # Asegurar que el seed se use

best_model_gb.fit(X_train, y_train)
y_scores_gb = best_model_gb.predict_proba(X_test)[:, 1]
y_pred_gb = best_model_gb.predict(X_test)

# Calcular métricas Gradient Boosting
accuracy_gb = accuracy_score(y_test, y_pred_gb)
precision_gb = precision_score(y_test, y_pred_gb, average='binary', zero_division=0)
recall_gb = recall_score(y_test, y_pred_gb, average='binary', zero_division=0)
f1_gb = f1_score(y_test, y_pred_gb, average='binary', zero_division=0)
fpr_gb, tpr_gb, thresholds_gb = roc_curve(y_test, y_scores_gb)
roc_auc_gb = roc_auc_score(y_test, y_scores_gb)

print("\n=== Gradient Boosting Classifier Results ===")
print(f"Accuracy: {accuracy_gb:.4f}")
print(f"Precision: {precision_gb:.4f}")
print(f"Recall: {recall_gb:.4f}")
print(f"F1-Score: {f1_gb:.4f}")
print(f"ROC AUC: {roc_auc_gb:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_gb))

# === Random Forest Classifier ===
best_model_rf = RandomForestClassifier(**model_params["RandomForestClassifier"])
best_model_rf.set_params(random_state=seed)  # Asegurar que el seed se use

best_model_rf.fit(X_train, y_train)
y_scores_rf = best_model_rf.predict_proba(X_test)[:, 1]
y_pred_rf = best_model_rf.predict(X_test)

# Calcular métricas Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, average='binary', zero_division=0)
recall_rf = recall_score(y_test, y_pred_rf, average='binary', zero_division=0)
f1_rf = f1_score(y_test, y_pred_rf, average='binary', zero_division=0)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_scores_rf)
roc_auc_rf = roc_auc_score(y_test, y_scores_rf)

print("\n=== Random Forest Classifier Results ===")
print(f"Accuracy: {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall: {recall_rf:.4f}")
print(f"F1-Score: {f1_rf:.4f}")
print(f"ROC AUC: {roc_auc_rf:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_rf))

# === Gráficas individuales ===

# ROC XGBoost
plt.figure(figsize=(8, 6))
plt.plot(fpr_xgb, tpr_xgb, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_xgb:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Tasa de falsos positivos (FPR)')
plt.ylabel('Tasa de verdaderos positivos (TPR)')
plt.title('Curva ROC - XGBoost')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/curva_ROC_XGB.png")
plt.show(block=False)

# ROC Gradient Boosting
plt.figure(figsize=(8, 6))
plt.plot(fpr_gb, tpr_gb, color='green', lw=2, label=f'ROC curve GB (AUC = {roc_auc_gb:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Tasa de falsos positivos (FPR)')
plt.ylabel('Tasa de verdaderos positivos (TPR)')
plt.title('Curva ROC - Gradient Boosting')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/curva_ROC_GB.png")
plt.show(block=False)

# ROC Random Forest
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, color='red', lw=2, label=f'ROC curve RF (AUC = {roc_auc_rf:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Tasa de falsos positivos (FPR)')
plt.ylabel('Tasa de verdaderos positivos (TPR)')
plt.title('Curva ROC - Random Forest')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/curva_ROC_RF.png")
plt.show(block=False)

# ROC Comparativa
plt.figure(figsize=(10, 8))
plt.plot(fpr_xgb, tpr_xgb, color='darkorange', lw=2, label=f'XGBoost (AUC = {roc_auc_xgb:.2f})')
plt.plot(fpr_gb, tpr_gb, color='green', lw=2, label=f'Gradient Boosting (AUC = {roc_auc_gb:.2f})')
plt.plot(fpr_rf, tpr_rf, color='red', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Tasa de falsos positivos (FPR)')
plt.ylabel('Tasa de verdaderos positivos (TPR)')
plt.title('Comparación de Curvas ROC')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/curva_ROC_comparativa.png")
plt.show(block=False)

# === Matrices de confusión ===

# Matriz de confusión XGBoost
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No F20.89', 'F20.89'], 
            yticklabels=['No F20.89', 'F20.89'])
plt.title('Matriz de Confusión - XGBoost')
plt.ylabel('Verdadero')
plt.xlabel('Predicho')
plt.tight_layout()
plt.savefig("results/confusion_matrix_XGB.png")
plt.show(block=False)

# Matriz de confusión Gradient Boosting
cm_gb = confusion_matrix(y_test, y_pred_gb)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['No F20.89', 'F20.89'], 
            yticklabels=['No F20.89', 'F20.89'])
plt.title('Matriz de Confusión - Gradient Boosting')
plt.ylabel('Verdadero')
plt.xlabel('Predicho')
plt.tight_layout()
plt.savefig("results/confusion_matrix_GB.png")
plt.show(block=False)

# Matriz de confusión Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Reds', 
            xticklabels=['No F20.89', 'F20.89'], 
            yticklabels=['No F20.89', 'F20.89'])
plt.title('Matriz de Confusión - Random Forest')
plt.ylabel('Verdadero')
plt.xlabel('Predicho')
plt.tight_layout()
plt.savefig("results/confusion_matrix_RF.png")
plt.show(block=False)

# === Comparación de Modelos ===
print("\n=== Comparación de Modelos ===")
comparison_data = {
    'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'],
    'XGBoost': [accuracy_xgb, precision_xgb, recall_xgb, f1_xgb, roc_auc_xgb],
    'Gradient Boosting': [accuracy_gb, precision_gb, recall_gb, f1_gb, roc_auc_gb],
    'Random Forest': [accuracy_rf, precision_rf, recall_rf, f1_rf, roc_auc_rf]
}

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.round(4))

# Gráfico de comparación
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
xgb_scores = [accuracy_xgb, precision_xgb, recall_xgb, f1_xgb, roc_auc_xgb]
gb_scores = [accuracy_gb, precision_gb, recall_gb, f1_gb, roc_auc_gb]
rf_scores = [accuracy_rf, precision_rf, recall_rf, f1_rf, roc_auc_rf]

x = np.arange(len(metrics))
width = 0.25

plt.figure(figsize=(14, 8))
plt.bar(x - width, xgb_scores, width, label='XGBoost', color='orange', alpha=0.8)
plt.bar(x, gb_scores, width, label='Gradient Boosting', color='green', alpha=0.8)
plt.bar(x + width, rf_scores, width, label='Random Forest', color='red', alpha=0.8)

plt.xlabel('Métricas')
plt.ylabel('Puntuación')
plt.title('Comparación de Modelos: XGBoost vs Gradient Boosting vs Random Forest')
plt.xticks(x, metrics)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/comparacion_modelos.png")
plt.show(block=False)

comparison_df.to_csv("results/comparacion_metricas_modelos.csv", index=False, sep="|")

print("\n=== Resumen Final ===")
# Obtener los nombres de las columnas de modelos (excluyendo 'Métrica')
model_columns = comparison_df.columns[1:].tolist()

# Encontrar el índice del mejor modelo para cada métrica
best_accuracy_idx = comparison_df.iloc[0, 1:].idxmax()
best_f1_idx = comparison_df.iloc[3, 1:].idxmax()
best_auc_idx = comparison_df.iloc[4, 1:].idxmax()

print(f"Mejor modelo por Accuracy: {best_accuracy_idx}")
print(f"Mejor modelo por F1-Score: {best_f1_idx}")
print(f"Mejor modelo por ROC AUC: {best_auc_idx}")

# Mostrar valores específicos
print(f"\nValores específicos:")
print(f"Mejor Accuracy: {comparison_df.iloc[0, 1:].max():.4f} ({best_accuracy_idx})")
print(f"Mejor F1-Score: {comparison_df.iloc[3, 1:].max():.4f} ({best_f1_idx})")
print(f"Mejor ROC AUC: {comparison_df.iloc[4, 1:].max():.4f} ({best_auc_idx})")

print(f"\nParámetros utilizados del dataset '{dataset_type}':")
for model_name, params in model_params.items():
    print(f"\n{model_name}:")
    for param, value in params.items():
        print(f"  {param}: {value}")