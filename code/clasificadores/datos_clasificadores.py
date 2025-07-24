import polars as pl
import random
import numpy as np
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

dataset =  "recursos/otros/diagnosticos_F20_F20.89_con_descripcion.csv"
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)


# Supón que estos son los mejores parámetros encontrados
best_model_xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=9,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=1,
    gamma=0.5,
    scale_pos_weight=2,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=seed
)

# Entrenar
best_model_xgb.fit(X_train, y_train)

# Predecir y calcular métricas
y_scores = best_model_xgb.predict_proba(X_test)[:, 1]
y_pred_xgb = best_model_xgb.predict(X_test)

# Calcular métricas XGBoost
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb, average='binary', zero_division=0)
recall_xgb = recall_score(y_test, y_pred_xgb, average='binary', zero_division=0)
f1_xgb = f1_score(y_test, y_pred_xgb, average='binary', zero_division=0)

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = roc_auc_score(y_test, y_scores)

print("=== XGBoost Classifier Results ===")
print(f"Accuracy: {accuracy_xgb:.4f}")
print(f"Precision: {precision_xgb:.4f}")
print(f"Recall: {recall_xgb:.4f}")
print(f"F1-Score: {f1_xgb:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_xgb))

# Plot ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Tasa de falsos positivos (FPR)')
plt.ylabel('Tasa de verdaderos positivos (TPR)')
plt.title('Curva ROC - XGBoost')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/curva_ROC_XGB.png")
plt.show(block=False)

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


best_model_gb = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=5,
    min_samples_leaf=1,
    subsample=1.0,
    max_features=None,
    random_state=seed
)

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

print("\n=== Comparación de Modelos ===")
comparison_data = {
    'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'],
    'XGBoost': [accuracy_xgb, precision_xgb, recall_xgb, f1_xgb, roc_auc],
    'Gradient Boosting': [accuracy_gb, precision_gb, recall_gb, f1_gb, roc_auc_gb]
}

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.round(4))

# Gráfico de comparación
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
xgb_scores = [accuracy_xgb, precision_xgb, recall_xgb, f1_xgb, roc_auc]
gb_scores = [accuracy_gb, precision_gb, recall_gb, f1_gb, roc_auc_gb]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(x - width/2, xgb_scores, width, label='XGBoost', color='orange', alpha=0.8)
plt.bar(x + width/2, gb_scores, width, label='Gradient Boosting', color='green', alpha=0.8)

plt.xlabel('Métricas')
plt.ylabel('Puntuación')
plt.title('Comparación de Modelos: XGBoost vs Gradient Boosting')
plt.xticks(x, metrics)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/comparacion_modelos.png")
plt.show(block=False)

# Guardar resultados en CSV
comparison_df.to_csv("results/comparacion_metricas_modelos.csv", index=False, sep="|")

