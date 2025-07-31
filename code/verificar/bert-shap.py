import shap
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TextClassificationPipeline
import pandas as pd

# Carga modelo BERT ya entrenado
model_name = "bert-base-multilingual-cased"  # o tu modelo fine-tuned guardado en disco
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Crear pipeline de predicción
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True, device=0 if torch.cuda.is_available() else -1)

# Selecciona textos de entrada
# Si ya tienes X_test como pandas Series o lista de textos:
ejemplos = X_test[:10].tolist()  # Puedes poner más

# Usa SHAP explainer para NLP
explainer = shap.Explainer(pipe)

# Explicación SHAP de predicciones
shap_values = explainer(ejemplos)

# Visualizar uno de los textos explicados
shap.plots.text(shap_values[0])  # Puedes cambiar el índice para ver otros
