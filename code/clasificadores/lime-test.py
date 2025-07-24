import shap
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from lime.lime_text import LimeTextExplainer
# ✅ Carga tu modelo ya entrenado
model_name = "dccuchile/bert-base-spanish-wwm-cased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 🔒 Coloca en modo evaluación
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

# 🧠 Función de predicción - Fixed to handle SHAP input
def predict(texts):
    # Handle different input types from SHAP
    if isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, np.ndarray):
        texts = texts.tolist()
    elif not isinstance(texts, list):
        texts = [str(texts)]
    
    # Convert to strings if needed
    texts = [str(text) if not isinstance(text, str) else text for text in texts]
    
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs.cpu().numpy()

# 🔍 Use a simpler explainer for text


# Create LIME explainer
explainer = LimeTextExplainer(class_names=['Otros tipos', 'Esquizofrenia'])

# Define the text to explain
texto = "Este es un ejemplo de texto para explicar con LIME."

# Get explanation
explanation = explainer.explain_instance(texto, predict, num_features=10)
explanation.show_in_notebook(text=True)