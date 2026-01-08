# bert_tool_full.py
import os
import json
from typing import Union, List, Dict, Any
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from llama_index.core.tools import FunctionTool

# ---------------------------
# Config / carga del modelo
# ---------------------------
MODEL_PATH = os.getenv("BERT_MODEL_PATH", "path/to/your/bert-model")  # cambia si hace falta
ID2LABEL_PATH = os.getenv("ID2LABEL_PATH", "id2label.json")  # opcional: mapa index -> 'F20.0'

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carga tokenizer y modelo (cached)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# Intenta construir id2label (fallback)
if hasattr(model.config, "id2label") and model.config.id2label:
    id2label = {int(k): v for k, v in model.config.id2label.items()}
else:
    # si existe archivo JSON con mapping, cárgalo
    if os.path.exists(ID2LABEL_PATH):
        with open(ID2LABEL_PATH, "r", encoding="utf-8") as fh:
            id2label = json.load(fh)
            # asegurar keys ints
            id2label = {int(k): v for k, v in id2label.items()}
    else:
        # fallback: generamos labels numéricas
        n_labels = model.config.num_labels
        id2label = {i: f"label_{i}" for i in range(n_labels)}

# Detección del tipo de problema (multilabel o multiclass)
# heurística: si model.config.problem_type == "multi_label_classification" or loss_name contains 'BCE' -> multilabel
_problem_type = getattr(model.config, "problem_type", None)
if _problem_type == "multi_label_classification":
    is_multilabel = True
else:
    # fallback: si num_labels > 1 y no hay explicit problem type, asumimos multiclass
    is_multilabel = False

# ---------------------------
# Helper: formatear entrada
# ---------------------------
def prepare_text_input(historial: Union[str, List[str]]) -> str:
    """
    Acepta:
      - string con comas: "Gastritis, Menopausia, Dolor de cabeza, Esquizofrenia"
      - lista de strings: ["Gastritis", "Menopausia", ...]
    Devuelve texto concatenado para pasar al BERT.
    """
    if historial is None:
        return ""
    if isinstance(historial, list):
        parts = [p.strip() for p in historial if p and str(p).strip()]
        return " ; ".join(parts)
    # si es string
    text = str(historial).strip()
    # si contiene comas, separamos y normalizamos
    if "," in text:
        parts = [p.strip() for p in text.split(",") if p.strip()]
        return " ; ".join(parts)
    return text

# ---------------------------
# Función principal: bert_diagnose
# ---------------------------
def bert_diagnose(historial: Union[str, List[str]],
                  top_k: int = 5,
                  return_all: bool = False) -> Dict[str, Any]:
    """
    Devuelve predicciones ordenadas con probabilidades.
    Args:
        historial: str o list[str] con lista/descripcion de comorbilidades/síntomas/diagnósticos.
        top_k: cuántas clases devolver (ordenadas por probabilidad).
        return_all: si True, devuelve vector completo de probabilidades.
    Returns:
        {
            "predictions": [
                {"label": "F20.0", "name": "Esquizofrenia paranoide", "prob": 0.58, "idx": 3},
                ...
            ],
            "raw_probs": [ ... ]  # opcional, solo si return_all True
        }
    """
    text = prepare_text_input(historial)
    if not text:
        return {"error": "No hay texto clínico proporcionado."}

    # Tokenize (truncation para evitar overflow)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    # mover tensores al device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # shape (1, n_labels) ó (batch, n_labels)
        logits = logits.cpu().numpy()[0]  # tomar el primer item si batch

    # Convertir logits a probabilidades según tipo de problema
    if is_multilabel:
        # sigmoid por etiqueta
        probs = 1.0 / (1.0 + np.exp(-logits))
        # normalizamos? en multilabel no hace falta normalizar (cada etiqueta independiente)
        probs_norm = probs  # mantener independientes
    else:
        # multiclass -> softmax
        exps = np.exp(logits - np.max(logits))
        probs_norm = exps / exps.sum()

    # Construir lista de (idx, prob)
    idxs = np.argsort(probs_norm)[::-1]
    preds = []
    for i in idxs[:top_k]:
        label = id2label.get(int(i), f"label_{i}")
        preds.append({
            "idx": int(i),
            "label": label,
            "prob": float(probs_norm[i])
        })

    result = {"predictions": preds}
    if return_all:
        result["raw_probs"] = [float(x) for x in probs_norm.tolist()]
    return result

# ---------------------------
# Ejemplo de uso directo
# ---------------------------
if __name__ == "__main__":
    ejemplo = "Gastronterintis, Menopausia, Dolor de cabeza, Esquizofrenia"
    out = bert_diagnose(ejemplo, top_k=6, return_all=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))

# ---------------------------
# Integración con LlamaIndex como FunctionTool
# ---------------------------
def bert_tool_func(historial_text: str) -> dict:
    """
    Wrapper que será llamado por LlamaIndex/Cohere (via tools).
    Acepta un string (historial) y devuelve el resultado serializable.
    """
    try:
        return bert_diagnose(historial_text, top_k=6, return_all=False)
    except Exception as e:
        return {"error": str(e)}

bert_tool = FunctionTool.from_defaults(
    name="bert_classifier",
    description="Clasifica un historial clínico (lista de diagnósticos/síntomas) y devuelve probabilidades por ICD.",
    func=bert_tool_func
)
