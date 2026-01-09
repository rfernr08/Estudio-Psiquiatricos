import os
from llama_index.llms.cohere import Cohere
from llama_index.core import Settings
from llama_index.core import KnowledgeGraphIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core.llms import ChatMessage
from neo4j import GraphDatabase
from llama_index.core.tools import FunctionTool
from dotenv import load_dotenv
from streamlit import text
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import re
import matplotlib.pyplot as plt

load_dotenv()
torch.set_default_device("cpu")
system_prompt = PromptTemplate(
"""
Eres un psiquiatra clínico profesional.
Reglas:

1. Nunca realices un diagnóstico sin antes llamar a la herramienta BERT.
2. Si falta información para diagnosticar, solicita más datos antes.
3. Cuando tengas un diagnóstico ICD:
    - Consulta Neo4J para obtener evidencia real del dataset.
4. Nunca inventes diagnósticos ni medicación.
5. Responde siempre con el siguiente formato:
6. Si el usuario describe síntomas, siempre debes llamar primero a la herramienta BERT para obtener el ICD.

Formato de respuesta:
- Resumen del caso
- Diagnóstico preliminar
- Evidencia clínica encontrada
- Información faltante
- Probabilidad de diagnóstico psiquiatrico segun el resultado de BERT
- Próximo paso recomendado
"""
)

LABELS = [
    "Esquizofrenia",
    "Otros tipos de esquizofrenia",
]

def extract_json_from_llm(text: str) -> dict:
    """
    Extrae el primer objeto JSON válido de una respuesta LLM.
    Funciona aunque haya texto antes/después o Markdown.
    """
    if not text or not text.strip():
        raise ValueError("Respuesta vacía del LLM")

    # Buscar el primer bloque JSON {...}
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError(f"No se encontró JSON en la respuesta:\n{text}")

    json_str = match.group(0)

    return json.loads(json_str)

def analyze_user_input(text: str):
    llm_extractor = Cohere(model="command-a-03-2025", api_key=os.getenv("COHERE_API_KEY"), temperature=0.0)

    prompt = f"""
        Analiza el siguiente texto clínico y extrae la información en formato JSON.

        Devuelve:
        - clinical_items: lista de síntomas o diagnósticos (en español o códigos ICD)
        - icd_codes: lista de códigos ICD detectados (si hay)
        - user_question: pregunta explícita del usuario (si existe, si no null)

        Texto:
        \"\"\"{text}\"\"\"

        Devuelve SOLO JSON PLANO.
        """
    
    prompt_template = PromptTemplate(
        input_variables="text",
        template=prompt
    )
    messages = [
        ChatMessage(role="system", content=system_prompt.format()),
        ChatMessage(role="user", content=prompt_template.format(text=text))
    ]

    response = llm_extractor.chat(messages)

    chat_text = response.message.content

    return extract_json_from_llm(chat_text)

def build_bert_input(clinical_items: list) -> str:
    """
    Convierte la lista de síntomas/diagnósticos
    en el formato esperado por BERT
    """
    return ", ".join(clinical_items)

def bert_predict(text):
    torch.set_default_device("cpu")
    MODEL_PATH = "models/dccuchile_bert-base-spanish-wwm-cased_codigos_final"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32
    )

    model.eval()

    if isinstance(text, list):
        bert_input_text = ", ".join(text)
    else:
        bert_input_text = text

    inputs = tokenizer(
        bert_input_text,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )

    inputs = {k: v.to("cpu") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)[0]

    prob_dict = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    predicted = max(prob_dict, key=prob_dict.get)

    return predicted, prob_dict

def neo4j_evidence(predictions):
    """
    Dado un código ICD, recupera casos reales del grafo Neo4J.
    """
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    )

    query = """
        MATCH (d_main:Diagnostico {terminoEN: $terminoEN})
        <-[:DIAGNOSTICO_PSIQUIATRICO]-
        (p:Paciente)
        WITH p, rand() AS r
        ORDER BY r
        LIMIT 25
        MATCH (p)-[:DIAGNOSTICO_ASOCIADO]->(d_assoc:Diagnostico)
        WHERE d_assoc.terminoEN <> $terminoEN
        RETURN d_assoc.terminoEN AS diagnostico, count(*) AS frecuencia
        ORDER BY frecuencia DESC
    """

    with driver.session() as session:
        result = session.run(query, terminoEN=predictions)
        data = [(r["diagnostico"], r["frecuencia"]) for r in result][:10]

    

    return data

def plot_bert_probabilities(prob_dict):
    labels = list(prob_dict.keys())
    values = list(prob_dict.values())

    fig, ax = plt.subplots()

    ax.barh(labels, values)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probabilidad")
    ax.set_title("Probabilidad estimada por el modelo BERT")
    for i, v in enumerate(values):
        plt.text(v + 0.02, i, f"{v:.2f}", va="center")

    return fig

def plot_diagnostic_frequencies(freq_data):
    if not freq_data:
        return None

    diagnoses, counts = zip(*freq_data)

    fig, ax = plt.subplots()
    ax.barh(diagnoses, counts)
    ax.set_xlabel("Frecuencia")
    ax.set_title("Diagnósticos asociados más frecuentes")
    ax.invert_yaxis()

    return fig

def final_llm_response(
    original_text: str,
    bert_probs: dict,
    graph_evidence: dict,
    user_question: str | None,
) -> str:
    
    llm_synthesizer = Cohere(model="command-a-03-2025", api_key=os.getenv("COHERE_API_KEY"), temperature=0.3)

    prompt = f"""
        Eres un asistente psiquiátrico experimental.

        Información disponible:

        Probabilidades diagnósticas (BERT):
        {bert_probs}

        Evidencia clínica (Neo4j):
        {graph_evidence}

        Pregunta del usuario:
        {user_question if user_question else "No hay pregunta explícita."}

        Redacta una respuesta clara, prudente y basada únicamente en la información proporcionada.
        Incluye advertencias éticas si procede e informacion sobre el diagnóstico.
        """
    
    prompt_template = PromptTemplate(
        input_variables="text",
        template=prompt
    )
    
    messages = [
        ChatMessage(role="system", content=system_prompt.format()),
        ChatMessage(role="user", content=prompt_template.format(text=text))
    ]

    response = llm_synthesizer.chat(messages)
    return response
