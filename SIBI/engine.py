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

load_dotenv()

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

    #response = llm_extractor.predict(prompt=prompt_template, text=text)
    response = llm_extractor.chat(messages)
        # DEBUG útil (déjalo de momento)
    print("RAW RESPONSE:", response)

    chat_text = response.message.content

    return extract_json_from_llm(chat_text)

def build_bert_input(clinical_items: list) -> str:
    """
    Convierte la lista de síntomas/diagnósticos
    en el formato esperado por BERT
    """
    return ", ".join(clinical_items)

def bert_predict(text):
    MODEL_PATH = "models\dccuchile_bert-base-spanish-wwm-cased_codigos_final"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    """
    Predict diagnósticos usando BERT a partir de síntomas normalizados.
    
    Args:
        normalized_text_items (list[str]): Lista de síntomas o códigos ICD
    
    Returns:
        predicted (str): diagnóstico con mayor probabilidad
        prob_dict (dict): probabilidades de todos los diagnósticos
        bert_input_text (str): texto que se pasó al modelo
    """

    # Convertimos lista a string (formato esperado por tu BERT)
    bert_input_text = ", ".join(text)

    # Tokenizamos
    inputs = tokenizer(
        bert_input_text,
        truncation=True,
        padding=True,
        return_tensors="pt"
    ).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    # Construir diccionario de probabilidades
    prob_dict = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

    # Diagnóstico más probable
    predicted = max(prob_dict, key=prob_dict.get)

    return predicted, prob_dict

def neo4j_evidence(predictions):
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    )
    """
    Dado un código ICD, recupera casos reales del grafo Neo4J.
    """
    query = """
    MATCH (d:Diagnostico {terminoEN: $terminoEN})<-[:`DIAGNOSTICO_PSIQUIATRICO`]-(p:Paciente)
    RETURN p.numero_historia AS paciente_id, d.terminoEN AS diagnostico, d.ICD10 as icd_code
    LIMIT 25
    """

    with driver.session() as session:
        results = session.run(query, terminoEN=predictions)

        data = [
            {
                "paciente_id": record["paciente_id"],
                "diagnostico": record["diagnostico"],
                "icd_code": record["icd_code"]
            }
            for record in results
        ]

    return {
        "diagnostico": predictions,
        "n_casos": len(data),
        "ejemplos": data
    }


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

"""
def build_query_engine(bert_tool, neo4j_tool):
    Settings.llm = Cohere(
        api_key=os.getenv("COHERE_API_KEY"),
        model="command-r",
        temperature=0.2
    )

    graph_store = Neo4jGraphStore(
        username=os.getenv("NEO4J_USER"),
        password=os.getenv("NEO4J_PASS"),
        url=os.getenv("NEO4J_URI"),
    )

    kg_index = KnowledgeGraphIndex.from_graph_store(
        graph_store=graph_store
    )

    query_engine = kg_index.as_query_engine(
        tools=[bert_tool, neo4j_tool],
        enforce_execution=True
    )

    return query_engine
"""