import os
from dotenv import load_dotenv
import cohere
from llama_index.llms.cohere import Cohere

# Cargar variables de entorno desde .env
load_dotenv()

# Obtener la API key de Cohere desde .env
api_key = os.getenv("COHERE_API_KEY")

if not api_key:
    raise ValueError("COHERE_API_KEY no encontrada en el archivo .env")

# Inicializar cliente de Cohere
llm_extractor = Cohere(model="command-r", api_key=os.getenv("COHERE_API_KEY"), temperature=0.0)

# Hacer una prueba simple
response = llm_extractor.complete(prompt="¿Hola, cómo estás?")


print("Respuesta de Cohere:")
print(response.message.content[0].text)
