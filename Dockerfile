FROM python:3.10-slim

# Evita prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive

# Directorio de trabajo
WORKDIR /SIBI

# Dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiamos requirements
COPY conda-env/environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml
ENV PATH /opt/conda/envs/diet-recommender/bin:$PATH

# Copiamos el código y los modelos
COPY app .\SIBI
COPY models .\models\dccuchile_bert-base-spanish-wwm-cased_codigos_final

ENV PYTHONPATH="/app:${PYTHONPATH}"

# Variables de entorno Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

# Arranque
CMD ["streamlit", "run", "app/interface.py", "--server.port=8501", "--server.address=0.0.0.0"]
