# Chatbot para Análisis de Diagnósticos de Esquizofrenia

El enfoque de este proyecto es desarrollar y construir un chatbot que analice y responda al historial de diagnósticos de un paciente con la probabilidad de padecer algún tipo o variante de esquizofrenia. El objetivo principal de esta herramienta es asistir en el proceso terapéutico mediante la implementación de un acompañante basado en inteligencia artificial (IA), que contribuya a facilitar y optimizar el trabajo de los profesionales de la salud mental.
Gracias a los grandes avances modernos en procesamiento de lenguaje natural y aprendizaje automático, acompañados de una interfaz web interactiva, se querrá desarrollar un asistente virtual para el terapeuta, sirviendo como herramienta de apoyo en dar un diagnóstico acorde y certero a cada paciente. De esta manera, se aspira a mejorar la calidad del acompañamiento psicológico y a promover un entorno de atención más accesible, preciso y humano mediante la integración de la inteligencia artificial en el ámbito de la salud mental.

## Índice
1. [Requisitos](#requisitos)
   - [Docker](#docker)
   - [Makefile](#makefile)
2. [Instrucciones para montarlo](#instrucciones-de-uso)

---

## Requisitos

Si ya tienes Docker y Makefile instalados, puedes saltarte esta sección.

### Docker
Es necesario tener Docker instalado en tu sistema.

- **Windows**: Sigue esta guía para instalar Docker:  
  https://docs.docker.com/desktop/setup/install/windows-install/

- **Linux**: Usa este tutorial según tu distribución:  
  https://docs.docker.com/engine/install/

### Makefile
Usamos un archivo Makefile para facilitar la ejecución de Docker. Asegúrate de tener `make` instalado.

#### En Windows
Si no tienes **Chocolatey** (choco) instalado, ejecuta el siguiente comando en PowerShell como administrador:
```
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

Luego instala `make` con el siguiente comando:
```
choco install make
```


#### En Linux
En la mayoría de distribuciones, `make` ya está instalado. Si no es así, usa uno de estos comandos según tu distribución:

- **Debian/Ubuntu**:
```
sudo apt update
sudo apt install make
```

- **Red Hat/Fedora**:
```
sudo dnf install make
```

- **Arch Linux**:
```
sudo pacman -S make
```


---


## Instrucciones de Uso

1. Abrir PowerShell como administrador si estás en Windows o una terminal si estás en Linux.

2. Clona el repositorio
```
git clone https://github.com/rfernr08/Mirage.git
```
O descarga manualmente el repositorio.

3. Ir al directorio del proyecto
```
cd Mirage
cd SIBI
```

4. Si tienes windows abre Docker Desktop

5. Inicia el proceso de construcción
- **Windows**:
```
make build
```

- **Linux**:
```
sudo make build
```


6. Ejecuta la aplicación
- **Windows**:
```
make run
```

- **Linux**:
```
sudo make run
```