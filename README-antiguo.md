# Project for a Chatbot that predicts Schizofrenia in a pacient

Mirage is a chatbot designed to receive a list of symptoms that a person could have and provide an analysis of the likelihood of that person having Schizophrenia. The system uses Python and various technologies like Docker to create an interactive experience. By receiving information about the symptoms, the chatbot offers an analysis adjusted to each profile.

## Table of Contents
1. [Requirements](#Requirements)
   - [Docker](#docker)
   - [Makefile](#makefile)
2. [Usage Instructions](#Usage_Instructions)

---

## Requirements

### Docker
Docker must be installed on your system.

- **Windows**: Follow this guide to install Docker:  
  https://docs.docker.com/desktop/setup/install/windows-install/

- **Linux**: Use this tutorial for your distribution:  
  https://docs.docker.com/engine/install/

### Makefile
A Makefile is used to simplify Docker execution. Make sure you have make installed. Below are the instructions based on your operating system:

#### On Windows
If you don't have **Chocolatey** (choco) installed, run the following command in PowerShell as administrator:
```
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

Then, install make with this command:
```
choco install make
```


#### On Linux
In most distributions, `make` is already installed. If not, use one of these commands based on your distribution:

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


## Usage Instructions

1. Open PowerShell as Administrator if you're on Windows, or a terminal if you're on Linux.

2. Clone the repository

```
git clone https://github.com/rfernr08/Mirage.git
```
Or manually download the repository.

3. Go to the project directory
```
cd Mirage
cd project
```

4. If you're on Windows, open Docker Desktop

5. Start the build process **(it may take up to two hours)**
- **Windows**:
```
make build
```

- **Linux**:
```
sudo make build
```


6. Run the application
- **Windows**:
```
make run
```

- **Linux**:
```
sudo make run
```