# Setup Guide

This guide provides instructions for setting up and running Docker containers across different operating systems.

## Prerequisites

- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- Git (optional, for cloning repositories)

## Docker Installation

### Windows
1. Download Docker Desktop from [Docker Hub](https://hub.docker.com/editions/community/docker-ce-desktop-windows)
2. Run the installer and follow the prompts
3. If using Windows 10 Home, enable WSL 2 when prompted
4. Restart your computer when installation completes

### macOS
1. Download Docker Desktop from [Docker Hub](https://hub.docker.com/editions/community/docker-ce-desktop-mac)
2. Drag Docker.app to your Applications folder
3. Launch Docker Desktop and follow the prompts
4. Wait for the Docker daemon to start

### Linux (Ubuntu)
1. Follow steps from official [Docker Hub](https://docs.docker.com/engine/install/ubuntu/) website

## Git Installation
1. Go to the [Git Installation Website](https://git-scm.com/downloads) and install Git for your repository. Ensure you have selected "ADD TO PATH" if on Windows.
2. Make a Github Account, or sign in if you have one already 

## WSL Installation (Only for Windows)
1. Open a terminal window and enter:  wsl --install
2. Accept the admin prompt that comes up.
3. Go to Microsofft Store and install "Ubuntu"

## Clone ARIA-Guard Code
1. Open up terminal application
2. Clone the ARIA-Guard Repository: git clone https://github.com/alpergel/MiceVision.git
3. Download the pre-trained models from the [Google Drive Folder]() Then move them into ARIA-Guard/models/
4. Open the newly cloned folder via terminal: cd ARIA-Guard

## Windows Startup Steps
1. Go to Windows Search bar and look up "Ubuntu on Windows". Executing the following commands will put you in the ARIA-Guard folder path from the Ubuntu Virtual Machine.:
    cd ..
    cd ..
    cd mnt/c/Users/{Your Windows Username}/ARIA-Guard/
2. Go to Windows Search bar and open "Docker Desktop". If the application
opens and asks to restart to accommodate ”WSL”, please do so. Now, go to the
Ubuntu on Windows terminal and run the following command. This command
might take a while to run. Make sure to include the ”.” at the end of the
command.
    docker build -t aria-guard .
3. Run ARIA-Guard Interface:
    docker run -p 8501:8501 --gpus all aria-guard
