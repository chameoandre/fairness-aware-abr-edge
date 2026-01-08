#!/bin/bash

# Crie um diretório para o ChromeDriver
mkdir -p /scripts/chromedriver
cd /scripts/chromedriver

# Baixe a versão correta do ChromeDriver
# wget https://chromedriver.storage.googleapis.com/129.0.6668.88/chromedriver_linux_arm64.zip  # verifique a versão correta
wget https://chromedriver.storage.googleapis.com/LATEST_RELEASE_99.0.4844
# Extraia o arquivo
unzip chromedriver_linux_arm64.zip

# Torne o ChromeDriver executável
chmod +x chromedriver
