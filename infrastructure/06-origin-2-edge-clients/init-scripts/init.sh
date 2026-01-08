#!/bin/bash

# Executar o script original (por exemplo, para iniciar o VNC e o Xvfb)
echo "Carregando start_vnc ..."
/usr/local/bin/start_vnc.sh &

# Caminho fixo para o requirements.txt
REQ_FILE="/config/requirements.txt"

# Verificar e criar o ambiente virtual, se necessário
if [ ! -d "/scripts/venv" ]; then
    echo "Criando ambiente virtual no diretório compartilhado..."
    rm -rf /scripts/venv

    python3.10 -m venv /scripts/venv
    /scripts/venv/bin/python3.10 -m ensurepip --default-pip
    # /scripts/venv/bin/python3.10 -m pip install --upgrade pip setuptools wheel --no-cache-dir
    /scripts/venv/bin/python3.10 -m pip install --force-reinstall --upgrade pip setuptools wheel
    echo "Instalando pacotes do PyTorch..."
    /scripts/venv/bin/python3.10 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

    echo "Instalando dependências do requirements.txt..."

    # /scripts/venv/bin/python3.10 -m pip install -r /scripts/requirements.txt --no-cache-dir
    # Usar o requirements.txt do local fixo
    echo "Instalando dependências do requirements.txt..."
    if [ -f "$REQ_FILE" ]; then
        /scripts/venv/bin/python3.10 -m pip install -r "$REQ_FILE" --no-cache-dir
    else
        echo "Arquivo $REQ_FILE não encontrado. Dependências não instaladas."
    fi


    echo "Listando pacotes instalados:"
    /scripts/venv/bin/pip list
else
    echo "Ambiente virtual já existe. Pulando criação."
fi

# AJUSTANDO O ACESSO DO DIRETÓRIO SCRIPTS PARA MAHIMAHI
# ======================================================
# Ajustar permissões do diretório mapeado se necessário
if [ -d "/scripts" ]; then
    echo "Ajustando permissões para o diretório mapeado /scripts..."
    chown -R mahimahi-user:mahimahi-user /scripts
fi

# Permitir que o usuário alterne manualmente para 'mahimahi-user' quando necessário
echo "Para alternar para o usuário 'mahimahi-user', use:"
echo "    su - mahimahi-user"


# Continue como root
exec "$@"



# Ativar o ambiente virtual
echo "Ativando o ambiente virtual..."
source /scripts/venv/bin/activate

# Verificar se um comando foi fornecido ao contêiner
if [ "$#" -gt 0 ]; then
    echo "Executando comando: $@"
    exec "$@"
else
    echo "Nenhum comando fornecido. Iniciando modo interativo..."
    exec /bin/bash
fi
