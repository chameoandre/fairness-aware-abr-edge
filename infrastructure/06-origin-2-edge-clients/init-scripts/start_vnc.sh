# =========== VERSION 02 [NEWEST] ===========

#!/bin/bash

# Defina a variável de ambiente USER (apenas para garantir compatibilidade com algumas ferramentas)
export USER=root

# Configuração de ambiente para iniciar o VNC
export DISPLAY=:1

# Crie o diretório do VNC caso não exista
mkdir -p ~/.vnc

# Se a senha do VNC existir, remova para forçar acesso sem senha
# rm -f ~/.vnc/passwd

# Configure a senha do VNC se ainda não estiver configurada
if [ ! -f ~/.vnc/passwd ]; then
    echo "Configurando a senha do VNC..."
    echo "vncpassword" | vncpasswd -f > ~/.vnc/passwd
    chmod 600 ~/.vnc/passwd
fi

# Inicie o Xvfb em segundo plano (servidor de framebuffer virtual)
echo "Iniciando Xvfb..."
Xvfb :1 -screen 0 1280x720x16 &
sleep 2
# Inicie o LXDE (ambiente de desktop leve)

echo "Iniciando LXDE..."
lxsession &
sleep 2

# Inicie o servidor VNC com as configurações desejadas
echo "Iniciando o servidor VNC..."
x11vnc -display :1 -xkb -forever -shared -noxdamage -repeat -capslock -geometry 1280x720 -passwd vncpassword &
# x11vnc -display :1 -xkb -forever -shared -noxdamage -repeat -capslock -geometry 1280x720 -nopw -auth guess &


# Mantenha o contêiner ativo
echo "Ambiente VNC iniciado. Aguardando conexões..."
tail -f /dev/null


# =========== VERSION 01 [OLD] ===========

# #!/bin/bash

# # Defina a variável de ambiente USER
# export USER=root

# # Setar monitor
# # export DISPLAY=:1

# # Configuração de ambiente para iniciar o VNC
# ENV DISPLAY=:1
# RUN mkdir -p ~/.vnc

# # Inicie o Xvfb em segundo plano
# # Xvfb :1 -screen 0 1920x1080x24 &
# Xvfb :1 -screen 0 1024x768x16 &

# # Inicie o VNC server com LXDE
# x11vnc -display :1 -forever -shared -passwd vncpassword &

# # Inicie o Openbox
# openbox &

# # Inicie o LXDE
# startlxde &

# # Verificar este
# lxsession &

# # Inicie o VNC server como processo principal para manter o contêiner ativo
# # x11vnc -display :1 -forever -shared -passwd vncpassword -nopw
# x11vnc -display :1 -xkb -forever -shared -noxdamage -repeat -capslock -geometry 1280x720 -passwd vncpassword -nopw


# # # Execute o comando padrão
# # exec "$@"
