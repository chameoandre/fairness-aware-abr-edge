#!/bin/bash

# Inicie o Xvfb
Xvfb :99 -ac &
export DISPLAY=:99

# Configure o DBus
mkdir -p /var/run/dbus
dbus-daemon --system --fork
export DBUS_SESSION_BUS_ADDRESS=/dev/null

# Adicionar uma pequena pausa para garantir que todos os serviços estejam prontos
sleep 2

# Inicie o Chromium
# /usr/bin/chromium-browser --no-sandbox --disable-dev-shm-usage --remote-debugging-port=9222 --headless --disable-gpu --disable-software-rasterizer --disable-features=VizDisplayCompositor
# /usr/bin/chromium --no-sandbox --disable-dev-shm-usage --remote-debugging-port=9222 --headless --disable-gpu --disable-software-rasterizer --disable-features=VizDisplayCompositor
chromium --no-sandbox --disable-dev-shm-usage --remote-debugging-port=9222 --headless --disable-gpu --disable-software-rasterizer --disable-features=VizDisplayCompositor

# Abre um shell interativo com o ambiente virtual ativado
/bin/bash -c "source /scripts/venv/bin/activate && exec /bin/bash"