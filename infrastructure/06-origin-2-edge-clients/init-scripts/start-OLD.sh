#!/bin/bash

# Iniciar o D-Bus
dbus-daemon --system &

# Iniciar o Xvfb
Xvfb :99 -screen 0 1920x1080x24 &

# Definir a variável DISPLAY
export DISPLAY=:99

# Iniciar o JMeter como servidor
jmeter-server &

