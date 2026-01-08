#!/usr/bin/env python3
"""
collect_docker_metrics.py
--------------------------------
Monitoramento automatizado de uso de CPU e memória de containers Docker durante simulações.

Cria um arquivo CSV com colunas:
timestamp, container, cpu_pct, mem_pct, mem_used, mem_limit

Compatível com Docker Desktop (macOS/ARM64) e Docker Engine Linux.

Uso standalone:
    python3 collect_docker_metrics.py --interval 5 --duration 400 --output system_metrics.csv

Integração:
    from collect_docker_metrics import start_metrics_collection, stop_metrics_collection
"""

import os
import csv
import time
import subprocess
import threading
from datetime import datetime

# Controle global de execução
_stop_flag = False
_thread = None


def _find_docker_binary():
    """Detecta automaticamente o binário do Docker em diferentes ambientes."""
    candidates = [
        "/usr/local/bin/docker",
        "/opt/homebrew/bin/docker",
        "/usr/bin/docker",
        "docker"
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return "docker"


def _prepare_docker_env():
    """Garante que o subprocess herde as variáveis corretas para acesso ao Docker."""
    env = os.environ.copy()
    if "DOCKER_HOST" not in env:
        user = os.getenv("USER", "chameoandre")
        socket_path = f"unix:///Users/{user}/.docker/run/docker.sock"
        if os.path.exists(f"/Users/{user}/.docker/run/docker.sock"):
            env["DOCKER_HOST"] = socket_path
    return env


def _collect_metrics(interval, output_file, wait_for_clients=True):
    """Loop interno de coleta periódica de métricas Docker."""
    global _stop_flag

    docker_bin = _find_docker_binary()
    docker_env = _prepare_docker_env()

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "container", "cpu_pct", "mem_pct", "mem_used", "mem_limit"])

        print(f"[METRICS] Coleta iniciada (intervalo = {interval}s) usando '{docker_bin}'.")

        # Espera opcional pelos containers client_*
        if wait_for_clients:
            print("[METRICS] Aguardando containers client_* iniciarem...")
            start_wait = time.time()
            while not _stop_flag:
                ps_result = subprocess.run(
                    [docker_bin, "ps", "--format", "{{.Names}}"],
                    capture_output=True, text=True, env=docker_env
                )
                active = [n for n in ps_result.stdout.splitlines() if n.startswith("client_")]
                if active:
                    print(f"[METRICS] Containers detectados: {active}")
                    break
                if time.time() - start_wait > 60:
                    print("[WARN] Nenhum container client_* detectado após 60s. Iniciando coleta mesmo assim.")
                    break
                time.sleep(2)

        # Loop principal de coleta
        while not _stop_flag:
            try:
                result = subprocess.run(
                    [
                        docker_bin, "stats", "--no-stream",
                        "--format", "{{.Name}},{{.CPUPerc}},{{.MemPerc}},{{.MemUsage}}"
                    ],
                    capture_output=True, text=True, timeout=5, env=docker_env
                )

                if result.returncode != 0:
                    print(f"[WARN] docker stats retornou código {result.returncode}")
                    time.sleep(interval)
                    continue

                if not result.stdout.strip():
                    print(f"[WARN] Nenhum container ativo. Aguardando {interval}s...")
                    time.sleep(interval)
                    continue

                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                for line in result.stdout.strip().splitlines():
                    parts = line.split(",")
                    if len(parts) == 4:
                        # Exemplo de MemUsage: "112.9MiB / 8.722GiB"
                        mem_parts = parts[3].split(" / ") if " / " in parts[3] else (parts[3], "")
                        mem_used = mem_parts[0].strip()
                        mem_limit = mem_parts[1].strip() if len(mem_parts) > 1 else ""
                        writer.writerow([ts, parts[0], parts[1], parts[2], mem_used, mem_limit])
                
                print(f"[DEBUG] Amostra coletada às {ts}: {len(result.stdout.strip().splitlines())} containers")
                csvfile.flush()

            except Exception as e:
                print(f"[ERROR] Falha ao coletar métricas Docker: {e}")

            time.sleep(interval)

    print("[METRICS] Coleta finalizada.")


def start_metrics_collection(interval=5, duration=None, output_file="system_metrics.csv"):
    """
    Inicia a coleta assíncrona de métricas do Docker.
    - interval: segundos entre coletas
    - duration: tempo total em segundos (None = coleta contínua até stop)
    """
    global _stop_flag, _thread
    _stop_flag = False

    # Inicia a thread de coleta
    _thread = threading.Thread(target=_collect_metrics, args=(interval, output_file), daemon=True)
    _thread.start()
    print(f"[METRICS] Thread de coleta iniciada. Salvando em {output_file}")

    # Aguarda um pequeno intervalo para garantir inicialização antes do timer
    time.sleep(2)

    # Thread de parada (opcional)
    if duration:
        def stop_after_delay():
            print(f"[METRICS] Timer de parada configurado para {duration}s.")
            time.sleep(duration)
            stop_metrics_collection()
        stopper = threading.Thread(target=stop_after_delay, daemon=True)
        stopper.start()

    return output_file


def stop_metrics_collection():
    """Encerra a coleta de métricas."""
    global _stop_flag
    _stop_flag = True
    print("[METRICS] Sinal de parada recebido. Encerrando coleta...")


# Execução standalone
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Docker container metrics (CPU/Mem).")
    parser.add_argument("--interval", type=int, default=5, help="Intervalo entre coletas (segundos)")
    parser.add_argument("--duration", type=int, default=400, help="Duração total da coleta (segundos)")
    parser.add_argument("--output", type=str, default="system_metrics.csv", help="Arquivo de saída CSV")
    args = parser.parse_args()

    try:
        start_metrics_collection(args.interval, args.duration, args.output)
        time.sleep(args.duration)
    except KeyboardInterrupt:
        pass
    finally:
        stop_metrics_collection()
