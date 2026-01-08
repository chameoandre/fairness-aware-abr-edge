#!/usr/bin/env python3
import subprocess
import argparse
import os
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collect_docker_metrics import start_metrics_collection, stop_metrics_collection

# ===============================
# === SIMULATION COMMAND EXECUTION ===
# ===============================
# python3 run_clients_in_docker_02.py --num_clients 10 --abr_mode DASH-DEFAULT --sim_time 200
# python3 run_clients_in_docker_02.py --num_clients 10 --abr_mode DASH-QOE-BASED --sim_time 200
# python3 run_clients_in_docker_02.py --num_clients 10 --abr_mode BOLA --sim_time 200
# python3 run_clients_in_docker_02.py --num_clients 10 --abr_mode HOTDASH --sim_time 200
# python3 run_clients_in_docker_02.py --num_clients 10 --abr_mode L2A --sim_time 200
# python3 run_clients_in_docker_02.py --num_clients 10 --abr_mode PENSIEVE-BASELINE --sim_time 200
# python3 run_clients_in_docker_02.py --num_clients 10 --abr_mode PENSIEVE-QOE-BASED --sim_time 200
# python3 run_clients_in_docker_02.py --num_clients 10 --abr_mode PENSIEVE-FAIRNESS --sim_time 200
# python3 run_clients_in_docker_02.py --num_clients 10 --abr_mode PENSIEVE-MULTI-FAIRNESS-AWARE	 --sim_time 200



# python3 run_clients_in_docker_02.py --num_clients 10 --abr_mode DASH-DEFAULT --reuse_containers --sim_time 200

# ===============================
# === MAIN CONFIGURATIONS ===
# ===============================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCKER_BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../06-origin-2-edge-clients"))
SCRIPTS_DIR = os.path.abspath(CURRENT_DIR)

# Nome da imagem principal usada pelos containers
DOCKER_IMAGE_NAME = "custom-seleniumarm/node-chromium:v2.0"
DOCKERFILE_PATH = os.path.join(DOCKER_BASE_DIR, "DockerFileClient")

# Caminho de saída para os resultados da simulação
OUTPUT_DIR = os.path.join(SCRIPTS_DIR, "simulations/output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[INFO] Docker base dir: {DOCKER_BASE_DIR}")
print(f"[INFO] Simulation scripts dir: {SCRIPTS_DIR}")
print(f"[INFO] Output dir: {OUTPUT_DIR}")

# Diretórios externos e variáveis do ambiente (ajuste conforme necessário)
ENV_PATHS = {
    "TRACES_DIR": os.path.join(SCRIPTS_DIR, "traces"),
    "SCRIPTS_DIR": SCRIPTS_DIR,
    "INIT_SCRIPTS": os.path.join(DOCKER_BASE_DIR, "init-scripts"),
    "X11_SOCKET": "/tmp/.X11-unix",
    "JMETER_PROPERTIES": os.path.join(DOCKER_BASE_DIR, "jmeter.properties"),
    "NGINX_LOGS": os.path.join(DOCKER_BASE_DIR, "nginx_logs"),
}

# ===============================
# === FUNÇÕES AUXILIARES ===
# ===============================

def ensure_docker_image():
    """Verifica se a imagem Docker existe; se não, executa o build."""
    print(f"[CHECK] Verificando se a imagem {DOCKER_IMAGE_NAME} existe...")
    result = subprocess.run(["docker", "images", "-q", DOCKER_IMAGE_NAME],
                            capture_output=True, text=True)
    if not result.stdout.strip():
        print(f"[BUILD] Imagem não encontrada. Construindo {DOCKER_IMAGE_NAME}...")
        subprocess.run([
            "docker", "build",
            "-t", DOCKER_IMAGE_NAME,
            "-f", DOCKERFILE_PATH,
            DOCKER_BASE_DIR  # contexto de build
        ], check=True)
    else:
        print(f"[OK] Imagem {DOCKER_IMAGE_NAME} já existente.")


def ensure_container_exists(container_name, image_name, network_name):
    """Verifica se o container existe; se não, cria um novo e o deixa ativo em modo idle."""
    # Verifica se o container já existe
    result = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}"],
        capture_output=True, text=True
    )
    containers = result.stdout.splitlines()

    if container_name in containers:
        print(f"[INFO] Container {container_name} já existe. Reutilizando.")
        # Se ele estiver parado, inicia novamente
        state = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
            capture_output=True, text=True
        ).stdout.strip()
        if state != "true":
            subprocess.run(["docker", "start", container_name])
        return

    # Caso não exista, cria um container em background, pronto para execuções futuras
    print(f"[CREATE] Criando novo container {container_name}...")
    subprocess.run([
        "docker", "run", "-d",  # modo background
        "--name", container_name,
        "--network", network_name,
        "--privileged",
        "--cap-add=NET_ADMIN",
        "--cap-add=NET_RAW",
        "-v", f"{ENV_PATHS['SCRIPTS_DIR']}:/scripts",
        "-v", f"{ENV_PATHS['TRACES_DIR']}:/traces",
        "-v", f"{ENV_PATHS['INIT_SCRIPTS']}:/init-scripts",
        "-v", f"{ENV_PATHS['NGINX_LOGS']}:/nginx_logs",
        "-v", f"{OUTPUT_DIR}:/simulations/output/",
        image_name,
        "sleep", "infinity"  # container permanece ativo aguardando comandos
    ], check=True)
    print(f"[OK] Container {container_name} criado e ativo.")


def run_container(client_id, abr_mode, execution_mode, sim_time, output_dir, reuse_containers=True):
    """Executa um container individual de cliente com ou sem reutilização."""
    container_name = f"client_{client_id}"
    network_name = detect_bridge_streaming_network()
    client_output = output_dir

    # === Verifica ou cria container (somente se reuse_containers=True) ===
    if reuse_containers:
        ensure_container_exists(container_name, DOCKER_IMAGE_NAME, network_name)
        print(f"[INFO] Executando simulação no container existente: {container_name}")

        # Comando para executar dentro de um container já existente
        command = [
            "docker", "exec",
            "-e", f"CLIENT_ID={client_id}",
            "-e", f"ABR_MODE={abr_mode}",
            "-e", f"EXECUTION_MODE={execution_mode}",
            "-e", f"SIMULATION_TIME={sim_time}",
            container_name,
            "/bin/bash", "-c",
            f"source /scripts/venv/bin/activate && "
            f"python3 /scripts/abr-client-aware-fairness-healthcare-5.py"
        ]
    else:
        print(f"[INFO] Criando e executando novo container {container_name}...")
        command = [
            "docker", "run", "--rm",
            "--network", network_name,
            "--name", container_name,
            "--privileged",
            "--cap-add=NET_ADMIN",
            "--cap-add=NET_RAW",
            "-e", f"CLIENT_ID={client_id}",
            "-e", f"ABR_MODE={abr_mode}",
            "-e", f"EXECUTION_MODE={execution_mode}",
            "-e", f"SIMULATION_TIME={sim_time}",
            "-e", f"DISPLAY={os.getenv('DISPLAY', ':0')}",
            "-e", "QT_X11_NO_MITSHM=1",
            "-v", f"{ENV_PATHS['SCRIPTS_DIR']}:/scripts",
            "-v", f"{ENV_PATHS['TRACES_DIR']}:/traces",
            "-v", f"{ENV_PATHS['INIT_SCRIPTS']}:/init-scripts",
            "-v", f"{ENV_PATHS['NGINX_LOGS']}:/nginx_logs",
            # "-v", f"{OUTPUT_DIR}:/simulations/output/",
            "-v", f"{output_dir}:/simulations/output/",
            DOCKER_IMAGE_NAME,
            "/bin/bash", "-c",
            "source /scripts/venv/bin/activate && "
            "python3 /scripts/abr-client-aware-fairness-healthcare-5.py"
        ]

    print(f"[INFO] Starting container {container_name} with {abr_mode}")

    try:
        result = subprocess.run(command, capture_output=True, text=True)
        log_path = os.path.join(output_dir, f"execution_client_{client_id}.log")

        with open(log_path, "w") as log_file:
            log_file.write(result.stdout + "\n" + result.stderr)
        print(f"[OK] {container_name} finished successfully.")

    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] Container {container_name} excedeu limite, forçando stop.")
        subprocess.run(["docker", "stop", container_name], stdout=subprocess.DEVNULL)
    except Exception as e:
        print(f"[ERROR] Container {container_name} falhou: {e}")


def detect_bridge_streaming_network():
    """Detecta o nome real da rede bridge-streaming (com prefixo do projeto Docker Compose)."""
    result = subprocess.run(["docker", "network", "ls", "--format", "{{.Name}}"],
                            capture_output=True, text=True)
    for net in result.stdout.splitlines():
        if "bridge-streaming" in net:
            print(f"[NET] Detected network: {net}")
            return net
    print("[NET] Network 'bridge-streaming' not found. Using default 'bridge'.")
    return "bridge"


def ensure_docker_network():
    network_name = "bridge-streaming"
    result = subprocess.run(["docker", "network", "ls", "--format", "{{.Name}}"],
                            capture_output=True, text=True)
    networks = result.stdout.splitlines()
    if network_name not in networks:
        print(f"[NET] Rede {network_name} não encontrada. Criando...")
        subprocess.run(["docker", "network", "create", "--driver", "bridge", network_name], check=True)
        print(f"[NET] Rede {network_name} criada com sucesso.")
    else:
        print(f"[NET] Rede {network_name} já existente.")

# ===============================
# === MAIN (PONTO DE ENTRADA) ===
# ===============================

def main_old():
    parser = argparse.ArgumentParser(description="Parallel container launcher for client simulations.")
    parser.add_argument("--num_clients", type=int, default=5, help="Número de containers de cliente a executar")
    parser.add_argument("--abr_mode", type=str, default="DASH-DEFAULT", help="Modo de algoritmo ABR")
    parser.add_argument("--execution_mode", type=str, default="process", help="Modo de execução dentro do container")
    parser.add_argument("--sim_time", type=int, default=400, help="Tempo de simulação (segundos)")
    parser.add_argument("--image_name", type=str, default=DOCKER_IMAGE_NAME, help="Imagem Docker a ser utilizada")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Diretório de saída dos resultados")
    parser.add_argument("--reuse_containers", action="store_true", help="Reutiliza containers existentes")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    ensure_docker_network() #check if docker network exists, if not create it!
    ensure_docker_image()

    
    # === [INÍCIO] COLETA DE MÉTRICAS DOCKER ===
    metrics_file = os.path.join(args.output_dir, "system_metrics.csv")
    start_metrics_collection(interval=5, duration=args.sim_time + 60, output_file=metrics_file)
    # ============================================
    
    start_time = time.time()
    print(f"[LAUNCHER] Running {args.num_clients} containers in parallel using image {args.image_name}")

    with ThreadPoolExecutor(max_workers=args.num_clients) as executor:
        futures = [
            executor.submit(run_container, i + 1, args.abr_mode, args.execution_mode,
                            args.sim_time, args.output_dir)
            for i in range(args.num_clients)
        ]
        for f in as_completed(futures):
            f.result()

    duration = time.time() - start_time
    print(f"[DONE] All containers completed in {duration:.2f} seconds.")

    # === [FINALIZING] DOCKER METRIC COLLECTOR ===
    stop_metrics_collection()
    print(f"[METRICS] Coleta encerrada. Resultados salvos em: {metrics_file}")
    # =============================================


def main():
    parser = argparse.ArgumentParser(description="Parallel container launcher for client simulations.")
    parser.add_argument("--num_clients", type=int, default=5, help="Número de containers de cliente a executar")
    parser.add_argument("--abr_mode", type=str, default="DASH-DEFAULT", help="Modo de algoritmo ABR")
    parser.add_argument("--execution_mode", type=str, default="process", help="Modo de execução dentro do container")
    parser.add_argument("--sim_time", type=int, default=400, help="Tempo de simulação (segundos)")
    parser.add_argument("--image_name", type=str, default=DOCKER_IMAGE_NAME, help="Imagem Docker a ser utilizada")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Diretório de saída dos resultados")
    parser.add_argument("--reuse_containers", action="store_true", help="Reutiliza containers existentes")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # === Cria diretório específico para esta simulação ===
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sim_dir_name = f"simulation-{timestamp}-{args.abr_mode}-{args.num_clients}clients"
    sim_output_dir = os.path.join(args.output_dir, sim_dir_name)
    os.makedirs(sim_output_dir, exist_ok=True)

    print(f"[INFO] Logs e métricas serão salvos em: {sim_output_dir}")

    # === Configuração e verificação de rede/imagem ===
    ensure_docker_network()
    ensure_docker_image()

    # === Início da coleta de métricas Docker ===
    metrics_file = os.path.join(sim_output_dir, "system_metrics.csv")
    start_metrics_collection(interval=5, duration=args.sim_time + 60, output_file=metrics_file)

    start_time = time.time()
    print(f"[LAUNCHER] Running {args.num_clients} containers in parallel using image {args.image_name}")

    # === Execução paralela dos containers ===
    with ThreadPoolExecutor(max_workers=args.num_clients) as executor:
        futures = [
            executor.submit(
                run_container,
                i + 1,
                args.abr_mode,
                args.execution_mode,
                args.sim_time,
                sim_output_dir,
                args.reuse_containers
            )
            for i in range(args.num_clients)
        ]
        for f in as_completed(futures):
            f.result()

    # === Finalização ===
    duration = time.time() - start_time
    print(f"[DONE] All containers completed in {duration:.2f} seconds.")

    # === Finaliza a coleta de métricas ===
    stop_metrics_collection()
    print(f"[METRICS] Coleta encerrada. Resultados salvos em: {metrics_file}")




if __name__ == "__main__":
    main()
