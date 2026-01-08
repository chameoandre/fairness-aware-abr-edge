import os
import subprocess

def contar_requisicoes(log_file, date, status_code=None):
    # Montar o comando grep
    command = f"grep '{date}' {log_file}"
    if status_code:
        command += f" | grep '{status_code}'"
    
    # Executar o comando e contar as linhas
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return len(result.stdout.splitlines())

# Defina os arquivos de log e a data de interesse
# log_dir = "/scripts/logs/nginx_logs"
log_dir = "/Users/chameoandre/Google-Drive-chameoandre/APRENDIZADO-TECNOLOGIAS/experimentos-doutorado-abr-streaming/abr-streaming-experiments/05-origin-2-edge-clients/nginx_logs"


logs = {
    "edge1": os.path.join(log_dir, "edge1/access.log"),
    "edge2": os.path.join(log_dir, "edge2/access.log"),
    "load-balancer": os.path.join(log_dir, "load-balancer/access.log")
}

date = "14/Nov/2024"

# Coletar as requisições de cada servidor
requisicoes = {server: contar_requisicoes(log, date) for server, log in logs.items()}
print(requisicoes)

# Função para calcular o Índice de Justiça de Jain
def jain_index(requisicoes):
    requisicoes_values = list(requisicoes.values())
    numerator = sum(requisicoes_values) ** 2
    denominator = len(requisicoes_values) * sum(x ** 2 for x in requisicoes_values)
    return numerator / denominator

# Calcular e imprimir o índice de Jain
indice_jain = jain_index(requisicoes)
print(f"Requisições por servidor: {requisicoes}")
print(f"Índice de Justiça de Jain: {indice_jain:.4f}")


