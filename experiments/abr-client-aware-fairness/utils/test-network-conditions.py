# CODE WITH ERROR - NOT WORKINKG

import os
import logging
import time
from threading import Thread

default_interface = "eth0"

def execute_command(command):
    result = os.system(command)
    if result != 0:
        logging.error(f"Command failed: {command}")
    return result

def set_network_conditions(interface, latency, throughput, loss):
    interface = str(interface)
    latency = str(latency)
    throughput = str(throughput)
    loss = str(loss)
    interface = str(interface)

    # Adicionar uma nova QDisc de tipo "htb" e configurar o throughput, latência e perda
    # sudo tc qdisc add dev eth0 root handle 1: htb default 11
    # sudo tc class add dev eth0 parent 1: classid 1:1 htb rate 1mbit
    # sudo tc class add dev eth0 parent 1:1 classid 1:11 htb rate 1mbit
    # Adicionar a QDisc netem para latência e perda
    # sudo tc qdisc add dev eth0 parent 1:11 handle 10: netem delay 90ms loss 3%

    execute_command(f"sudo tc qdisc add dev {interface} root handle 1: htb default 11")
    execute_command(f"sudo tc class add dev {interface} parent 1: classid 1:1 htb rate {throughput}")
    execute_command(f"sudo tc class add dev {interface} parent 1:1 classid 1:11 htb rate {throughput}")
    execute_command(f"sudo tc qdisc add dev {interface} parent 1:11 handle 10: netem delay {latency}ms loss {loss}%")
        
    logging.info(f"Configurações de Latência: {latency}, Throughput: {throughput} e Packet Loss: {loss} aplicadas na interface {interface}.")

def reset_network_conditions(interface):
    logging.info(f"Resetando as condições de rede no namespace {interface}.")
    # Tentativa de resetar múltiplas vezes se falhar
    for _ in range(3):
        result = execute_command(f"sudo tc qdisc del dev {interface} root")
        if result == 0:
            break
        
        # comando padrão para deletar regras TC na interface eth0
        # sudo tc qdisc del dev eth0 root
def show_network_conditions(interface):
    logging.info(f"Mostrando as atuais configurações de {interface}.")
    execute_command(f"sudo tc qdisc show dev {interface}")
    execute_command(f"tc class show dev {interface}")

def make_request(interface,command,latency, throughput, loss):
    try:
        set_network_conditions(interface, latency, throughput, loss)
        # response = os.system(f"sudo ip netns exec {interface} curl -s -o /dev/null -w '%{{http_code}}' {url}")
        # response = os.system(f"curl -s -o /dev/null -w '%{http_code}' {url}")
        response = os.system(command)
        print(f"Response Code: {response} from {url} in interface {interface}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        reset_network_conditions(interface)

def run_simulation(interface,url, conditions):
    threads = []
    for i, condition in enumerate(conditions):
        latency, throughput, loss = condition
        # ns_name = f"netns_{i}"
        thread = Thread(target=make_request, args=(interface,url,latency, throughput, loss))
        threads.append(thread)
        thread.start()
        time.sleep(5)

    for thread in threads:
        thread.join()

# Defina as condições (latência, throughput, perda)
conditions = [
    (0, "NO SHAPPING", 0),   # NO SHAPPING - BEST CONDITIONS
    (0, "100kbit", 0),   # 50ms de latência, 500kbit/s, 0% de perda
    (5, "250kbit", 0),   # 50ms de latência, 500kbit/s, 0% de perda
    (5, "500kbit", 0),   # 50ms de latência, 500kbit/s, 0% de perda
    (10, "500kbit", 0),   # 50ms de latência, 500kbit/s, 0% de perda
    (25, "500kbit", 0),   # 50ms de latência, 500kbit/s, 0% de perda
    (50, "500kbit", 1),   # 50ms de latência, 500kbit/s, 1% de perda
    (100, "500kbit", 2),   # 
    (200, "500kbit", 5),   # 
    (300, "500kbit", 10),   # 
    (50, "1mbit", 0.1),  # 100ms de latência, 1Mbit/s, 0.1% de perda
    (100, "1mbit", 0.5),  # 100ms de latência, 1Mbit/s, 0.1% de perda
    (100, "1mbit", 1),  # 100ms de latência, 1Mbit/s, 0.1% de perda
    (100, "1mbit", 5),  #
    (200, "1mbit", 5),  #
    (200, "1mbit", 10),  #
    (50, "5mbit", 0.5),  # 200ms de latência, 5Mbit/s, 0.5% de perda
    (100, "5mbit", 1),  # 200ms de latência, 5Mbit/s, 1.0% de perda
    (100, "5mbit", 5),  #
    (100, "5mbit", 10),
    (10, "50mbit", 0),  # 
    (50, "50mbit", 0.5),  # 
    (50, "50mbit", 3),  # 
    (100, "50mbit", 1),  # 
    (200, "50mbit", 2),  # 
    (200, "50mbit", 5),  # 
    (200, "50mbit", 10),  # 
    (10, "100mbit", 0),  # 
    (50, "100mbit", 0.5),  # 
    (100, "100mbit", 1),  # 
    (200, "100mbit", 2),  # 
    (200, "100mbit", 5),  # 
    (200, "100mbit", 10),  # 
    (10, "250mbit", 0),  # 
    (50, "250mbit", 0.5),  # 
    (100, "250mbit", 1),  # 
    (200, "250mbit", 2),  # 
    (200, "250mbit", 5),  # 
    (200, "250mbit", 10),  # 
    (10, "500mbit", 0),  # 
    (25, "500mbit", 0.5),  # 
    (50, "500mbit", 0.5),  # 
    (100, "500mbit", 0),  # 
    (100, "500mbit", 1),  # 
    (200, "500mbit", 2),  # 
    (200, "500mbit", 5),  # 
    (250, "500mbit", 10)  # 
]

download = [
    (10, "10MB", 'curl -o /dev/null http://speedtest.tele2.net/10MB.zip'),
    (50, "50MB", 'curl -o /dev/null http://speedtest.tele2.net/50MB.zip'),
    (100, "100MB", 'curl -o /dev/null http://speedtest.tele2.net/100MB.zip')
    # (1000, "1000MB", 'curl -o /dev/null http://speedtest.tele2.net/1000MB.zip'),
    # (1500, "1500MB", 'curl -o /dev/null http://speedtest.tele2.net/1500MB.zip')
]

url="globo.com"
command_1 = (f"ping -c 4 {url}")
command_2 = (f"curl -s {url}")
logging.basicConfig(level=logging.INFO)
# run_simulation(default_interface, command_1, conditions)
opcao = ""
while opcao !=0:
    # print("========================================================")
    # print("Opção 1: Configurar condições de Rede MANUALMENTE")
    # print("Opção 2: Usar modelo PRE-DEFINIDO de configuração de rede")
    # print("Opção 3: Visualizar configuração atual para ser aplicada")
    # print("Opção 4: Aplicar atuais configurações na Interface eth0")
    # print("Opção 5: Resetar configurações da interface eth0")
    # print("Opção 6: Mostrar configurações da interface eth0")
    # print("Opção 7: Testar downloads para avaliar throughput disponível")
    # print("Opção 0: Sair do programa")
    # print("========================================================")


    print("========================================================")
    print("Option 1: Configure Network Conditions MANUALLY")
    print("Option 2: Use PRE-DEFINED network configuration template")
    print("Option 3: View current configuration to be applied")
    print("Option 4: Apply current settings to eth0 interface")
    print("Option 5: Reset eth0 interface settings")
    print("Option 6: Show eth0 interface settings")
    print("Option 7: Test downloads to evaluate available throughput")
    print("Option 0: Exit the program")
    print("========================================================")

    opcao = input("Select option: ")
    if opcao == "1":
        print("Selecione a opção de configuração de Tráfego:")
        throughput = input("Throughput desejado [100mbit, 500kbit]:")
        latency = input("Latência desejada [50, 100, 150]:")
        loss = input("Percentual de Packet loss desejado [0.3, 1, 2, 4]:")
        print("\n\n")

    elif opcao == "2":
        print("Showing predefinitions of network model configuration:")
        for idx, (lat, thr, los) in enumerate(conditions, start=1):
            print(f"Condição {idx}  - THROUGHPUT: {thr} - LATENCY: {lat}ms and PACKET LOSS: {los}%")

        try:
            condition_selected = int(input("Select the number of condition: "))
            if 1 <= condition_selected <= len(conditions):
                # Obter os valores de acordo com a condição escolhida
                latency, throughput, loss = conditions[condition_selected - 1]
                set_network_conditions("eth0",latency, throughput,loss)
                print("\n\nConfigurações aplicadas com sucesso.")
            else:
                print("Número fora do intervalo. Tente novamente com uma condição válida.")
        except ValueError:
            print("Entrada inválida. Por favor, insira um número.")
        print("\n\n")

    elif opcao == "3":
        print("Mostrando a atual configuração que será aplicada:")
        print("========================================================")
        print(f"Latência de {latency}, Throughput de {throughput} e Packet loss de {loss}% \n")
        print("\n\n")

    elif opcao == "4":
        print("Aplicando configurações...")
        print("========================================================")
        set_network_conditions("eth0", latency, throughput, loss)
        print("\n\n")

    elif opcao == "5":
        print("Resetando configurações da interface eth0")
        print("========================================================")
        reset_network_conditions(default_interface)
        print("\n\n")

    elif opcao == "6":
        print("Mostrando a atual configuração vinda da interface pelo traffic control:")
        print("========================================================")
        show_network_conditions(default_interface)
        print("\n\n")

    if opcao == "7":
        print("Select download traffic for test:")
        for idx, (siz, desc, cmd) in enumerate(download, start=1):
            print(f"Condição {idx} - FILE SIZE: {siz}MB via comando: {cmd}")

        try:
            condition_selected = int(input("Select the number of condition: "))
            if 1 <= condition_selected <= len(download):
                # Obter os valores de acordo com a condição escolhida
                size, desc, cmd = download[condition_selected - 1]

                # Executar o comando selecionado
                print(f"Executando download de {desc}...")
                os.system(cmd)

                print("\n\nConfigurações aplicadas com sucesso.")
            else:
                print("Número fora do intervalo. Tente novamente com uma condição válida.")
        except ValueError:
            print("Entrada inválida. Por favor, insira um número.")
        print("\n\n")
    elif opcao == "0":
        print("Saindo do programa")
        print("\n\n")
        break
        

    
