from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import logging
import time

# == Configuração do Selenium
chrome_options = Options()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--headless")  
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--disable-software-rasterizer")
chrome_options.add_argument("--enable-logging")
chrome_options.add_argument("--v=1")

# == Caminho do Chrome e do WebDriver
chrome_binary_path = "/usr/bin/chromium"
chromedriver_path = "/usr/bin/chromedriver"
chrome_options.binary_location = chrome_binary_path

# == Inicializa o WebDriver
service = Service(chromedriver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)

logging.basicConfig(level=logging.INFO)



# == Access Player Page
try:
    player_url = "http://172.25.0.3:80/dash-player.html"
    logging.info("Opening URL Player!")
    driver.get(player_url)
    logging.info(f"Page loaded successfully! {driver.title}")

    # == Aguarda o carregamento do Dash.js
    WebDriverWait(driver, 10).until(
        lambda d: d.execute_script("return typeof dashjs !== 'undefined';")
        #  lambda d: d.execute_script("return dashjs.MediaPlayer().getInstance().getSource() !== null;")
    )
    logging.info(f"Dash.js loaded sucessfully!")

    # == Aguarda o elemento de vídeo estar presente
    video_element = WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.ID, "videoPlayer"))
    )
    logging.info(f"<video> element detected!")


    # == Wait for MPD video be loaded on player
    WebDriverWait(driver, 15).until(
        lambda d: d.execute_script("return window.player.getSource() !== null;")
    )
    mpd_url = driver.execute_script("return window.player.getSource();")
    logging.info(f"MPD carregado: {mpd_url}")


    # == Start Reprodution
    logging.info("Iniciando a reprodução...")
    video_element.click()
    time.sleep(3)  # Aguarde alguns segundos para o player carregar e começar a tocar

    # == Collect DASH metrics
    average_throughput = driver.execute_script("return window.player.getAverageThroughput('video');")
    buffer_level = driver.execute_script("return window.player.getDashMetrics().getCurrentBufferLevel('video');")
    current_quality = driver.execute_script("return window.player.getQualityFor('video');")

    logging.info(f"Taxa de transferência média: {average_throughput} Kbps")
    logging.info(f"Nível de buffer: {buffer_level} segundos")
    logging.info(f"Qualidade atual: {current_quality}")

    print(f"MPD carregado: {mpd_url}")
    print(f"Taxa de transferência média: {average_throughput} Kbps")
    print(f"Nível de buffer: {buffer_level} segundos")
    print(f"Qualidade atual: {current_quality}")


except Exception as e:
    # player_initialized = str(e)
    logging.error(f"Erro durante a execução: {e}")

# print("Player inicializado:", player_initialized)

finally:
    driver.quit()
