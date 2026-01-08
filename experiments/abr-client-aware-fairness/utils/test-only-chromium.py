from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

chrome_binary_path = '/usr/bin/chromium'
chromedriver_path = "/usr/bin/chromedriver"

# Configurar as opções do navegador
chrome_options = Options()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--headless")  # Adicione esta linha para rodar em modo headless
# chrome_options.add_argument("--remote-debugging-port=9222")  # Adicione esta linha
chrome_options.add_argument("--disable-gpu")  # Adicione esta linha

chrome_options.binary_location = chrome_binary_path

# Inicializar o WebDriver
service = Service(chromedriver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)

# Abrir uma página de teste
driver.get("http://172.25.0.3")
# driver.get("http://www.google.com")
print("Título da página:", driver.title)

# Manter o navegador aberto por um tempo
import time
time.sleep(5)

# Fechar o navegador
driver.quit()
