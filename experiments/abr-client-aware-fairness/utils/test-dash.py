from selenium import webdriver

options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
driver = webdriver.Chrome(options=options)

# Acessar a página do player
driver.get("http://172.25.0.3:80/dash-player.html")

# Testar se o Dash.js está carregado no contexto do navegador
dashjs_loaded = driver.execute_script("return typeof dashjs !== 'undefined';")
print("Dash.js carregado:", dashjs_loaded)

# Testar se o player conseguiu encontrar o MPD
mpd_loaded = driver.execute_script("""
    var player = dashjs.MediaPlayer().getInstance();
    return player.getSource();
""")
print("MPD carregado:", mpd_loaded)

driver.quit()
