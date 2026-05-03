import pystray
from PIL import Image, ImageDraw
import subprocess
import webbrowser
import threading
import time
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Sunucuyu başlat
server = subprocess.Popen(
    [sys.executable, "server_deepgram.py"],
    creationflags=subprocess.CREATE_NO_WINDOW
)

# Tarayıcıyı 3 saniye sonra aç
def open_browser():
    time.sleep(3)
    webbrowser.open("http://localhost:8001")

threading.Thread(target=open_browser, daemon=True).start()

# Tepsi ikonu oluştur
def create_icon():
    img = Image.new("RGB", (64, 64), color=(8, 8, 8))
    d = ImageDraw.Draw(img)
    d.ellipse([8, 8, 56, 56], fill=(200, 169, 106))
    d.polygon([(22, 16), (22, 48), (50, 32)], fill=(8, 8, 8))
    return img

def on_open(icon, item):
    webbrowser.open("http://localhost:8001")

def on_quit(icon, item):
    icon.stop()
    server.terminate()
    os._exit(0)

icon = pystray.Icon(
    "SimultanePro",
    create_icon(),
    "SimultanePro",
    menu=pystray.Menu(
        pystray.MenuItem("Tarayıcıda Aç", on_open),
        pystray.MenuItem("Kapat", on_quit)
    )
)

icon.run()
