# telegram_auto.py
import atexit
import sys
import inspect
from pathlib import Path
import requests

BOT_TOKEN = "8947052047:AAFUxRLOa9w5E8OFx8fx-hVl76HKQYYTYtU"
CHAT_ID = "260840397"


def send_telegram_message(text: str) -> None:
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": text}
    try:
        requests.post(url, data=data, timeout=10)
    except Exception:
        # Falls Telegram nicht erreichbar ist, ignorieren
        pass

# aktuelle Datei ermitteln
caller_file = inspect.stack()[1].filename
filename = Path(caller_file).name

# Handler für normale Beendigung
def notify_success():
    send_telegram_message(f"✅ Fertig: {filename}")

# Handler für uncaught Exceptions
def notify_error(exc_type, exc_value, exc_traceback):
    send_telegram_message(f"❌ Fehler in: {filename}\n\n{exc_value}")
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

# Registrierung
atexit.register(notify_success)
sys.excepthook = notify_error