import os
from dotenv import load_dotenv

load_dotenv()

# Configurações do Telegram
API_KEY = os.getenv("API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
print(TELEGRAM_TOKEN)
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # ex: https://abcd1234.ngrok.io/telegram/webhook
print(WEBHOOK_URL)
BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
print(BASE_URL)
