from fastapi import APIRouter, Request
from dotenv import load_dotenv
import httpx, os

router = APIRouter()

load_dotenv()


TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"


# Função para enviar mensagem de volta
async def send_message(chat_id: int, text: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/sendMessage",
            json={"chat_id": chat_id, "text": text}
        )
    


#Rota que recebe as mensagem vindas do Telegram (webhook)
@router.post("/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()

    #Verifica se tem mensagem 
    if "message" in data:
        chat_id = data["message"]["chat"]["id"]
        text = data["message"].get("text", "")

        print(f"Mensagem de {chat_id}: {text}") 

        #Enviar a reposta 
        await send_message(chat_id,  f"Oi! Vai tomar no cú")

    return {"ok": True}