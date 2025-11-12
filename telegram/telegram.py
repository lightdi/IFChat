from fastapi import APIRouter, Request
import httpx
from core.config import BASE_URL
from rag.rag import query_rag

router = APIRouter()


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
        await send_message(chat_id,  query_rag(text))

    return {"ok": True}