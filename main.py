from contextlib import asynccontextmanager
from fastapi import FastAPI
from telegram.telegram import router as telegram_router
from rag.upload import router as upload_router
import httpx 
from core.config import TELEGRAM_TOKEN, WEBHOOK_URL, BASE_URL

@asynccontextmanager
async def start_up( app: FastAPI):
    try:
        #Definir o webhook a cada inicialização
        if not TELEGRAM_TOKEN or not WEBHOOK_URL:
            print("Variáveis TELEGRAM_TOKEN ou WEBHOOK_URL não configuradas.")
            yield
            return
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BASE_URL}/setWebhook",
                data={"url": WEBHOOK_URL}
            )
        if response.status_code == 200 and response.json().get("ok"):
            print(f"✅ Webhook configurado com sucesso: {WEBHOOK_URL}")
        else:
            print(f"❌ Erro ao configurar webhook: {response.text}")

        yield
    finally:
        pass

app = FastAPI(title="IFChat - Assistente do IFSousa", lifespan=start_up)

#Registrando Módulos
app.include_router(telegram_router, prefix='/telegram', tags=["Telegram"])
#app.include_router(upload_router, prefix='/upload', tags=["Upload"])



@app.get("/")
def root():
    return {"message": "IFChat API online 🚀"}