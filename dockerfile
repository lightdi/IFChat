# Etapa base
FROM python:3.11-slim

# Diretório de trabalho
WORKDIR /app

# Copiar dependências
COPY requirements.txt .

# Instalar dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar aplicação
COPY app ./app

# Copiar .env
COPY .env .

# Expor porta
EXPOSE 8100

# Comando de inicialização
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100"]