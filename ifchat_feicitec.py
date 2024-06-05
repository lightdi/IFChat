import os
import PyPDF2
import streamlit as st 
import google.generativeai as genai


GOOGLE_API_KEY = "AIzaSyAP79zhxjEVyYrgSj7BCaFUKt3j2N7UYQw"

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-pro")


def extract_pdf(arquivo):
  parts = [f"--- START OF PDF {arquivo} ---"]
  file = open(arquivo, 'rb')
  reader = PyPDF2.PdfReader(file)
  for page_num in range(len(reader.pages)):
    page = reader.pages[page_num]
    text = page.extract_text()
    parts.append(f"--- PAGE {page_num + 1} ---")
    parts.append(text)

  return parts

history=[
            {
                "role": "user",
                "parts": extract_pdf(os.path.join(os.getcwd(), "regint.pdf"))
            },
            {
                "role": "user",
                "parts": "Você é um assistente para tarefas de resposta a perguntas sobre o IFPB. "
                "Use as mensagem anterior para responder as perguntas. "
                "Se você não souber a resposta, apenas diga que não sabe. "
                "Use no máximo três frases e mantenha a resposta concisa"
            }
        ]


chat = model.start_chat(history=history)


def main(): 
    st.set_page_config("IFChat")

    st.header("Chat com o Regimento didático do IFPB")

    pergunta = st.chat_input("Qual sua dúvida?")


    if pergunta: 
        print(pergunta)
        resposta = chat.send_message (pergunta)
        with st.chat_message("Você"):
                st.markdown(pergunta)

        with st.chat_message("Assistente"):
                st.markdown(resposta.text)


if __name__ == "__main__":
   main()