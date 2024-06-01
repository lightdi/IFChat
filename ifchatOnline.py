import streamlit as st

#Imports to LLM
import os
import PyPDF2
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



class Gemini_model:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Gemini_model, cls).__new__(cls)
            cls.chat = None
        return cls._instance
    
    def extract_pdf_pages(self, pathname: str) -> list[str]: 
        parts = [f"--- START OF PDF {pathname} ---"] 
 
        with open(pathname, "rb") as file: 
            reader = PyPDF2.PdfReader(file) 
            for page_num in range(len(reader.pages)): 
                page = reader.pages[page_num] 
                text = page.extract_text() 
                parts.append(f"--- PAGE {page_num + 1} ---") 
                parts.append(text) 
             
        return parts
    
    def criar_modelo(self):


        if self.chat is not None :
            return self.chat

        GOOGLE_API_KEY = "AIzaSyAP79zhxjEVyYrgSj7BCaFUKt3j2N7UYQw"

        genai.configure(api_key=GOOGLE_API_KEY)

        model = genai.GenerativeModel("gemini-pro")

         # Carregando o PDF
        loader = PyPDFLoader(os.path.join(os.getcwd(), "regint.pdf"))
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = loader.load_and_split(text_splitter)


        # Set up the model 
        generation_config = { 
            "temperature": 1, 
            "top_p": 0.95, 
            "top_k": 64, 
            "max_output_tokens": 8192, 
        } 

        safety_settings = [ 
            { 
                "category": "HARM_CATEGORY_HARASSMENT", 
                "threshold": "BLOCK_MEDIUM_AND_ABOVE" 
            }, 
            { 
                "category": "HARM_CATEGORY_HATE_SPEECH", 
                "threshold": "BLOCK_MEDIUM_AND_ABOVE" 
            }, 
            { 
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", 
                "threshold": "BLOCK_MEDIUM_AND_ABOVE" 
            }, 
            { 
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT", 
                "threshold": "BLOCK_MEDIUM_AND_ABOVE" 
            }, 
        ]            

        history=[ 
            { 
                "role": "user", 
                "parts": self.extract_pdf_pages(os.path.join(os.getcwd(), "regint.pdf"))
            }, 
            { 
                "role": "user", 
                "parts": "Você é um assistente para tarefas de resposta a perguntas sobre o IFPB. "
                "Use as mensagem anterior para responder as perguntas. "
                "Se você não souber a resposta, apenas diga que não sabe. "
                "Use no máximo três frases e mantenha a resposta concisa"
            }
        ] 

        self.chat =  model.start_chat( history = history) 

        return self.chat



def main():

    st.set_page_config("IFChat")

    st.header("Chat com o Regimento didático do IFPB")

       # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    pergunta = st.chat_input("Qual sua dúvida?")

    if pergunta:
        with st.spinner('Pensando... por favor aguarde'):
            gemini_model = Gemini_model()
            chat = gemini_model.criar_modelo()
            #Acessa o modelo e retorna uma resposta 
            resposta = chat.send_message (pergunta)

        # Display user message in chat message container
            with st.chat_message("Você"):
                st.markdown(pergunta)
            # Add user message to chat history
            st.session_state.messages.append({"role": "Você", "content": pergunta})

    
            response = f"Echo: {resposta.text}"
            # Display assistant response in chat message container
            with st.chat_message("Assistente"):
                st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "Assistente", "content": response})
    

if __name__ == "__main__":
    main()