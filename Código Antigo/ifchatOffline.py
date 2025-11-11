import streamlit as st

#Imports to LLM
import os
from langchain_community.llms import LlamaCpp

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate



from langchain import hub
from langchain_core.runnables import RunnablePassthrough, RunnablePick


#Criando uma classe singleton
# Classe Singleton para o Chain
class Llama_model:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Llama_model, cls).__new__(cls)
            cls._instance.chain = None
            cls._instance.vectorstore = None
        return cls._instance

    def criar_modelo(self):
        print("Criando o modelo")
        if self.chain is not None and self.vectorstore is not None:
            return self.chain, self.vectorstore

        # Carregando o modelo do Llama
        n_gpu_layers = -1
        n_batch = 2048
        llm = LlamaCpp(
            model_path=os.path.join(os.getcwd(), "llama-2-7b-chat.Q4_K_M.gguf"),
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            n_ctx=2048,
            f16_kv=True,
            verbose=True,
        )

        # Carregando o PDF
        loader = PyPDFLoader(os.path.join(os.getcwd(), "regint.pdf"))
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = loader.load_and_split(text_splitter)

        # Criando os Embeddings
        model_name = "all-MiniLM-L6-v2.gguf2.f16.gguf"
        gpt4all_kwargs = {'allow_download': 'True'}
        embeddings = GPT4AllEmbeddings(
            model_name=model_name,
            gpt4all_kwargs=gpt4all_kwargs
        )

        self.vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)

        # Criando o prompt RAG
        rag_prompt = hub.pull("rlm/rag-prompt")
        rag_prompt.messages[0].prompt.template = (
            "Você é um assistente para tarefas de resposta a perguntas. "
            "Use as seguintes partes do contexto recuperado para responder à pergunta. "
            "Se você não souber a resposta, apenas diga que não sabe. "
            "Use no máximo três frases e mantenha a resposta concisa.\n"
            "Pergunta: {question} \nContexto: {context} \nResposta:"
        )

        # Criando o objeto de conversação
        self.chain = (
            RunnablePassthrough.assign(context=RunnablePick("context") | format_docs)
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        print("modelo criado")
        return self.chain, self.vectorstore

    def get_model(self):
        return self.criar_modelo()
#Juntando os documetnos em uma só variável
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



def pergunta_usuario(pergunta):
    llama_model = Llama_model()
    chain, vectorstore = llama_model.get_model()
    question = pergunta
    #Realiza a pesquisa no vetor que tem os documetnos armazendados 
    docs = vectorstore.similarity_search(question)
    #envia a pergunta para o objeto de conversação
    return chain.invoke({"context": docs, "question": question})



#Iniciando a página
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
            #Acessa o modelo e retorna uma resposta 
            resposta = pergunta_usuario(pergunta)


            # Display user message in chat message container
            with st.chat_message("Você"):
                st.markdown(pergunta)
            # Add user message to chat history
            st.session_state.messages.append({"role": "Você", "content": pergunta})

    
            response = f"Echo: {resposta}"
            # Display assistant response in chat message container
            with st.chat_message("Assistente"):
                st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "Assistente", "content": resposta})

if __name__ == "__main__":
    main()