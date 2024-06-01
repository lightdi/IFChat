# ifchat
 Chat usando IA baseado no regimento didático do IFPB

Preferencialmente crie um ambiente virtual do Python
->python3 -m venv nome_do_ambiente


*Rquisitos*

modulos a instalar

pip install streamlit

pip install langchain

pip install langchain-community

pip install langchain-text-splitters

pip install langchain-chroma

pip install -U google-generativeai

pip install PyPDF2

pip install pypdf

pip install gpt4all

pip install langchainhub

pip install --upgrade --quiet  llama-cpp-python ou pip install llama-cpp-python --verbose
    #Caso queira usar com GPU
    https://python.langchain.com/v0.2/docs/integrations/llms/llamacpp/ 
    https://medium.com/@piyushbatra1999/installing-llama-cpp-python-with-nvidia-gpu-acceleration-on-windows-a-short-guide-0dfac475002d
    Precisa instalar as ferramentas do C++ se estiver no windows https://visualstudio.microsoft.com/pt-br/visual-cpp-build-tools/
    

*Modelos*

Os modelos utilizados podem ser baixados no https://huggingface.co/
São eles:
    all-MiniLM-L6-v2-f16.gguf
    llama-2-7b-chat.Q4_K_M.gguf

*Executando*

streamlit run ifchatOnline.py

ou

streamlit run ifchatOffline.py