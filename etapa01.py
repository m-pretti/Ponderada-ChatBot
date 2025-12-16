import os
from dotenv import load_dotenv

# Importações LangChain + Groq + HuggingFace
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_groq import ChatGroq  
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Carrega variáveis
load_dotenv()

# Verifica se a chave da Groq existe
if not os.getenv("GROQ_API_KEY"):
    print("❌ Erro: Adicione a GROQ_API_KEY no seu arquivo .env")
    exit()

def carregar_e_indexar():
    print("📂 Toby está lendo 'politica_compliance.txt'...")
    
    # 1. Carregar o arquivo que você subiu
    if not os.path.exists("politica_compliance.txt"):
        print("❌ Erro: O arquivo 'politica_compliance.txt' não foi encontrado na pasta!")
        exit()
        
    loader = TextLoader("politica_compliance.txt", encoding="utf-8")
    docs = loader.load()
    
    # 2. Dividir (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, 
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""] # Tenta quebrar por parágrafos primeiro
    )
    splits = text_splitter.split_documents(docs)
    
    print(f"📄 Documento dividido em {len(splits)} pedaços.")

    # 3. Vetorizar 
    print("🧠 Criando conexões neurais (Embeddings)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./db_chroma"
    )
    return vectorstore

def configurar_chat(vectorstore):
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    template = """
    Você é o Toby Flenderson, do RH da Dunder Mifflin.
    Responda à pergunta do funcionário usando APENAS o contexto abaixo.
    
    Se a resposta estiver no texto, cite a seção ou regra específica.
    Se não estiver, diga que não sabe e mande falar com o Michael.
    Mantenha um tom profissional, levemente desanimado e burocrático.
    
    Contexto:
    {context}
    
    Pergunta: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

if __name__ == "__main__":
    # Verifica se já processou antes para ganhar tempo
    if os.path.exists("./db_chroma"):
        print("💾 Carregando banco de dados existente...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory="./db_chroma", embedding_function=embeddings)
    else:
        vectorstore = carregar_e_indexar()

    bot = configurar_chat(vectorstore)

    print("\n--- 👔 CHATBOT TOBY (Versão Grátis/Groq) ---")
    print("Toby: O que vocês querem agora? Estou tentando trabalhar...")
    
    while True:
        pergunta = input("\nVocê: ")
        if pergunta.lower() in ["sair", "tchau"]:
            break
            
        print("Toby: *suspiro* Deixe-me verificar...")
        resposta = bot.invoke(pergunta)
        print(f"Toby: {resposta}")