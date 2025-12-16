import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Carrega as chaves do .env
load_dotenv()

# --- CONFIGURAÇÃO ---
ARQUIVO_EMAILS = "emails.txt"  # Nome exato do seu arquivo enviado
DIRETORIO_DB = "./db_emails"   # Pasta separada para a investigação

def realizar_investigacao():
    print("🕵️‍♂️  Iniciando Protocolo de Investigação 'Toby-Holmes'...")

    # 1. Carregar e Processar o Dump de E-mails
    if not os.path.exists(ARQUIVO_EMAILS):
        raise FileNotFoundError(f"O arquivo {ARQUIVO_EMAILS} não foi encontrado!")

    print(f"📂 Lendo evidências em '{ARQUIVO_EMAILS}'...")
    with open(ARQUIVO_EMAILS, "r", encoding="utf-8") as f:
        texto_emails = f.read()

 
    text_splitter = CharacterTextSplitter(
        separator="-------------------------------------------------------------------------------",
        chunk_size=1500,  
        chunk_overlap=0
    )
    documentos = text_splitter.create_documents([texto_emails])
    print(f"📄 E-mails processados: {len(documentos)}")

    # 2. Criar Embeddings 
    print("🧠 Criando conexões neurais (Indexando e-mails)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 3. Armazenar no Banco Vetorial 
    if os.path.exists(DIRETORIO_DB):
        print("💾 Carregando banco de e-mails existente...")
        vectorstore = Chroma(persist_directory=DIRETORIO_DB, embedding_function=embeddings)
    else:
        print("💾 Criando novo banco de dados forense...")
        vectorstore = Chroma.from_documents(
            documents=documentos,
            embedding=embeddings,
            persist_directory=DIRETORIO_DB
        )

    # 4. Configurar a LLM
    chat_model = ChatGroq(model_name="llama-3.1-8b-instant")

    # Prompt focado em investigação e citação de provas
    template_investigacao = """
    Você é um investigador forense analisando e-mails corporativos da Dunder Mifflin.
    
    Contexto (E-mails recuperados):
    {context}

    Pergunta da Investigação: {question}

    Instruções:
    1. Responda se a suspeita é verdadeira ou falsa baseada APENAS no texto.
    2. Se encontrar provas, cite: QUEM enviou, PARA QUEM e o ASSUNTO.
    3. Seja direto e profissional, como um relatório policial.
    """

    prompt = ChatPromptTemplate.from_template(template_investigacao)
    chain = prompt | chat_model

    # 5. A Investigação
    pergunta_investigacao = "O Michael Scott está conspirando contra o Toby Flenderson? Existem planos de demissão, armadilhas ou operações secretas mencionadas?"

    print(f"\n🔍 Buscando respostas para: '{pergunta_investigacao}'")
    
    # Busca os 4 e-mails mais suspeitos
    docs_relacionados = vectorstore.similarity_search(pergunta_investigacao, k=4)
    contexto = "\n\n".join([doc.page_content for doc in docs_relacionados])

    resposta = chain.invoke({"context": contexto, "question": pergunta_investigacao})

    print("\n" + "="*40)
    print("📋 RELATÓRIO FINAL DE INVESTIGAÇÃO")
    print("="*40)
    print(resposta.content)

if __name__ == "__main__":
    realizar_investigacao()