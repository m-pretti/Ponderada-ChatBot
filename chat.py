import streamlit as st
import pandas as pd
import os
import time
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# --- CONFIGURAÇÃO INICIAL ---
load_dotenv()
st.set_page_config(
    page_title="Toby AI - Compliance System",
    page_icon="📂",
    layout="wide"
)

# --- ARQUIVOS E DIRETÓRIOS ---
ARQUIVO_CSV = "transacoes_bancarias.csv"
DB_COMPLIANCE = "./db_chroma"
DB_EMAILS = "./db_emails"

# --- CACHE DE RECURSOS (Para não recarregar a cada clique) ---
@st.cache_resource
def get_llm():
    return ChatGroq(model_name="llama-3.1-8b-instant")

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_vectorstore(tipo):
    embeddings = get_embeddings()
    diretorio = DB_COMPLIANCE if tipo == "rules" else DB_EMAILS
    if not os.path.exists(diretorio):
        return None
    return Chroma(persist_directory=diretorio, embedding_function=embeddings)

# --- CSS CUSTOMIZADO (Estilo Dunder Mifflin) ---
st.markdown("""
    <style>
    .stApp { background-color: #fffffff; }
    .status-box { padding: 15px; border-radius: 10px; margin-bottom: 10px; border: 1px solid #ddd; }
    .status-risk { background-color: #ffe6e6; border-left: 5px solid #ff4b4b; color: #8a1f1f;}
    .status-ok { background-color: #e6fffa; border-left: 5px solid #00cc99; color: #1f6b58;}
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR (MENU) ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/9/9c/Dunder_Mifflin%2C_Inc.svg", width=200)
    st.title("Sistema de Compliance")
    st.markdown("**Usuário:** Toby Flenderson")
    
    modo = st.radio("Selecione o Módulo:", 
        ["💬 Consultor de RH", "🕵️‍♂️ Investigação Forense", "📊 Auditoria Financeira"])
    
    st.markdown("---")
    if st.button("Limpar Histórico de Chat"):
        st.session_state.messages = []
        st.rerun()

# --- FUNÇÃO: AUDITORIA INTELIGENTE ---
def executar_auditoria(df_filtrado):
    llm = get_llm()
    v_rules = get_vectorstore("rules")
    v_emails = get_vectorstore("emails")
    
    if not v_rules or not v_emails:
        st.error("Bancos de dados não encontrados. Rode etapa 1 e 2 primeiro.")
        return

    progress_bar = st.progress(0)
    total = len(df_filtrado)
    
    for index, row in df_filtrado.iterrows():
        # Atualiza barra
        progress = (index + 1) / total if total > 0 else 0 # (index relativo seria melhor, mas para demo serve)
        # Ajuste simples da barra apenas visual
        progress_bar.progress(min(int(((list(df_filtrado.index).index(index) + 1) / total) * 100), 100))

        transacao_str = f"Func: {row['funcionario']} | Item: {row['descricao']} | Valor: ${row['valor']} | Cat: {row['categoria']}"
        
        # 1. Checar Regras
        docs_regras = v_rules.similarity_search(f"regras {row['categoria']} limite valor", k=2)
        ctx_regras = "\n".join([d.page_content for d in docs_regras])
        prompt_r = ChatPromptTemplate.from_template("Analise se viola regras. Transacao: {t}. Regras: {c}. Se violar, comece com REPROVADO. Se não, APROVADO. Seja breve.")
        res_regras = (prompt_r | llm).invoke({"t": transacao_str, "c": ctx_regras}).content

        # 2. Checar Fraude
        docs_emails = v_emails.similarity_search(f"{row['funcionario']} {row['descricao']} esquema fraude", k=3)
        ctx_emails = "\n".join([d.page_content for d in docs_emails])
        prompt_f = ChatPromptTemplate.from_template("Analise risco de fraude cruzando com emails. Transacao: {t}. Emails: {c}. Se houver indicio forte, comece com ALTO RISCO. Se não, BAIXO RISCO. Seja breve.")
        res_fraude = (prompt_f | llm).invoke({"t": transacao_str, "c": ctx_emails}).content

        # 3. Exibir Card
        with st.expander(f"💰 {row['id_transacao']} | {row['funcionario']} - ${row['valor']}", expanded=("REPROVADO" in res_regras or "ALTO RISCO" in res_fraude)):
            col1, col2 = st.columns(2)
            
            # Card Compliance
            if "REPROVADO" in res_regras:
                col1.markdown(f'<div class="status-box status-risk"><b>📜 Compliance:</b> {res_regras}</div>', unsafe_allow_html=True)
            else:
                col1.markdown(f'<div class="status-box status-ok"><b>📜 Compliance:</b> Aprovado</div>', unsafe_allow_html=True)
            
            # Card Fraude
            if "ALTO RISCO" in res_fraude:
                col2.markdown(f'<div class="status-box status-risk"><b>🚨 Investigação:</b> {res_fraude}</div>', unsafe_allow_html=True)
            else:
                col2.markdown(f'<div class="status-box status-ok"><b>🚨 Investigação:</b> Sem indícios</div>', unsafe_allow_html=True)

# --- LÓGICA DAS PÁGINAS ---

# 1. MÓDULO DE CHAT (RH ou Investigação)
if modo in ["💬 Consultor de RH", "🕵️‍♂️ Investigação Forense"]:
    st.header(modo)
    st.caption("Converse com a IA para tirar dúvidas ou investigar.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Digite sua pergunta..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Seleciona o DB correto
        tipo_db = "rules" if modo == "💬 Consultor de RH" else "emails"
        vectorstore = get_vectorstore(tipo_db)
        
        if vectorstore:
            with st.spinner("Processando..."):
                docs = vectorstore.similarity_search(prompt, k=4)
                contexto = "\n\n".join([d.page_content for d in docs])
                
                sys_prompt = "Você é o Toby do RH. Responda com base no contexto." if tipo_db == "rules" else "Você é um Detetive. Responda com base nos e-mails."
                
                template = ChatPromptTemplate.from_messages([("system", sys_prompt), ("human", "Contexto: {c}\nPergunta: {p}")])
                chain = template | get_llm()
                resposta = chain.invoke({"c": contexto, "p": prompt})
                
                with st.chat_message("assistant"):
                    st.markdown(resposta.content)
                st.session_state.messages.append({"role": "assistant", "content": resposta.content})

# 2. MÓDULO DE AUDITORIA (O Principal)
elif modo == "📊 Auditoria Financeira":
    st.header("Painel de Controle de Fraudes")
    st.markdown("Cruza transações bancárias com regras de compliance e dumps de e-mail.")

    if os.path.exists(ARQUIVO_CSV):
        df = pd.read_csv(ARQUIVO_CSV)
        
        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        col_kpi1.metric("Total de Transações", len(df))
        col_kpi2.metric("Volume Financeiro", f"${df['valor'].sum():,.2f}")
        
        st.markdown("### Configurar Auditoria")
        
        filtro_usuario = st.selectbox("Filtrar Funcionário:", ["Todos", "Michael Scott", "Kevin Malone", "Ryan Howard"])
        
        if st.button("🚀 INICIAR VARREDURA DO SISTEMA"):
            st.write("Iniciando Agentes de IA...")
            
            # Lógica de Filtro para Demo
            if filtro_usuario == "Todos":
                df_analise = df.head(10) # Trava de segurança
            else:
                df_analise = df[df['funcionario'] == filtro_usuario]
                # Se for Michael, pega aquelas especificas que vimos
                if filtro_usuario == "Michael Scott":
                     df_analise = df_analise.head(15) 

            col_kpi3.metric("Transações em Análise", len(df_analise))
            executar_auditoria(df_analise)
            st.success("Varredura Completa!")
            
    else:
        st.error("Arquivo CSV não encontrado.")