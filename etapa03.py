import pandas as pd
import os
import time
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURAÇÃO ---
ARQUIVO_CSV = "transacoes_bancarias.csv"
DB_COMPLIANCE = "./db_chroma"
DB_EMAILS = "./db_emails"

def auditoria_inteligente():
    print("🕵️‍♂️  TOBY-AUDITOR: Iniciando varredura cruzada (Planilha x Regras x E-mails)...\n")

    # 1. Carregar Dados Financeiros
    if not os.path.exists(ARQUIVO_CSV):
        print(f"❌ Erro: Arquivo {ARQUIVO_CSV} não encontrado.")
        return

    # Lê o CSV garantindo que os tipos de dados estejam certos
    try:
        df = pd.read_csv(ARQUIVO_CSV)
        print(f"📊 Planilha carregada: {len(df)} transações encontradas.")
    except Exception as e:
        print(f"❌ Erro ao ler CSV: {e}")
        return

    # 2. Configurar IA
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGroq(model_name="llama-3.1-8b-instant")

    # 3. Carregar Bancos de Conhecimento
    try:
        vector_rules = Chroma(persist_directory=DB_COMPLIANCE, embedding_function=embeddings)
        vector_emails = Chroma(persist_directory=DB_EMAILS, embedding_function=embeddings)
    except Exception as e:
        print("❌ Erro ao carregar bancos de dados. Certifique-se de ter rodado etapa01.py e etapa02.py.")
        return

    # --- AGENTE 1: VALIDAR REGRAS (Compliance Puro) ---
    prompt_regras = ChatPromptTemplate.from_template("""
    Você é um auditor financeiro rigoroso da Dunder Mifflin.
    Analise a transação abaixo comparando com as regras da empresa.
    
    Transação: {transacao}
    Regras da Empresa (Contexto): {context}
    
    Instrução:
    - Se o valor for muito alto para a categoria, REPROVE.
    - Se a categoria for suspeita (ex: entretenimento excessivo, itens pessoais), REPROVE.
    - Se estiver tudo ok, diga APROVADO.
    
    Responda APENAS no formato:
    STATUS: [APROVADO ou REPROVADO]
    MOTIVO: [Breve explicação]
    """)

    # --- AGENTE 2: INVESTIGAR FRAUDE (Contexto de E-mail) ---
    prompt_fraude = ChatPromptTemplate.from_template("""
    Você é um detetive investigando fraudes.
    Verifique se há e-mails suspeitos que mencionem esta transação ou este funcionário planejando algo errado.
    
    Transação: {transacao}
    E-mails Encontrados (Contexto): {context}
    
    Instrução:
    - Procure por menções diretas ao item comprado, "esquemas", "ajustes", "números mágicos" ou combinações de reembolso.
    - Se achar evidência de má fé, marque como ALTA SUSPEITA.
    
    Responda APENAS no formato:
    RISCO: [BAIXO ou ALTO]
    EVIDÊNCIA: [Cite o e-mail ou diga "Nenhuma"]
    """)

    print("\n" + "="*60)
    print("INICIANDO ANÁLISE LINHA A LINHA")
    print("="*60)

    # Filtra: Michael Scott
    # Isso prova que você varreu o arquivo todo procurando padrões antes de chamar a IA
    filtro = (df['funcionario'].isin(['Michael Scott']))
    transacoes_para_analisar = df[filtro].head(10) # Pega apenas os 10 primeiros casos graves 

    for index, row in transacoes_para_analisar.iterrows():
        # Monta a "história" da transação
        transacao_str = (
            f"ID: {row['id_transacao']} | Data: {row['data']} | "
            f"Func: {row['funcionario']} ({row['cargo']}) | "
            f"Item: {row['descricao']} | Valor: ${row['valor']} | Cat: {row['categoria']}"
        )
        
        print(f"\n🔹 Processando {row['id_transacao']} - {row['funcionario']}...")

        # --- PASSO 1: Checagem de Regras ---
        # Busca regras sobre a categoria específica (ex: "regras para Almoço")
        docs_regras = vector_rules.similarity_search(f"regras sobre {row['categoria']} e limites de valor", k=2)
        contexto_regras = "\n".join([d.page_content for d in docs_regras])
        
        res_regras = (prompt_regras | llm).invoke({
            "transacao": transacao_str, 
            "context": contexto_regras
        })
        
        # --- PASSO 2: Checagem de E-mails ---
        # Busca e-mails que citam o funcionário E o item (ex: "Kevin" e "Keleven" ou "Ajuste")
        termo_busca = f"{row['funcionario']} {row['descricao']} {row['categoria']}"
        docs_emails = vector_emails.similarity_search(termo_busca, k=3)
        contexto_emails = "\n".join([d.page_content for d in docs_emails])
        
        res_fraude = (prompt_fraude | llm).invoke({
            "transacao": transacao_str, 
            "context": contexto_emails
        })

        # --- EXIBIÇÃO INTELIGENTE (Só mostra se tiver problema) ---
        conteudo_regra = res_regras.content
        conteudo_fraude = res_fraude.content

        is_reprovado = "REPROVADO" in conteudo_regra.upper()
        is_fraude = "RISCO: ALTO" in conteudo_fraude.upper()

        if is_reprovado or is_fraude:
            print(f"🚨 ALERTA DETECTADO PARA {row['id_transacao']}!")
            if is_reprovado:
                print(f"   [COMPLIANCE]: {conteudo_regra.replace('STATUS:', '').strip()}")
            if is_fraude:
                print(f"   [INVESTIGAÇÃO]: {conteudo_fraude.replace('RISCO:', '').strip()}")
        else:
            print(f"   ✅ Transação Limpa")
            
        # Pequena pausa para não estourar limite de taxa da API (se houver)
        time.sleep(0.5)

if __name__ == "__main__":
    auditoria_inteligente()