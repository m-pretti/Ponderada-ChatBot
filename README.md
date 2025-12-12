# 👔 Toby AI - Sistema Inteligente de Compliance (Dunder Mifflin)

> *"Por que você é do jeito que é?" - Michael Scott para Toby*

Este projeto é um **Agente de IA para Auditoria e Compliance** desenvolvido para analisar documentos corporativos, e-mails vazados e transações financeiras. Utilizando **RAG (Retrieval-Augmented Generation)** e **Agentes Autônomos**, o sistema cruza dados estruturados e não estruturados para detectar fraudes, violações de política e conspirações no escritório.

---

## 🚀 Funcionalidades

O sistema opera em três camadas de inteligência:

1.  **Consultor de RH (Compliance):** Lê e interpreta políticas internas (`.txt`/PDF) para responder dúvidas sobre regras.
2.  **Investigador Forense:** Analisa dumps de e-mails (`.txt`) para detectar conluio, sentimentos e intenções de fraude.
3.  **Auditor Financeiro (Agente Cruzado):** Lê planilhas bancárias (`.csv`), verifica cada transação contra as regras e cruza com os e-mails para encontrar evidências de má fé (ex: "Smurfing", desvio de verba).

---

## 🛠️ Tecnologias Utilizadas

* **Python 3.12**
* **LangChain** (Orquestração de LLMs e RAG)
* **Groq API** (Modelo Llama 3.1 - Inferência ultra-rápida)
* **ChromaDB** (Banco de Dados Vetorial Local)
* **HuggingFace Embeddings** (`sentence-transformers/all-MiniLM-L6-v2`)
* **Pandas** (Análise de Dados Estruturados)
* **Streamlit** (Interface Web Interativa)

---

## ⚙️ Pré-requisitos e Instalação

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/seu-usuario/seu-repo.git](https://github.com/m-pretti/Ponderada-ChatBot.git)
   cd Ponderada-ChatBot
````

2.  **Crie e ative o ambiente virtual:**

      * **Linux/Mac:**
        ```bash
        python -m venv venv
        source venv/bin/activate
        ```

3.  **Instale as dependências:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure as Chaves de API:**

      * Crie um arquivo `.env` na raiz do projeto.
      * Adicione sua chave da Groq:
        ```env
        GROQ_API_KEY=sua_chave_aqui_gsk_...
        ```

-----

## ▶️ Como Executar o Projeto

O sistema funciona em etapas. É **obrigatório** rodar as etapas 1 e 2 pelo menos uma vez para criar os bancos de dados vetoriais antes de usar a interface.

### 1️⃣ Inicialização dos Bancos de Dados (Backend)

Rode os scripts abaixo para processar os arquivos de texto e criar a memória da IA (pastas `db_chroma` e `db_emails`).

**Etapa 1: Processar Regras de Compliance**

```bash
python etapa01.py
```

*Saída esperada:* O sistema lê `politica_compliance.txt` e cria o banco vetorial de regras.

**Etapa 2: Processar Dump de E-mails**

```bash
python etapa02.py
```

*Saída esperada:* O sistema lê `emails.txt` e cria o banco vetorial de investigação forense.

-----

### 2️⃣ Auditoria via Terminal (Demo Rápida)

Para ver o Agente Auditor trabalhar em tempo real no terminal, filtrando suspeitas do Michael Scott:

```bash
python etapa03.py
```

*O que acontece:* O script lê o `transacoes_bancarias.csv`, cruza com as regras e e-mails, e imprime alertas de **FRAUDE** ou **VIOLAÇÃO** diretamente no console.

-----

### 3️⃣ Interface Visual (Streamlit)

Para acessar o Painel de Controle completo (Dashboard):

```bash
streamlit run app.py
```

O navegador abrirá automaticamente em `http://192.168.100.112:8501`.

**Navegação na Interface:**

  * **Menu Lateral:** Escolha entre "Consultor de RH", "Investigação" ou "Auditoria Financeira".
  * **Modo Auditoria:**
    1.  Selecione "Auditoria Financeira".
    2.  No filtro, escolha **"Michael Scott"**.
    3.  Clique em **"INICIAR VARREDURA"**.
    4.  Veja os cards de alerta aparecerem com detalhes das evidências encontradas.

-----

## 🕵️‍♂️ Exemplos de Detecção

O sistema é capaz de detectar casos complexos como:

  * **Smurfing:** Michael Scott dividindo despesas para evitar aprovação da matriz.
  * **Lavagem de Categoria:** Compra de itens pessoais (Walkie-Talkies) categorizados como "Segurança".
  * **Conluio:** Identificação de e-mails onde funcionários combinam fraudes financeiras.