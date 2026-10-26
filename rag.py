import os

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    GEMINI_MODEL,
    TOP_K,
)


load_dotenv()


def verificar_api_key():
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY não encontrada. "
            "Crie um arquivo .env com GOOGLE_API_KEY=sua_chave"
        )


def criar_embeddings():
    verificar_api_key()

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL
    )


def carregar_vectorstore():
    embeddings = criar_embeddings()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    return vectorstore


def criar_llm():
    verificar_api_key()

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        max_retries=2,
    )

def recuperar_documentos(vectorstore, pergunta):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K}
    )

    documentos = retriever.invoke(pergunta)

    return documentos


def formatar_contexto(documentos):
    partes = []

    for i, doc in enumerate(documentos, start=1):
        arquivo = doc.metadata.get("arquivo", "desconhecido")
        pagina = doc.metadata.get("page", "desconhecida")

        trecho = f"""
TRECHO {i}
Fonte: {arquivo}
Página: {pagina}

{doc.page_content}
"""

        partes.append(trecho)

    return "\n".join(partes)


def responder_com_rag(pergunta):
    print("1. Carregando banco vetorial...")
    vectorstore = carregar_vectorstore()

    print("2. Criando LLM...")
    llm = criar_llm()

    print("3. Recuperando documentos...")
    documentos = recuperar_documentos(vectorstore, pergunta)

    print(f"4. {len(documentos)} documentos recuperados.")

    contexto = formatar_contexto(documentos)

    print("5. Montando prompt...")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Você é um assistente especializado em responder perguntas
com base exclusivamente em documentação fornecida.

Regras:
1. Utilize apenas o contexto recuperado.
2. Não invente informações que não estejam no contexto.
3. Caso a informação não esteja presente, diga claramente:
   "Não encontrei essa informação nos documentos consultados."
4. Responda de forma objetiva e técnica.
5. Quando possível, indique de qual documento veio a informação.
""",
            ),
            (
                "human",
                """
Contexto recuperado:

{contexto}

Pergunta:

{pergunta}
""",
            ),
        ]
    )

    print("6. Chamando Gemini...")
    chain = prompt | llm

    resposta = chain.invoke(
        {
            "contexto": contexto,
            "pergunta": pergunta,
        }
    )

    print("7. Resposta recebida.")

    conteudo = resposta.content

    if isinstance(conteudo, list):
        textos = []

        for bloco in conteudo:
            if isinstance(bloco, dict) and bloco.get("type") == "text":
                textos.append(bloco.get("text", ""))

        conteudo = "\n".join(textos)

    return conteudo, documentos


def imprimir_fontes(documentos):
    print()
    print("-" * 60)
    print("FONTES RECUPERADAS")
    print("-" * 60)

    for i, doc in enumerate(documentos, start=1):
        arquivo = doc.metadata.get("arquivo", "desconhecido")
        pagina = doc.metadata.get("page", "desconhecida")

        print(f"{i}. Arquivo: {arquivo} | Página: {pagina}")


def main():
    print("=" * 60)
    print("RAG - CONSULTA DE DOCUMENTAÇÃO TÉCNICA")
    print("=" * 60)

    while True:
        print()
        pergunta = input(
            "Digite sua pergunta ou 'sair': "
        ).strip()

        if pergunta.lower() in {"sair", "exit", "quit"}:
            print("Encerrando aplicação.")
            break

        if not pergunta:
            continue

        try:
            resposta, documentos = responder_com_rag(pergunta)

            print()
            print("=" * 60)
            print("RESPOSTA")
            print("=" * 60)
            print(resposta)

            imprimir_fontes(documentos)

        except Exception as erro:
            print()
            print(f"Erro: {erro}")


if __name__ == "__main__":
    main()