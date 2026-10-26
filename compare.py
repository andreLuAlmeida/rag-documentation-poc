from langchain_core.prompts import ChatPromptTemplate

from rag import (
    criar_llm,
    responder_com_rag,
)


def responder_sem_rag(pergunta):
    llm = criar_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Você é um assistente geral.

Responda à pergunta utilizando apenas seu conhecimento próprio.
Não utilize documentos externos ou mecanismos de recuperação.
""",
            ),
            (
                "human",
                "{pergunta}",
            ),
        ]
    )

    chain = prompt | llm

    resposta = chain.invoke(
        {
            "pergunta": pergunta
        }
    )

    return resposta.content


def main():
    print("=" * 60)
    print("COMPARAÇÃO: SEM RAG X COM RAG")
    print("=" * 60)

    pergunta = input("\nDigite uma pergunta: ").strip()

    if not pergunta:
        print("Pergunta vazia.")
        return

    print()
    print("=" * 60)
    print("1. RESPOSTA SEM RAG")
    print("=" * 60)

    resposta_sem_rag = responder_sem_rag(pergunta)

    print(resposta_sem_rag)

    print()
    print("=" * 60)
    print("2. RESPOSTA COM RAG")
    print("=" * 60)

    resposta_com_rag, documentos = responder_com_rag(pergunta)

    print(resposta_com_rag)

    print()
    print("=" * 60)
    print("DOCUMENTOS RECUPERADOS")
    print("=" * 60)

    for i, doc in enumerate(documentos, start=1):
        arquivo = doc.metadata.get("arquivo", "desconhecido")
        pagina = doc.metadata.get("page", "desconhecida")

        print(
            f"{i}. {arquivo} - página {pagina}"
        )


if __name__ == "__main__":
    main()