from pathlib import Path
import shutil

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from config import (
    DOCUMENTS_DIR,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def carregar_documentos():
    documentos = []

    arquivos_pdf = list(DOCUMENTS_DIR.glob("*.pdf"))

    if not arquivos_pdf:
        raise FileNotFoundError(
            f"Nenhum PDF encontrado em: {DOCUMENTS_DIR}"
        )

    print(f"Encontrados {len(arquivos_pdf)} arquivo(s) PDF.")

    for arquivo in arquivos_pdf:
        print(f"Carregando: {arquivo.name}")

        loader = PyPDFLoader(str(arquivo))
        paginas = loader.load()

        for pagina in paginas:
            pagina.metadata["arquivo"] = arquivo.name

        documentos.extend(paginas)

    print(f"Total de páginas carregadas: {len(documentos)}")

    return documentos


def dividir_documentos(documentos):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documentos)

    print(f"Total de chunks criados: {len(chunks)}")

    return chunks

def criar_embeddings():
    print("Configurando embeddings do Gemini...")

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL
    )

def criar_banco_vetorial(chunks, embeddings):
    if CHROMA_DIR.exists():
        print("Removendo banco vetorial anterior...")
        shutil.rmtree(CHROMA_DIR)

    print("Criando banco vetorial Chroma...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )

    return vectorstore


def main():
    print("=" * 60)
    print("INGESTÃO DE DOCUMENTOS")
    print("=" * 60)

    documentos = carregar_documentos()

    chunks = dividir_documentos(documentos)

    embeddings = criar_embeddings()

    criar_banco_vetorial(chunks, embeddings)

    print()
    print("=" * 60)
    print("INGESTÃO CONCLUÍDA")
    print("=" * 60)
    print(f"Banco vetorial salvo em: {CHROMA_DIR}")


if __name__ == "__main__":
    main()