# RAG para Consulta Inteligente de Documentação Técnica

Prova de conceito desenvolvida para a disciplina de Engenharia de Software da Universidade Federal de Lavras.

## Sobre o projeto

Este projeto demonstra uma arquitetura de Retrieval-Augmented Generation (RAG) para consulta de documentos técnicos.

O sistema recebe documentos PDF, extrai e divide seu conteúdo em pequenos trechos, gera embeddings, armazena esses vetores em um banco vetorial e utiliza busca semântica para localizar informações relevantes para uma pergunta.

Os trechos encontrados são posteriormente enviados como contexto para um modelo Gemini, que gera uma resposta baseada na documentação recuperada.

## Arquitetura
```
Documentos PDF
      │
      ▼
Extração de texto
      │
      ▼
Chunking
      │
      ▼
Gemini Embeddings
      │
      ▼
Chroma Vector Database
      │
      ▼
Pergunta do usuário
      │
      ▼
Embedding da consulta
      │
      ▼
Busca semântica
      │
      ▼
Top-K chunks
      │
      ▼
Prompt + contexto
      │
      ▼
Gemini
      │
      ▼
Resposta + fontes
```

## Tecnologias utilizadas

- Python
- LangChain
- Google Gemini API
- Gemini Embeddings
- Chroma
- PyPDF
- python-dotenv

## Estrutura
```
rag-documentacao/
│
├── documents/
│   ├── Apostila_git.pdf
│   └── Codigo_de_etica_SBC.pdf
│
├── chroma_db/
│
├── config.py
├── ingest.py
├── rag.py
├── compare.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## Fluxo de ingestão

O arquivo `ingest.py`:

1. localiza arquivos PDF;
2. extrai o conteúdo;
3. divide os documentos em chunks;
4. gera embeddings;
5. cria uma base vetorial Chroma;
6. salva os vetores e metadados localmente.

Execute:
```
python ingest.py
```

Na execução utilizada na PoC foram processadas 60 páginas, resultando em 75 chunks.

## Consulta utilizando RAG

Execute:
```
python rag.py
```

Exemplo:
```
Digite sua pergunta: O que é Git?
```

O sistema:

1. gera o embedding da pergunta;
2. pesquisa os chunks semanticamente mais próximos;
3. recupera os quatro melhores resultados;
4. cria um contexto;
5. envia o contexto e a pergunta para o Gemini;
6. exibe a resposta;
7. apresenta as fontes recuperadas.

## Comparação sem RAG x com RAG

Execute:
```
python compare.py
```

Esse programa realiza a mesma pergunta em dois cenários:

### Sem RAG

A pergunta é enviada diretamente para a LLM.
```
Pergunta → Gemini → Resposta
```

### Com RAG

A aplicação recupera primeiro informações dos documentos.
```
Pergunta
   ↓
Busca vetorial
   ↓
Documentos relevantes
   ↓
Gemini
   ↓
Resposta contextualizada
```

Essa comparação ajuda a demonstrar a diferença entre utilizar apenas o conhecimento interno da LLM e utilizar documentos externos como fonte de contexto.

## Instalação

Clone o repositório:
```
git clone URL_DO_REPOSITORIO
cd rag-documentacao
```

Crie um ambiente virtual:
```
python3 -m venv .venv
```

Ative:
```
source .venv/bin/activate
```

Instale as dependências:
```
pip install -r requirements.txt
```

## Configuração da API

Crie um arquivo `.env`:
```
GOOGLE_API_KEY=SUA_CHAVE
```

O arquivo `.env` não deve ser enviado ao GitHub.

## Documentos

Adicione os PDFs desejados ao diretório:
```
documents/
```

Depois execute novamente:
```
python ingest.py
```

para reconstruir a base vetorial.

## Exemplos de perguntas
```
O que é Git?

Para que serve um arquivo .gitignore?

O que é o Código de Ética do Profissional de Informática?

Quais são alguns deveres do profissional de informática?
```

## Principais componentes

### `config.py`

Centraliza parâmetros do sistema, como:

- diretório dos documentos;
- diretório do Chroma;
- modelo de embeddings;
- modelo generativo;
- tamanho dos chunks;
- sobreposição dos chunks;
- quantidade de documentos recuperados.

### `ingest.py`

Implementa o pipeline de ingestão:
```
PDF → texto → chunks → embeddings → Chroma
```

### `rag.py`

Implementa:
```
pergunta → recuperação → contexto → Gemini → resposta
```

### `compare.py`

Permite comparar a geração utilizando RAG com uma chamada direta ao modelo.

## Limitações

Esta implementação é uma prova de conceito e não possui:

- autenticação;
- interface gráfica;
- upload de arquivos;
- avaliação automática das respostas;
- reranking;
- recuperação híbrida;
- controle avançado de acesso;
- tratamento completo de indisponibilidade das APIs.

A qualidade das respostas também depende diretamente da qualidade dos documentos e dos trechos recuperados.

## Problemas encontrados

Durante o desenvolvimento foi inicialmente considerada a utilização de embeddings locais com `sentence-transformers`.

A instalação dessa solução adicionou PyTorch e dependências CUDA muito grandes para o ambiente disponível.

A abordagem foi substituída por embeddings fornecidos pela API Gemini.

Também foram encontrados limites de requisições durante a geração dos embeddings e indisponibilidades temporárias da API generativa.

Esses problemas demonstram que aplicações com LLMs também dependem de aspectos como infraestrutura, disponibilidade, quotas e custos de serviços externos.

## Segurança

Nunca envie a chave da API para o repositório.

O arquivo `.gitignore` deve conter:
```
.env
.venv/
__pycache__/
*.pyc
chroma_db/
```

## Autor

André Lucas Cunha de Almeida

Sistemas de Informação — Universidade Federal de Lavras

## Contexto acadêmico

Projeto desenvolvido como Prova de Conceito para o trabalho de Prospecção Teórica e Tecnológica da disciplina de Engenharia de Software.
