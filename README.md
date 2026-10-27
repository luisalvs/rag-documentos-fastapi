
# RAG para Consulta de Documentos PDF

Aplicação desenvolvida em Python e FastAPI para realizar
perguntas sobre documentos PDF utilizando Retrieval-Augmented
Generation (RAG), embeddings e modelos da OpenAI.

## Funcionalidades

- Extração de texto de documentos PDF com PyMuPDF.
- Divisão do conteúdo em chunks com sobreposição.
- Geração de embeddings com text-embedding-3-small.
- Busca semântica utilizando similaridade de cosseno.
- Cache local de embeddings em JSON.
- Geração de respostas utilizando gpt-4o-mini.
- Referência às páginas dos trechos recuperados.
- API REST desenvolvida com FastAPI.
- Validação de entradas com Pydantic.

## Tecnologias

Python, FastAPI, OpenAI API, PyMuPDF, Pydantic e Uvicorn.

## Como funciona

1. O documento PDF é processado e dividido em chunks.
2. Os chunks são transformados em embeddings.
3. Os vetores são armazenados em cache local.
4. A pergunta do usuário é transformada em embedding.
5. O sistema recupera os três chunks mais relevantes.
6. Um LLM gera a resposta com base no contexto recuperado.

## Instalação

Clone o repositório e instale as dependências:

```bash
pip install -r requirements.txt
```

Crie um arquivo `.env`:

```dotenv
OPENAI_API_KEY=sua_chave_aqui
```

Adicione um documento PDF com texto selecionável em:

`documentos/exemplo.pdf`

## Executando a API

```bash
uvicorn api_rag:app --reload --port 8001
```

Documentação interativa:

`http://127.0.0.1:8001/docs`

## Endpoints

### GET /health

Verifica o funcionamento da API e retorna a quantidade
de chunks indexados.

### POST /perguntar

Exemplo de requisição:

```json
{
  "pergunta": "Quais são as vantagens do FastAPI?"
}
```

A resposta contém o texto gerado, os metadados dos
trechos recuperados e as informações de consumo de tokens.

## Limitações atuais

- Processa um PDF configurado localmente.
- Utiliza cache JSON, sem banco vetorial.
- Não possui autenticação ou limites de requisição.
- A recuperação de trechos não garante que toda resposta
  gerada esteja correta.
- Ainda não possui testes automatizados próprios.

## Próximas melhorias

- Upload de documentos pela API.
- Suporte a múltiplos PDFs.
- Testes automatizados com Pytest.
- Melhorias na validação das fontes e respostas.

## Objetivo

Projeto de estudos em Engenharia de IA, com foco
na integração de RAG, processamento documental e
desenvolvimento de APIs em Python.
