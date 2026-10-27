# RAG para Consulta de Documentos PDF

Aplicação desenvolvida em **Python e FastAPI** para realizar perguntas sobre documentos PDF utilizando **Retrieval-Augmented Generation (RAG)**, embeddings e modelos da OpenAI.

O projeto recupera trechos relevantes do documento e os utiliza como contexto para gerar respostas com indicação das páginas de origem.

## Funcionalidades

- Extração de texto de PDFs com PyMuPDF.
- Divisão do conteúdo em chunks com sobreposição.
- Geração de embeddings com `text-embedding-3-small`.
- Busca semântica usando similaridade de cosseno.
- Cache local de embeddings em JSON, reutilizado quando o documento não muda.
- Geração de respostas com `gpt-4o-mini` a partir dos trechos recuperados.
- Indicação das páginas dos trechos usados como referência.
- API REST com FastAPI e validação de entradas com Pydantic.
- Upload de PDFs pela API, com validação de formato, tamanho e texto extraível.
- Testes automatizados com Pytest e FastAPI TestClient, sem chamadas reais à OpenAI.

## Tecnologias

**Python**, **FastAPI**, **OpenAI API**, **PyMuPDF**, **Pydantic**, **Uvicorn**, **Pytest** e **HTTPX**.

## Como funciona

1. Um PDF é carregado, tem seu texto extraído e dividido em chunks.
2. Os chunks são convertidos em embeddings e armazenados em um cache local.
3. Quando o usuário faz uma pergunta, ela também é convertida em embedding.
4. O sistema compara os vetores por similaridade de cosseno e recupera os três chunks mais relevantes.
5. O `gpt-4o-mini` recebe a pergunta e os trechos recuperados para elaborar a resposta.
6. A API devolve a resposta, as páginas dos trechos recuperados e informações de consumo de tokens.

> O cache é reutilizado quando o conteúdo do PDF e as configurações de indexação permanecem iguais. Cada nova pergunta ainda gera seu próprio embedding.

## Instalação

Clone o repositório e instale as dependências:

```bash
git clone https://github.com/luisalvs/rag-documentos-fastapi.git
cd rag-documentos-fastapi
python3 -m pip install -r requirements.txt
```

Crie um arquivo `.env` na raiz do projeto, baseado no `.env.example`:

```dotenv
OPENAI_API_KEY=sua_chave_aqui
```

Antes de iniciar a API pela primeira vez, adicione um PDF **com texto selecionável** em:

```text
documentos/exemplo.pdf
```

Esse arquivo é necessário para a inicialização do índice atual. Depois que a API estiver no ar, é possível substituí-lo pelo endpoint de upload.

**Segurança:** não publique `.env`, documentos privados nem o conteúdo da pasta `cache/`.

## Executando a API

```bash
python3 -m uvicorn api_rag:app --reload --port 8001
```

Documentação interativa: `http://127.0.0.1:8001/docs`

## Endpoints

### `GET /health`

Verifica o funcionamento da API e informa a quantidade de chunks indexados.

Exemplo de resposta:

```json
{
  "status": "ok",
  "chunks_indexados": 12
}
```

### `POST /perguntar`

Recebe uma pergunta e gera uma resposta com base nos trechos recuperados do PDF ativo.

**Body — JSON:**

```json
{
  "pergunta": "Quais são as vantagens do FastAPI?"
}
```

A resposta inclui os campos `resposta`, `trechos_recuperados`, `tokens_embedding` e `tokens_llm_total`.

Perguntas com menos de 10 caracteres retornam **HTTP 422**.

### `POST /documentos/upload`

Recebe um novo PDF, processa o texto, gera os embeddings e atualiza o índice utilizado nas consultas seguintes.

No Postman, utilize **Body → form-data**:

| Campo | Tipo | Valor |
| --- | --- | --- |
| `arquivo` | File | Selecione um arquivo `.pdf` |

O tamanho máximo aceito é **10 MB**. Extensões inválidas retornam **HTTP 400**; arquivos maiores que o limite retornam **HTTP 413**; PDFs sem texto extraível retornam **HTTP 422**.

**Atenção:** nesta versão, somente um documento fica ativo por vez. Um novo upload substitui o documento anterior.

## Testes automatizados

O projeto utiliza **Pytest** e **FastAPI TestClient** para testar:

- Health check da API.
- Perguntas válidas com respostas simuladas.
- Validação de perguntas muito curtas.
- Upload de PDFs válidos.
- Rejeição de arquivos com extensão inválida.
- Rejeição de PDFs sem texto extraível.

Execute:

```bash
python3 -m pytest -v tests/
```

Os testes usam mocks e arquivos temporários, sem chamadas reais à OpenAI e sem substituir o PDF usado localmente.

## Limitações atuais

- Apenas um documento PDF ativo por vez; novos uploads substituem o anterior.
- O upload é limitado a 10 MB.
- PDFs digitalizados sem camada de texto não são processados via OCR.
- O cache é armazenado em JSON local, sem banco vetorial.
- Não há autenticação, controle de acesso ou limitação de requisições por usuário.
- A recuperação de trechos não garante a precisão de todas as respostas geradas.
- As citações de páginas ainda não passam por uma validação automática independente.

## Próximas melhorias

- Suporte a múltiplos documentos PDF e identificação do documento de origem.
- Banco vetorial para volumes maiores de documentos.
- Validação mais rigorosa das fontes e respostas geradas pela IA.
- Autenticação e controle de acesso à API.
- Divisão de texto mais inteligente, respeitando parágrafos e sentenças.

## Objetivo

Projeto de estudos em **Engenharia de IA**, com foco em aplicações práticas de RAG, processamento documental e desenvolvimento de APIs em Python.
