
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from openai import (
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    AuthenticationError,
    OpenAIError,
)

from busca import carregar_ou_criar_indice
from rag import responder_pergunta


# Carregamento único do índice na inicialização
@asynccontextmanager
async def lifespan(app: FastAPI):

    print("[API] Carregando índice do RAG...")

    app.state.indice = carregar_ou_criar_indice()

    print(
        f"[API] {len(app.state.indice)} chunks disponíveis"
    )

    yield

    app.state.indice = None


app = FastAPI(
    title="API RAG - Consulta Inteligente de Documentos",
    version="1.0.0",
    lifespan=lifespan
)


# Modelo da requisição
class PerguntaEntrada(BaseModel):

    pergunta: str = Field(
        min_length=10,
        max_length=1000
    )


# Modelo dos trechos recuperados
class TrechoRecuperado(BaseModel):

    pagina: int
    similaridade: float


# Modelo da resposta
class RespostaRAG(BaseModel):

    resposta: str
    trechos_recuperados: list[TrechoRecuperado]
    tokens_embedding: int
    tokens_llm_total: int | None


@app.get("/health")
def health(request: Request):

    return {
        "status": "ok",
        "chunks_indexados": len(request.app.state.indice)
    }


@app.post(
    "/perguntar",
    response_model=RespostaRAG
)
def perguntar(dados: PerguntaEntrada, request: Request):

    try:

        resposta, resultados, uso, tokens_busca = (
            responder_pergunta(
                dados.pergunta,
                request.app.state.indice
            )
        )

    except APITimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Tempo limite excedido na OpenAI."
        )

    except RateLimitError:
        raise HTTPException(
            status_code=503,
            detail="Limite de requisições da OpenAI atingido."
        )

    except AuthenticationError:
        raise HTTPException(
            status_code=502,
            detail="Falha na autenticação com a OpenAI."
        )

    except APIConnectionError:
        raise HTTPException(
            status_code=502,
            detail="Falha de conexão com a OpenAI."
        )

    except OpenAIError:
        raise HTTPException(
            status_code=502,
            detail="Erro ao processar a solicitação na OpenAI."
        )

    if not resposta or not resposta.strip():
        raise HTTPException(
            status_code=502,
            detail="O modelo não retornou uma resposta."
        )

    return RespostaRAG(
        resposta=resposta,

        trechos_recuperados=[
            TrechoRecuperado(
                pagina=item["pagina"],
                similaridade=round(
                    item["similaridade"], 4
                )
            )
            for item in resultados
        ],

        tokens_embedding=tokens_busca,

        tokens_llm_total=(
            uso.total_tokens
            if uso is not None
            else None
        )
    )
