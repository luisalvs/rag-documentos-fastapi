
import asyncio
import pymupdf
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi import File, UploadFile
from starlette.concurrency import run_in_threadpool
from busca import PDF
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


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("[API] Carregando índice do RAG...")

    app.state.indice = carregar_ou_criar_indice()
    app.state.upload_lock = asyncio.Lock()

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


@app.post("/documentos/upload")
async def upload_documento(
    request: Request,
    arquivo: UploadFile = File(...)
):

    LIMITE_BYTES = 10 * 1024 * 1024  # 10 MB

    nome = arquivo.filename or ""

    if Path(nome).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Envie um arquivo PDF."
        )

    conteudo = await arquivo.read(LIMITE_BYTES + 1)
    await arquivo.close()

    if len(conteudo) > LIMITE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="O PDF deve ter no máximo 10 MB."
        )

    if not conteudo.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="Arquivo PDF inválido."
        )

    try:
        with pymupdf.open(
            stream=conteudo,
            filetype="pdf"
        ) as documento:

            if documento.needs_pass:
                raise HTTPException(
                    status_code=422,
                    detail="PDF protegido por senha."
                )

            paginas = len(documento)

            possui_texto = any(
                pagina.get_text("text").strip()
                for pagina in documento
            )

            if not possui_texto:
                raise HTTPException(
                    status_code=422,
                    detail="PDF sem texto extraível."
                )

    except (pymupdf.FileDataError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Não foi possível abrir o PDF."
        )

    async with request.app.state.upload_lock:

        # Guardar o documento anterior para recuperação
        documento_anterior = (
            PDF.read_bytes() if PDF.exists() else None
        )

        try:
            PDF.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            PDF.write_bytes(conteudo)

            # Recria o índice se o PDF mudou
            novo_indice = await run_in_threadpool(
                carregar_ou_criar_indice
            )

        except Exception:

            # Restaurar o PDF anterior
            if documento_anterior is not None:
                PDF.write_bytes(documento_anterior)
            else:
                PDF.unlink(missing_ok=True)

            raise HTTPException(
                status_code=500,
                detail="Falha ao processar o documento."
            )

        # Atualizar somente após gerar o índice
        request.app.state.indice = novo_indice

    return {
        "mensagem": "Documento processado com sucesso.",
        "arquivo": nome,
        "paginas": paginas,
        "chunks_indexados": len(novo_indice)
    }
