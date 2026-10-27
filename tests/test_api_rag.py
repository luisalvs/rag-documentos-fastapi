
from types import SimpleNamespace

import pytest
import pymupdf

from fastapi.testclient import TestClient

import api_rag


@pytest.fixture
def cliente(monkeypatch, tmp_path):

    # Impede que os testes substituam seu PDF real.
    monkeypatch.setattr(
        api_rag,
        "PDF",
        tmp_path / "teste.pdf"
    )

    # Simula a indexação, sem chamar a OpenAI.
    def indice_simulado():
        return [{
            "pagina": 2,
            "texto": "O documento explica consultas de veículos.",
            "vetor": [0.1, 0.2, 0.3]
        }]

    monkeypatch.setattr(
        api_rag,
        "carregar_ou_criar_indice",
        indice_simulado
    )

    # Executa o lifespan da aplicação.
    with TestClient(api_rag.app) as client:
        yield client


def criar_pdf(texto=None):

    documento = pymupdf.open()
    pagina = documento.new_page()

    if texto:
        pagina.insert_text(
            (72, 72),
            texto
        )

    conteudo = documento.tobytes()
    documento.close()

    return conteudo


def test_health(cliente):

    response = cliente.get("/health")

    assert response.status_code == 200
    assert response.json()["chunks_indexados"] == 1


def test_pergunta_valida(cliente, monkeypatch):

    def resposta_simulada(pergunta, indice):

        return (
            "O documento trata de consultas de veículos. [Página 2]",
            [{
                "pagina": 2,
                "similaridade": 0.75
            }],
            SimpleNamespace(total_tokens=90),
            12
        )

    monkeypatch.setattr(
        api_rag,
        "responder_pergunta",
        resposta_simulada
    )

    response = cliente.post(
        "/perguntar",
        json={
            "pergunta": "Qual é o assunto do documento?"
        }
    )

    assert response.status_code == 200

    dados = response.json()

    assert "veículos" in dados["resposta"]
    assert dados["trechos_recuperados"][0]["pagina"] == 2
    assert dados["tokens_embedding"] == 12
    assert dados["tokens_llm_total"] == 90


def test_pergunta_curta(cliente):

    response = cliente.post(
        "/perguntar",
        json={"pergunta": "Teste"}
    )

    assert response.status_code == 422


def test_upload_valido(cliente):

    pdf = criar_pdf(
        "Manual de consultas e procedimentos veiculares."
    )

    response = cliente.post(
        "/documentos/upload",
        files={
            "arquivo": (
                "manual.pdf",
                pdf,
                "application/pdf"
            )
        }
    )

    assert response.status_code == 200

    dados = response.json()

    assert dados["paginas"] == 1
    assert dados["chunks_indexados"] == 1

    # Verifica se o upload foi salvo no arquivo temporário.
    assert api_rag.PDF.read_bytes() == pdf


def test_upload_extensao_invalida(cliente):

    response = cliente.post(
        "/documentos/upload",
        files={
            "arquivo": (
                "arquivo.txt",
                b"Conteudo de teste",
                "text/plain"
            )
        }
    )

    assert response.status_code == 400


def test_upload_pdf_sem_texto(cliente):

    # Gera um PDF válido, mas com página vazia.
    pdf = criar_pdf()

    response = cliente.post(
        "/documentos/upload",
        files={
            "arquivo": (
                "vazio.pdf",
                pdf,
                "application/pdf"
            )
        }
    )

    assert response.status_code == 422
