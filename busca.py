
import hashlib
import json
from pathlib import Path
from main import extrair_texto, dividir_texto
from embeddings import (
    MODELO,
    gerar_embeddings,
    similaridade_cosseno
)


BASE_DIR = Path(__file__).resolve().parent

PDF = BASE_DIR / "documentos" / "exemplo.pdf"
CACHE = BASE_DIR / "cache" / "indice.json"

TAMANHO_CHUNK = 800
SOBREPOSICAO = 150


def gerar_metadados():

    hash_pdf = hashlib.sha256(
        PDF.read_bytes()
    ).hexdigest()

    return {
        "hash_pdf": hash_pdf,
        "modelo": MODELO,
        "tamanho_chunk": TAMANHO_CHUNK,
        "sobreposicao": SOBREPOSICAO
    }


def carregar_ou_criar_indice():

    metadados = gerar_metadados()

    if CACHE.exists():

        try:
            dados = json.loads(
                CACHE.read_text(encoding="utf-8")
            )

            if (
                dados.get("metadados") == metadados
                and dados.get("itens")
            ):
                print("[CACHE] Embeddings carregados do arquivo.")
                return dados["itens"]

        except (OSError, ValueError, AttributeError):
            print("[CACHE] Arquivo inválido. Recriando índice.")

    print("[CACHE] Gerando embeddings do documento...")

    paginas = extrair_texto(PDF)

    itens = []

    for pagina in paginas:

        trechos = dividir_texto(
            pagina["texto"],
            tamanho=TAMANHO_CHUNK,
            sobreposicao=SOBREPOSICAO
        )

        for trecho in trechos:
            itens.append({
                "pagina": pagina["pagina"],
                "texto": trecho
            })

    if not itens:
        raise ValueError(
            "Nenhum texto extraível encontrado no PDF."
        )

    textos = [item["texto"] for item in itens]

    vetores, tokens = gerar_embeddings(textos)

    if len(vetores) != len(itens):
        raise RuntimeError(
            "Quantidade de vetores diferente dos chunks."
        )

    for item, vetor in zip(itens, vetores):
        item["vetor"] = vetor

    CACHE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    CACHE.write_text(
        json.dumps(
            {
                "metadados": metadados,
                "itens": itens
            },
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(f"[CACHE] {len(itens)} embeddings salvos.")
    print(f"[CACHE] Tokens utilizados: {tokens}")

    return itens


def buscar(pergunta, indice, limite=3):

    vetores_pergunta, tokens = gerar_embeddings(
        [pergunta]
    )

    vetor_pergunta = vetores_pergunta[0]

    resultados = []

    for item in indice:

        similaridade = similaridade_cosseno(
            vetor_pergunta,
            item["vetor"]
        )

        resultados.append({
            "pagina": item["pagina"],
            "texto": item["texto"],
            "similaridade": similaridade
        })

    resultados.sort(
        key=lambda item: item["similaridade"],
        reverse=True
    )

    return resultados[:limite], tokens


def main():

    indice = carregar_ou_criar_indice()

    print(f"\nChunks disponíveis: {len(indice)}")
    print("Buscador semântico iniciado.")
    print("Digite 'sair' para encerrar.")

    while True:

        pergunta = input(
            "\nDigite sua pergunta: "
        ).strip()

        if pergunta.lower() == "sair":
            print("Buscador encerrado.")
            break

        if not pergunta:
            continue

        resultados, tokens = buscar(
            pergunta,
            indice
        )

        print("\nTOP 3 RESULTADOS")
        print(f"Tokens da pergunta: {tokens}")

        for posicao, item in enumerate(
            resultados,
            start=1
        ):

            print("\n" + "=" * 50)
            print(f"Posição: {posicao}")
            print(f"Página: {item['pagina']}")
            print(
                f"Similaridade: "
                f"{item['similaridade']:.4f}"
            )
            print(f"Texto:\n{item['texto']}")


if __name__ == "__main__":
    main()
