
from pathlib import Path
from math import sqrt
from dotenv import load_dotenv
from openai import OpenAI
from main import extrair_texto, dividir_texto


BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

client = OpenAI(timeout=30.0, max_retries=1)

MODELO = "text-embedding-3-small"
PDF = BASE_DIR / "documentos" / "exemplo.pdf"


def gerar_embeddings(textos):

    response = client.embeddings.create(
        model=MODELO,
        input=textos,
        encoding_format="float"
    )

    vetores = [
        item.embedding
        for item in sorted(
            response.data,
            key=lambda item: item.index
        )
    ]

    return vetores, response.usage.total_tokens


def similaridade_cosseno(vetor1, vetor2):

    produto = sum(
        a * b
        for a, b in zip(vetor1, vetor2)
    )

    norma1 = sqrt(sum(a * a for a in vetor1))
    norma2 = sqrt(sum(b * b for b in vetor2))

    if norma1 == 0 or norma2 == 0:
        return 0.0

    return produto / (norma1 * norma2)


def main():

    paginas = extrair_texto(PDF)

    chunks = []

    for pagina in paginas:

        trechos = dividir_texto(
            pagina["texto"],
            tamanho=800,
            sobreposicao=150
        )

        for trecho in trechos:
            chunks.append({
                "pagina": pagina["pagina"],
                "texto": trecho
            })

    print(f"\nTotal de chunks: {len(chunks)}")

    if not chunks:
        print("Nenhum texto encontrado no PDF.")
        return

    textos = [item["texto"] for item in chunks]

    vetores, tokens_documento = gerar_embeddings(textos)

    print(f"Embeddings gerados: {len(vetores)}")
    print(f"Dimensões por vetor: {len(vetores[0])}")

    pergunta = (
        "O FastAPI utiliza quais bibliotecas "
        "para validação de dados e backend?"
    )

    print(f"\nPergunta: {pergunta}")

    vetores_pergunta, tokens_pergunta = gerar_embeddings(
        [pergunta]
    )

    vetor_pergunta = vetores_pergunta[0]

    resultados = []

    for chunk, vetor in zip(chunks, vetores):

        similaridade = similaridade_cosseno(
            vetor_pergunta,
            vetor
        )

        resultados.append({
            "pagina": chunk["pagina"],
            "texto": chunk["texto"],
            "similaridade": similaridade
        })

    resultados.sort(
        key=lambda item: item["similaridade"],
        reverse=True
    )

    print("\nTOP 3 TRECHOS MAIS RELEVANTES")

    for posicao, item in enumerate(resultados[:3], start=1):

        print("\n" + "=" * 50)
        print(f"Posição: {posicao}")
        print(f"Página: {item['pagina']}")
        print(f"Similaridade: {item['similaridade']:.4f}")
        print(f"Texto:\n{item['texto']}")

    tokens_total = tokens_documento + tokens_pergunta

    custo = tokens_total * 0.02 / 1_000_000

    print("\n" + "=" * 50)
    print(f"Tokens utilizados: {tokens_total}")
    print(f"Custo estimado: US$ {custo:.8f}")


if __name__ == "__main__":
    main()
