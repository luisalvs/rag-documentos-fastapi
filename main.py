from pathlib import Path
import pymupdf


CAMINHO_PDF = Path("documentos/exemplo.pdf")


def extrair_texto(caminho):
    paginas = []

    with pymupdf.open(caminho) as documento:
        print(f"Total de páginas: {len(documento)}")

        for numero, pagina in enumerate(documento, start=1):
            texto = pagina.get_text("text", sort=True).strip()

            if texto:
                paginas.append({
                    "pagina": numero,
                    "texto": texto
                })

    return paginas


def dividir_texto(texto, tamanho=500, sobreposicao=100):
    if tamanho <= 0 or not 0 <= sobreposicao < tamanho:
        raise ValueError("Tamanho e sobreposição inválidos")

    trechos = []
    passo = tamanho - sobreposicao

    for inicio in range(0, len(texto), passo):
        trecho = texto[inicio:inicio + tamanho]

        if trecho.strip():
            trechos.append(trecho)

        if inicio + tamanho >= len(texto):
            break

    return trechos


def main():
    paginas = extrair_texto(CAMINHO_PDF)

    todos_trechos = []

    for pagina in paginas:
        trechos = dividir_texto(pagina["texto"])

        for numero, trecho in enumerate(trechos, start=1):
            todos_trechos.append({
                "pagina": pagina["pagina"],
                "trecho": numero,
                "texto": trecho
            })

    print(f"Total de trechos: {len(todos_trechos)}")

    for item in todos_trechos[:3]:
        print("\n" + "=" * 50)
        print(f"Página: {item['pagina']}")
        print(f"Trecho: {item['trecho']}")
        print(item["texto"])


if __name__ == "__main__":
    main()
