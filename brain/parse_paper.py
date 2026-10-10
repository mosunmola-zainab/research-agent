import os
import sys
from pathlib import Path

from docling.datamodel.base_models import ConversionStatus
from docling.document_converter import DocumentConverter

PDF_PATH = Path("evaluation/papers/s11831-026-10675-8.pdf")
OUTPUT_DIR = Path("evaluation/parsed")
OUTPUT_NAME = "paper_01"


def main() -> None:
    # Paths are relative to brain/, so the launch folder must not matter.
    os.chdir(Path(__file__).resolve().parent)

    converter = DocumentConverter()
    result = converter.convert(PDF_PATH)
    document = result.document

    # Partial conversions don't raise, so check before saving anything.
    print(f"Status: {result.status}")
    if result.status != ConversionStatus.SUCCESS:
        sys.exit(f"Conversion did not fully succeed: {result.errors}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    markdown_path = OUTPUT_DIR / f"{OUTPUT_NAME}.md"
    json_path = OUTPUT_DIR / f"{OUTPUT_NAME}.json"

    # JSON keeps the page numbers that Markdown drops.
    document.save_as_markdown(markdown_path)
    document.save_as_json(json_path)

    print(f"Pages: {document.num_pages()}")
    print(f"Markdown: {markdown_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
