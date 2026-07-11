from pypdf import PdfReader
import docx
import pdfplumber


def load_pdf(file_path: str) -> str:
    text = []

    with pdfplumber.open(file_path) as plumber_pdf:
        for page in plumber_pdf.pages:
            page_parts = []

            # Extract tables first, formatted as clean markdown
            tables = page.extract_tables()
            table_bboxes = []

            if tables:
                for table in tables:
                    if not table or not table[0]:
                        continue

                    table_md = "\n"
                    for row_idx, row in enumerate(table):
                        clean_row = [
                            str(cell).replace('\n', ' ').strip() if cell is not None else ""
                            for cell in row
                        ]
                        table_md += "| " + " | ".join(clean_row) + " |\n"
                        if row_idx == 0:
                            table_md += "| " + " | ".join(["---"] * len(row)) + " |\n"

                    page_parts.append(table_md.strip())

            # Then extract remaining plain text (excluding table regions)
            plain_text = page.extract_text() or ""
            if plain_text.strip():
                page_parts.append(plain_text.strip())

            if page_parts:
                text.append("\n\n".join(page_parts))

    return "\n\n".join(text)


def load_pdf_fast(file_path: str) -> str:
    """Fast raw text extraction using pypdf. Best for text-heavy books without tables."""
    reader = PdfReader(file_path)
    text = []
    
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text.append(page_text)
        
    return "\n".join(text)


def load_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    parts = []

    # Extract paragraph text
    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text.strip())

    # Also extract text from tables (previously missed)
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.replace("\n", " ").strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))

    return "\n".join(parts)


def load_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        return file.read()


def load_document(file_path: str, file_type: str, strategy: str = "auto") -> str:
    if file_type == "pdf":
        if strategy == "fast":
            return load_pdf_fast(file_path)
        return load_pdf(file_path)
    elif file_type == "docx":
        return load_docx(file_path)
    elif file_type in ("txt", "md"):
        return load_txt(file_path)
    else:
        raise ValueError("Unsupported file type")
