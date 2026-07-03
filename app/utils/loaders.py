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


def load_pdf_timetable(file_path: str) -> str:
    """
    Timetable-optimized PDF extractor.

    Converts every table row into a natural-language sentence so the LLM
    can answer questions like "When is CS301?" or "What is on Monday at 08:00?"

    Example output row:
        Monday 08:00-10:00 | Course: CS301 Database Systems | Type: Theory |
        Venue: UDSM Lab 3 | Lecturer: Dr. Makundi
    """
    sentences = []
    header_context = ""  # carries day/date header across rows

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # Grab any page-level text for context (programme name, semester, year, etc.)
            page_header = page.extract_text() or ""
            header_lines = [
                line.strip() for line in page_header.splitlines()
                if line.strip() and len(line.strip()) < 120
            ]
            # Keep the first few lines as context for this page
            context_prefix = " | ".join(header_lines[:4]) if header_lines else ""

            tables = page.extract_tables()
            if not tables:
                # No tables on this page — keep plain text as-is
                if page_header.strip():
                    sentences.append(page_header.strip())
                continue

            for table in tables:
                if not table:
                    continue

                # Detect header row — look for row containing "Time" or "Day"
                header_row = None
                data_rows = table
                for i, row in enumerate(table):
                    row_text = " ".join(str(c or "") for c in row).lower()
                    if any(kw in row_text for kw in ["time", "day", "date", "period", "slot", "from", "to"]):
                        header_row = [str(c or "").replace("\n", " ").strip() for c in row]
                        data_rows = table[i + 1:]
                        break

                col_count = len(header_row) if header_row else (len(table[0]) if table else 0)

                for row in data_rows:
                    if not row or all(c is None or str(c).strip() == "" for c in row):
                        continue

                    cells = [str(c or "").replace("\n", " ").strip() for c in row]

                    # If first cell looks like a day name, save it as running context
                    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
                    if cells[0].lower() in days:
                        header_context = cells[0]

                    # Build a natural-language sentence from the row
                    if header_row:
                        parts = []
                        if header_context:
                            parts.append(header_context)
                        for col_name, cell_val in zip(header_row, cells):
                            if cell_val:
                                parts.append(f"{col_name}: {cell_val}")
                        sentence = " | ".join(parts)
                    else:
                        # No header — join non-empty cells
                        parts = [c for c in cells if c]
                        if header_context:
                            parts.insert(0, header_context)
                        sentence = " | ".join(parts)

                    if sentence.strip():
                        # Prepend programme context from page header
                        full_sentence = f"{context_prefix} | {sentence}" if context_prefix else sentence
                        sentences.append(full_sentence)

    return "\n".join(sentences)


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
        if strategy == "timetable":
            return load_pdf_timetable(file_path)
        return load_pdf(file_path)
    elif file_type == "docx":
        return load_docx(file_path)
    elif file_type == "txt":
        return load_txt(file_path)
    else:
        raise ValueError("Unsupported file type")
