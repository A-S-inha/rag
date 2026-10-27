"""Structure-aware HTML extraction. Tables and images bypass prose splitting."""
from pathlib import Path
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter

PIPELINE_VERSION = "structure-v2"


def nearby_headings(tag):
    result = {}
    # Walk backwards until each level is encountered; stop at its parent boundary.
    level = 4
    for previous in tag.previous_elements:
        name = getattr(previous, "name", None)
        if name in ("h1", "h2", "h3") and int(name[1]) < level:
            level = int(name[1])
            result[name] = previous.get_text(" ", strip=True)
    return result


def table_grid(table):
    """Expand merged cells into an explicit rectangular grid."""
    occupied = {}
    rows = [row for row in table.find_all("tr") if row.find_parent("table") is table]
    if len(rows) > 10000:
        raise ValueError("Table exceeds 10,000 rows")
    for r, row in enumerate(rows):
        c = 0
        for cell in row.find_all(["th", "td"], recursive=False):
            while (r, c) in occupied:
                c += 1
            rs, cs = int(cell.get("rowspan", 1)), int(cell.get("colspan", 1))
            if not 1 <= rs <= 100 or not 1 <= cs <= 100:
                raise ValueError("Table spans must be between 1 and 100")
            text = cell.get_text(" ", strip=True)
            for dr in range(rs):
                for dc in range(cs):
                    occupied[r + dr, c + dc] = text
            c += cs
    width = max((c for _, c in occupied), default=-1) + 1
    return [[occupied.get((r, c), "") for c in range(width)] for r in range(len(rows))]


def table_documents(table, metadata, table_id, chunk_size):
    if table.find("table"):
        raise ValueError("Nested tables require a custom parser")
    grid = table_grid(table)
    if not grid:
        return []
    first = table.find("tr")
    has_headers = bool(first and first.find("th"))
    headers = grid[0] if has_headers else [f"Column {n+1}" for n in range(len(grid[0]))]
    rows = grid[1:] if has_headers else grid
    caption = table.find("caption")
    prefix = f"Table {table_id}: {caption.get_text(' ', strip=True) if caption else ''}\n"
    prefix += "Headers: " + " | ".join(headers) + "\n"
    docs, batch, start = [], [], 1

    def emit():
        docs.append(Document(page_content=prefix + "\n".join(batch), metadata={
            **metadata, "kind": "table", "table_id": table_id,
            "row_start": start, "row_end": start + len(batch) - 1,
        }))

    for number, row in enumerate(rows, 1):
        line = f"Row {number}: " + "; ".join(f"{key} = {value}" for key, value in zip(headers, row))
        if batch and len(prefix) + sum(len(x) + 1 for x in batch) + len(line) > chunk_size:
            emit()
            batch, start = [], number
        batch.append(line)
    if batch:
        emit()
    elif not rows:
        docs.append(Document(page_content=prefix, metadata={**metadata, "kind": "table", "table_id": table_id}))
    return docs


def process_html(path, chunk_size=900, chunk_overlap=150, image_describer=None):
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
    path = Path(path).resolve()
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else path.name
    for tag in soup.select("script, style, noscript, nav, footer, template"):
        tag.decompose()
    content = soup.find("main") or soup.find("article") or soup.body or soup
    base = {"source": str(path), "title": title}
    special = []
    for number, table in enumerate(list(content.find_all("table")), 1):
        special.extend(table_documents(table, {**base, **nearby_headings(table)}, number, chunk_size))
        table.decompose()
    for number, image in enumerate(list(content.find_all("img")), 1):
        src = image.get("src", "")
        figure = image.find_parent("figure")
        caption = figure.find("figcaption") if figure else None
        description = image_describer(path, src) if image_describer and src else ""
        text = "\n".join(x for x in [
            f"Image {number}: {image.get('alt', '')}",
            caption.get_text(" ", strip=True) if caption else "", description,
        ] if x)
        special.append(Document(page_content=text, metadata={
            **base, **nearby_headings(image), "kind": "image", "image_src": src,
            "vision_processed": bool(description),
        }))
        image.decompose()
    sections = HTMLHeaderTextSplitter(
        headers_to_split_on=[("h1", "h1"), ("h2", "h2"), ("h3", "h3")]
    ).split_text(str(content))
    if not sections:
        text = content.get_text(" ", strip=True)
        sections = [Document(page_content=text)] if text else []
    # Some splitter versions emit headings as separate documents. Metadata already
    # carries these headings; avoid spending retrieval slots on heading-only text.
    sections = [s for s in sections if s.page_content.strip() not in s.metadata.values()]
    for section in sections:
        section.metadata.update(**base, kind="text")
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    ).split_documents(sections) + special
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
    if not chunks:
        raise ValueError("HTML contains no indexable text")
    return chunks
