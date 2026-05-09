from __future__ import annotations


def _join_paragraphs(paragraphs: list[str], sep: str) -> str:
    parts = []
    for i, para in enumerate(paragraphs):
        if i < len(paragraphs) - 1 and para.endswith("."):
            para = para[:-1]
        parts.append(para)
    return sep.join(parts)


def chunk_text(text: str, chunk_size: int = 10240) -> list[str]:
    """Split text into chunks at paragraph boundaries.

    Paragraphs within a chunk are joined with double newlines so TTS
    inserts a natural pause between them. A single paragraph larger than
    chunk_size is kept as its own chunk rather than split mid-paragraph.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    SEP = " ... "
    SEP_LEN = len(SEP.encode("utf-8"))

    chunks: list[str] = []
    current: list[str] = []
    current_bytes = 0

    for para in paragraphs:
        para_bytes = len(para.encode("utf-8"))
        overhead = SEP_LEN if current else 0

        if current and current_bytes + overhead + para_bytes > chunk_size:
            chunks.append(_join_paragraphs(current, SEP))
            current = [para]
            current_bytes = para_bytes
        else:
            current.append(para)
            current_bytes += overhead + para_bytes

    if current:
        chunks.append(_join_paragraphs(current, SEP))

    return chunks
