from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 10240) -> list[str]:
    """Split text into chunks at paragraph boundaries.

    Paragraphs within a chunk are joined with a single space so the text
    flows naturally for TTS. A single paragraph larger than chunk_size is
    kept as its own chunk rather than split mid-paragraph.
    """
    raw = [p.strip() for p in text.split("\n\n") if p.strip()]
    paragraphs = [
        p if p[-1] in ".!?" else p + "."
        for p in raw
    ]

    SEP = " "
    SEP_LEN = len(SEP.encode("utf-8"))

    chunks: list[str] = []
    current: list[str] = []
    current_bytes = 0

    for para in paragraphs:
        para_bytes = len(para.encode("utf-8"))
        overhead = SEP_LEN if current else 0

        if current and current_bytes + overhead + para_bytes > chunk_size:
            chunks.append(SEP.join(current))
            current = [para]
            current_bytes = para_bytes
        else:
            current.append(para)
            current_bytes += overhead + para_bytes

    if current:
        chunks.append(SEP.join(current))

    return chunks
