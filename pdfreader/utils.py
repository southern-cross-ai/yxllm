import hashlib
from typing import Iterable, List

def file_sha256(path: str, buf: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]

def split_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    chunks, n = [], len(text or "")
    if n == 0:
        return chunks
    step = max(1, chunk_size - overlap)
    i = 0
    while i < n:
        chunks.append(text[i : i + chunk_size])
        if i + chunk_size >= n:
            break
        i += step
    return chunks

def batched(seq: Iterable, n: int):
    buf = []
    for x in seq:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf
