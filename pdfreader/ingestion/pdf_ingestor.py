from typing import Any, Dict, List, Tuple
from ..utils import split_text, file_sha256, batched
from ..embeddings.base import BaseEmbedder
from ..storage.firestore_store import FirestoreChunkStore

class PDFIngestor:
    def __init__(self, embedder: BaseEmbedder, store: FirestoreChunkStore,
                 chunk_size: int = 800, overlap: int = 120, embed_batch: int = 96,
                 meta_backend: str = "", meta_model: str = ""):
        self.embedder = embedder
        self.store = store
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.embed_batch = embed_batch
        self.meta_backend = meta_backend
        self.meta_model = meta_model
    
    def extract_pages_text(pdf_path: str) -> List[Tuple[int, str]]:
        """
        Return [(page_no, text),...]
        First use pdfplumber, and if it fails, use PyMuPDF as a fallback.
        """
        pages = []
        # pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for pageno, page in enumerate(pdf.pages, start=1):
                    text = (page.extract_text() or "").strip()
                    pages.append((pageno, text))
            return pages
        except Exception:
            pass
        try:
            import fitz  # pymupdf
            doc = fitz.open(pdf_path)
            for pageno, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                pages.append((pageno, text))
            return pages
        except Exception as e:
            raise RuntimeError(f"Cannot extract text from {pdf_path}: {e}")

    def ingest(self, pdf_path: str) -> int:
        import os
        source = os.path.basename(pdf_path)
        doc_id = file_sha256(pdf_path)
        pages = self.extract_pages_text(pdf_path)

        chunks: List[Dict[str, Any]] = []
        for pageno, text in pages:
            if not text:
                continue
            pieces = split_text(text, self.chunk_size, self.overlap)
            for idx, piece in enumerate(pieces, start=1):
                cid = f"{doc_id}_p{pageno:04d}_c{idx:03d}"
                chunks.append({
                    "id": cid,
                    "text": piece,
                    "metadata": {
                        "source": source,
                        "page": pageno,
                        "doc_id": doc_id,
                        "chunk_index": idx,
                        "chunk_id": cid,
                    },
                })
        if not chunks:
            return 0

        total = 0
        for pack in batched(chunks, self.embed_batch):
            vecs = self.embedder.embed_texts([c["text"] for c in pack])
            total += self.store.upsert_chunks(
                pack, vecs, batch_size=400,
                meta_extra={"embedding_backend": self.meta_backend,
                            "embedding_model": self.meta_model}
            )
        return total
        

    
