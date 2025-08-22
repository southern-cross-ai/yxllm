from typing import List

class BaseEmbedder:
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, q: str) -> List[float]:
        vecs = self.embed_texts([q])
        return vecs[0] if vecs else []