from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from openai import OpenAI

class RAGQA:
    def __init__(
        self,
        store,
        embedder,
        *,
        gen_model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_ctx_chars: int = 6000,
        default_k: int = 8,
        min_sim: Optional[float] = None,
        temperature: float = 0.2,
        system_msg: str = (
            "You are a rigorous retrieval-augmented assistant. "
            "Answer strictly based on the provided materials. "
            "If the materials are insufficient, say 'insufficient information' instead of guessing. "
            "Use bracketed citations like [file.pdf#p3]."
        ),
    ):
        self.store = store
        self.embedder = embedder
        self.client = self._resolve_openai_client(
            embedder=embedder,
            api_key=api_key,
            base_url=base_url,
        )
        self.gen_model = gen_model
        self.max_ctx_chars = max_ctx_chars
        self.default_k = default_k
        self.min_sim = min_sim
        self.temperature = temperature
        self.system_msg = system_msg

    @staticmethod
    def _resolve_openai_client(
        *,
        embedder: Any,
        api_key: Optional[str],
        base_url: Optional[str],
    ) -> OpenAI:
        """
        The client used for parsing and generation, priority:
        1) Explicitly pass api_key/base_url -> create a new OpenAI(...)
        3) Reuse embedder.client (if existing and available)
        4) Fallback: OpenAI() (Use the environment variable OPENAI_API_KEY/default configuration)
        """
        # 1) Externally, the overwritten key/base_url is provided
        if api_key or base_url:
            return OpenAI(api_key=api_key, base_url=base_url)

        # 2) Reuse the client within the embedder
        c = getattr(embedder, "client", None)
        if c is not None:
            return c

        # 3) Catch-all: Environment variable/default
        return OpenAI()

    def _cosine_topk(
        self,
        question: str,
        metas: List[Dict[str, Any]],
        vecs: np.ndarray,
        k: int
    ) -> List[Tuple[Dict[str, Any], float]]:
        if vecs.size == 0:
            return []
        q = np.asarray(self.embedder.embed_query(question), dtype=np.float32)
        V = vecs.astype(np.float32)
        denom = (np.linalg.norm(V, axis=1) * (np.linalg.norm(q) + 1e-12))
        sims = (V @ q) / (denom + 1e-12)
        idx = np.argsort(-sims)[:k]
        hits = [(metas[i], float(sims[i])) for i in idx]
        if self.min_sim is not None:
            hits = [h for h in hits if h[1] >= self.min_sim]
        return hits

    def _build_context(self, hits: List[Tuple[Dict[str, Any], float]]) -> Tuple[str, List[str]]:
        blocks, refs, cur = [], [], 0
        for m, _s in hits:
            ref = f"{m.get('source','?')}#p{m.get('page','?')}"
            txt = (m.get("text") or "").strip().replace("\n", " ")
            line = f"[{ref}] {txt}"
            if cur + len(line) > self.max_ctx_chars:
                break
            blocks.append(line)
            cur += len(line)
            if ref not in refs:
                refs.append(ref)
        return "\n\n".join(blocks), refs

    def _generate(self, context: str, question: str) -> str:
        user = (
            f"Context (multiple passages; each already includes a source tag):\n\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer concisely in English. Include bracketed citations where appropriate."
        )
        r = self.client.chat.completions.create(
            model=self.gen_model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.system_msg},
                {"role": "user",   "content": user},
            ],
        )
        return r.choices[0].message.content.strip()

    def answer(
        self,
        question: str,
        *,
        doc_id: Optional[str] = None,
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        metas, vecs = self.store.load_vectors(doc_id=doc_id)
        if vecs.size == 0:
            return {"answer": "Insufficient information: no content in the store.", "refs": [], "hits": []}

        hits = self._cosine_topk(question, metas, vecs, k or self.default_k)
        if not hits:
            return {"answer": "Insufficient information: no relevant passages found.", "refs": [], "hits": []}

        context, refs = self._build_context(hits)
        if not context:
            return {"answer": "Insufficient information: relevant passages are too long to include.", "refs": refs, "hits": hits}

        ans = self._generate(context, question)
        return {"answer": ans, "refs": refs, "hits": hits}
