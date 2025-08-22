import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

import firebase_admin
from firebase_admin import credentials, firestore

def ensure_firebase(cred_source: Optional[str] = None, project_id: Optional[str] = None):
    """Prioritize the use of the GOOGLE_APPLICATION_CREDENTIALS environment variable for initialization."""
    if firebase_admin._apps:
        return
    if isinstance(cred_source, str):
        cred = credentials.Certificate(cred_source)
        firebase_admin.initialize_app(cred, {"projectId": project_id} if project_id else None)
        return
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        env_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        cred = credentials.Certificate(env_path)
        firebase_admin.initialize_app(cred, {"projectId": project_id} if project_id else None)
        return
    try:
        firebase_admin.initialize_app(options={"projectId": project_id} if project_id else None)
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize Firebase Admin. "
            "Please set GOOGLE_APPLICATION_CREDENTIALS or pass cred_source."
        ) from e
    
def get_db(cred_source: Optional[str] = None, project_id: Optional[str] = None) -> firestore.Client:
    ensure_firebase(cred_source, project_id)
    return firestore.client()

class FirestoreChunkStore:
    def __init__(self, collection: str = "chunks",cred_source: Optional[str] = None, project_id: Optional[str] = None):
        self.db = get_db(cred_source, project_id)
        self.collection = collection

    def upsert_chunks(self, chunks: List[Dict[str, Any]], vectors: List[List[float]],
                      batch_size: int = 400, meta_extra: Optional[Dict[str, Any]] = None) -> int:
        assert len(chunks) == len(vectors)
        meta_extra = meta_extra or {}
        total = 0
        pairs: List[Tuple[Dict[str, Any], List[float]]] = list(zip(chunks, vectors))
        for pack in self._batched(pairs, batch_size):
            wb = self.db.batch()
            for c, v in pack:
                ref = self.db.collection(self.collection).document(str(c["id"]))
                data = {
                    "text": c["text"],
                    "embedding": [float(x) for x in v],
                    **c["metadata"],
                    **meta_extra,
                }
                wb.set(ref, data, merge=True)
            wb.commit()
            total += len(pack)
        return total

    def delete_by_doc_id(self, doc_id: str) -> int:
        q = self.db.collection(self.collection).where("doc_id", "==", doc_id)
        n, wb = 0, self.db.batch()
        for snap in q.stream():
            wb.delete(snap.reference); n += 1
            if n % 400 == 0:
                wb.commit(); wb = self.db.batch()
        if n % 400:
            wb.commit()
        return n

    def fetch_by_doc_id(self, doc_id: str) -> List[Dict[str, Any]]:
        q = self.db.collection(self.collection).where("doc_id", "==", doc_id)
        return [d.to_dict() for d in q.stream()]

    def load_vectors(self, doc_id: Optional[str] = None) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        metas, vecs = [], []
        col = self.db.collection(self.collection)
        it = col.where("doc_id", "==", doc_id).stream() if doc_id else col.stream()
        for snap in it:
            d = snap.to_dict()
            if "embedding" in d and "text" in d:
                metas.append(d)
                vecs.append(d["embedding"])
        if not metas:
            return [], np.empty((0, 0), dtype=np.float32)
        import numpy as np  # 局部导入避免不必要依赖
        return metas, np.array(vecs, dtype=np.float32)

    @staticmethod
    def _batched(seq, n):
        buf = []
        for x in seq:
            buf.append(x)
            if len(buf) >= n:
                yield buf; buf = []
        if buf:
            yield buf
