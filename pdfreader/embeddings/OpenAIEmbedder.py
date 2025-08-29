import os, logging
from typing import List, Optional
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, before_sleep_log
from openai import APIConnectionError, APITimeoutError, RateLimitError, APIStatusError

from .base import BaseEmbedder

logger = logging.getLogger("embedder")

# ---- OpenAI ----
class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        from openai import OpenAI  
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError, APIStatusError)),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(6),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        '''
        Batch computing of text embeddings.
        :param texts: Text array (Suggested 64 to 128 pieces per batch)
        :return: Vector array (corresponding one-to-one with the input order)
        '''
        if not texts:
            return []
        r = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in r.data]
