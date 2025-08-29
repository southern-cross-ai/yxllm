# server.py
# pip install fastapi uvicorn "pydantic<2" firebase-admin openai numpy tenacity

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# 按你的项目结构导入（注意大小写）
from pdfreader.storage.firestore_store import FirestoreChunkStore
from pdfreader.embeddings.OpenAIEmbedder import OpenAIEmbedder
from pdfreader.QA.answer import RAGQA

# ------------ 配置（环境变量） ------------
FIRESTORE_COLLECTION = os.getenv("FIRESTORE_COLLECTION", "chunks")
GCP_PROJECT_ID       = os.getenv("GCP_PROJECT_ID")
GOOGLE_CREDS         = os.getenv("GOOGLE_APPLICATION_CREDENTIALS","./pdfreader/modelworks-firebase.json")  # 推荐设置
EMBED_MODEL          = os.getenv("EMBED_MODEL", "text-embedding-3-small")
GEN_MODEL            = os.getenv("GEN_MODEL", "gpt-4o-mini")
GEN_BASE_URL         = os.getenv("GEN_BASE_URL")  # 可接 Joey: http://13.239.88.166:8000/v1
GEN_API_KEY          = os.getenv("GEN_API_KEY", "sk-xxx"
)   # Joey 用 "EMPTY"；直连 OpenAI 用 sk-...

DEFAULT_MIN_SIM      = float(os.getenv("MIN_SIM", "0.18"))
MAX_CTX_CHARS        = int(os.getenv("MAX_CTX_CHARS", "6000"))

# ------------ 应用与 CORS ------------
app = FastAPI(title="PDF RAG API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产改成你的前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------ 启动时初始化单例 ------------
@app.on_event("startup")
def _startup():
    try:
        store = FirestoreChunkStore(
            collection=FIRESTORE_COLLECTION,
            cred_source=GOOGLE_CREDS,      # 若未设置，会尝试 ADC
            project_id=GCP_PROJECT_ID,
        )
        embedder = OpenAIEmbedder(model=EMBED_MODEL,api_key=GEN_API_KEY)
        qa = RAGQA(
            store, embedder,
            gen_model=GEN_MODEL,
            base_url=GEN_BASE_URL,
            min_sim=DEFAULT_MIN_SIM,
            max_ctx_chars=MAX_CTX_CHARS,
        )
        app.state.qa = qa
    except Exception as e:
        # 不让服务直接挂掉，方便 /health 自检
        app.state.qa = None
        print("Startup error:", repr(e))

# ------------ 模型 ------------
class AskBody(BaseModel):
    question: str
    doc_id: str | None = None
    model_config = {"extra": "ignore"}
    # k: int = 30
    # min_sim: float | None = None
    # max_ctx_chars: int | None = None

# ------------ 路由 ------------
@app.get("/health")
def health():
    return {"ok": app.state.qa is not None}

from fastapi import Request

@app.post("/ask")
async def ask(body: AskBody, request: Request):
    print("RAW:", await request.body())
    if not body.question.strip():
        raise HTTPException(400, "question is required")

    if app.state.qa is None:
        # 后端没初始化成功时给出友好提示
        raise HTTPException(
            500,
            "QA engine not initialized. Check GOOGLE_APPLICATION_CREDENTIALS / OPENAI_API_KEY / Firestore."
        )

    # 每次请求可临时覆盖相似度阈值与上下文长度
    qa = app.state.qa
    result = qa.answer(body.question, doc_id=body.doc_id)
    # 约定返回结构：{ answer, refs, hits }
    return result
