import os
from pdfreader import OpenAIEmbedder, FirestoreChunkStore, PDFIngestor, RAGQA
# 1) Initialize the storage and embedder
api_key="sk-xxx"
embedder = OpenAIEmbedder(model="text-embedding-3-small", api_key=api_key)
store = FirestoreChunkStore(collection="chunks",cred_source="../modelworks-firebase.json")

# If you want to ingest a new PDF file, uncomment the following lines:
# 2) Take in a PDF (slice → Embed → Write to Firestore)
# ingestor = PDFIngestor(embedder, store,
#                        chunk_size=800, overlap=120,
#                        embed_batch=96,
#                        meta_backend="openai", meta_model="text-embedding-3-small")
# n = ingestor.ingest("C:/Users/10266/Desktop/DeepSeek 2.pdf")
# print("ingested chunks:", n)

# 3) Initialize the QA engine (OpenAI )
qa = RAGQA(store, embedder, gen_model="gpt-4o-mini", min_sim=0.15)
#   （OR other supported models - Joey ）
# qa = RAGQA(store, embedder, gen_model="Joey",
#            base_url="http://13.239.88.166:8000/v1", api_key="EMPTY", min_sim=0.15)
# 4) Ask a question (If only a certain document is searched, pass the doc_id; if not, it equals the entire database)
res = qa.answer("what is DeekSeek?", doc_id=None, k=5)
print("—— Answer ——")
print(res["answer"])
print("\n—— Reference  ——")
print(", ".join(res["refs"]))
# To view the hit segment and score:
for m, s in res["hits"]:
    print(round(s, 4), m.get("source"), "p", m.get("page"))