# Configuration
You can configure the project in two ways:  
- Option A (Recommended): Environment variables
- Option B: In-code arguments.

## Option A — Environment variables (recommended)
```
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\...\firebase-sa.json"
$env:OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxx"
```
## Option B — Explicit in code
Pass secrets directly to the classes.
GO to main.py change:
```
# 1) OpenAI key provided explicitly to the embedder
embedder = OpenAIEmbedder(
    model="text-embedding-3-small",
    api_key="sk-xxxxxxxxxxxxxxxx",            # example only
)
# 2) Firestore service account provided explicitly to the store
store = FirestoreChunkStore(
    collection="chunks",
    cred_source=r"C:\...\firebase-sa.json", # or a Python dict with the JSON contents
    project_id="your-gcp-project-id",          # optional; usually inferred from the SA file
)
```

# Install dependencies from requirements.txt
```
pip install -r requirements.txt
```

# Use main.py
