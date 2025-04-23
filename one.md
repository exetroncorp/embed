qemu-system-x86_64 -m 5G -smp 1 -device virtio-net-pci,netdev=net0 -netdev user,id=net0,hostfwd=tcp::8080-:8080  -drive file=k.raw,format=raw,cache=writeback -virtfs local,path=/home/gitpod,security_model=mapped-xattr,mount_tag=hostshare,multidevs=remap  -nographic  -monitor  telnet:127.0.0.1:4444,server,nowait


@RadFromOrange ➜ /workspaces/stapi (main) $ ls /home/codespace/.cache/huggingface/hub/
models--intfloat--e5-small-v2                     models--sentence-transformers--all-MiniLM-L6-v2
models--intfloat--multilingual-e5-large           models--sentence-transformers--all-mpnet-base-v2
models--jinaai--jina-bert-implementation          models--sentence-transformers--all-roberta-large-v1
models--jinaai--jina-embeddings-v2-base-en        models--sentence-transformers--msmarco-MiniLM-L6-cos-v5
models--nomic-ai--nomic-bert-2048                 models--sentence-transformers--multi-qa-MiniLM-L6-cos-v1
models--nomic-ai--nomic-embed-text-v1             models--sentence-transformers--paraphrase-MiniLM-L3-v2
models--nomic-ai--nomic-embed-text-v2-moe         models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2
models--sentence-transformers--all-MiniLM-L12-v2  models--thenlper--gte-small


Clone the repo:
```bash
git clone https://github.com/substratusai/stapi
cd stapi
```

Install dependencies:
```bash
pip3 install -r requirements.txt
```

Run the webserver:
```bash
uvicorn main:app --port 8080 --reload




Great questions! Let's clarify and then give you the full working setup.

---

## ❓ Is it the same as OpenAI embedding API?

**No.**  
This is a local alternative to the OpenAI `/v1/embeddings` API. It's **not the same**, but **it does the same job** — it takes text input and returns a list of float embeddings.

If you want, I can adjust the API to **mimic OpenAI's exact format** (`model`, `input`, etc.).

---

## ✅ Plan:
1. Use **FastAPI** to expose the `/embedding` endpoint.
2. Use a **very small, fast embedding model** from HuggingFace.
3. Add an automatic **model download script** (if `toto` doesn't exist).
4. Keep it **super lightweight** — no GPU needed, runs anywhere.

---

## ✅ Model: `all-MiniLM-L6-v2`

This is one of the **lightest and best general-purpose embedding models** (~80MB).

---

## ✅ Folder Structure
```
.
├── app.py           <-- Main API server
├── download_model.py <-- One-time setup script
├── toto/            <-- Auto-created by script if not present
```

---

## 🧠 `download_model.py`

```python
from sentence_transformers import SentenceTransformer

print("Downloading model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model.save('toto')
print("Model saved to 'toto'")
```

Run this once:
```bash
python download_model.py
```

---

## 🚀 `app.py`

```python
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import os

MODEL_PATH = "toto"

# Auto-download if model doesn't exist
if not os.path.exists(MODEL_PATH):
    from download_model import model  # triggers save
else:
    model = SentenceTransformer(MODEL_PATH)

app = FastAPI()

class EmbedRequest(BaseModel):
    text: str

@app.post("/embedding")
async def get_embedding(req: EmbedRequest):
    embedding = model.encode(req.text).tolist()
    return {"embedding": embedding}
```

---

## 🟢 Run the server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## 🔁 Optional: Match OpenAI API Signature

If you want to match OpenAI’s style like this:

```json
{
  "input": "hello world",
  "model": "local-mini"
}
```

Just let me know, and I’ll rewrite the endpoint to mimic it.

---

Want me to zip all this into a single ready-to-run project for you?


Certainly! To expose your locally cached embedding and reranking models via a RESTful API using FastAPI, you can set up endpoints that allow clients to request embeddings or rerankings for given inputs. Below is a comprehensive guide to achieve this.

---

## 🧠 Prerequisites

Ensure you have the necessary packages installed:

```bash
pip install fastapi uvicorn sentence-transformers
```

---

## 🚀 FastAPI Server Implementation

Here's a FastAPI application that exposes two endpoints:

- **`/embed`**: Accepts a list of texts and returns their embeddings.
- **`/rerank`**: Accepts a query and a list of documents, returning the documents ranked by relevance to the query.

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder

app = FastAPI()

# Load embedding models
embedding_models = {
    "default": SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
    # Add other models as needed
}

# Load reranking models
reranking_models = {
    "default": CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2"),
    # Add other models as needed
}

class EmbedRequest(BaseModel):
    texts: List[str]
    model: Optional[str] = "default"

class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    model: Optional[str] = "default"

@app.post("/embed")
def embed(request: EmbedRequest):
    model = embedding_models.get(request.model)
    if not model:
        raise HTTPException(status_code=404, detail="Embedding model not found")
    embeddings = model.encode(request.texts, convert_to_tensor=False)
    return {"embeddings": embeddings}

@app.post("/rerank")
def rerank(request: RerankRequest):
    model = reranking_models.get(request.model)
    if not model:
        raise HTTPException(status_code=404, detail="Reranking model not found")
    pairs = [(request.query, doc) for doc in request.documents]
    scores = model.predict(pairs)
    ranked_docs = sorted(zip(request.documents, scores), key=lambda x: x[1], reverse=True)
    return {"reranked_documents": ranked_docs}
```

---

## 🧪 Testing the API

Run the server:

```bash
uvicorn your_script_name:app --reload
```

Replace `your_script_name` with the name of your Python file (without the `.py` extension).

### Example Requests

**Embedding:**

```bash
curl -X POST "http://127.0.0.1:8000/embed" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Hello world", "FastAPI is great!"], "model": "default"}'
```

**Reranking:**

```bash
curl -X POST "http://127.0.0.1:8000/rerank" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is FastAPI?", "documents": ["FastAPI is a web framework.", "It is used for building APIs."], "model": "default"}'
```

---

## 📚 Additional Resources

For more detailed guidance on deploying machine learning models with FastAPI, consider the following resources:

- [Deploying ML Models as API using FastAPI | GeeksforGeeks](https://www.geeksforgeeks.org/deploying-ml-models-as-api-using-fastapi/)
- [Using FastAPI to Build Python Web APIs – Real Python](https://realpython.com/fastapi-python-web-apis/)

These tutorials provide step-by-step instructions and best practices for building and deploying APIs with FastAPI.

---

If you need further assistance or customization, feel free to ask! 
