# src/rag_utils.py
# Retrieval-Augmented Generation utils for Prototype 3
# - Uses OpenAI Embeddings + FAISS if available.
# - Falls back to TF-IDF (scikit-learn) if not.
# - Optional LLM answer via OpenAI Chat (fallback: show retrieved context).

import os, json
from pathlib import Path
from typing import List, Dict

# Optional deps
try:
    import faiss  # type: ignore
except Exception:
    faiss = None

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity       # type: ignore
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None

ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / "rag" / "kb"
ART = ROOT / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

EMBED_INDEX = ART / "kb_index.faiss"
EMBED_META  = ART / "kb_chunks.json"
TFIDF_VECT  = ART / "kb_tfidf.pkl"
TFIDF_MAT   = ART / "kb_tfidf.npy"

CHUNK_SIZE = 700
OVERLAP = 100

def _maybe_seed_kb():
    """Create a minimal KB if none exists (so the tab works out-of-the-box)."""
    KB_DIR.mkdir(parents=True, exist_ok=True)
    if any(KB_DIR.glob("*.md")):
        return
    (KB_DIR / "mpa.md").write_text(
        "# Marine Protected Areas (MPAs)\n"
        "Eco-safe planning minimizes traffic through MPAs when feasible, to reduce disturbance to wildlife.\n"
    )
    (KB_DIR / "hazards.md").write_text(
        "# Navigation Hazards\n"
        "Storms, high waves, poor visibility, ice, debris, crowded lanes, GPS interference.\n"
    )
    (KB_DIR / "algorithms.md").write_text(
        "# Algorithms for Safe & Eco-Aware Navigation\n"
        "- A*: distance + hazard + MPA penalties\n"
        "- RL (Q-learning): learns from rewards\n"
        "- Transformer policy: imitation of A* expert\n"
    )

def _chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(L, start + chunk_size)
        chunks.append(text[start:end])
        if end == L:
            break
        start = max(0, end - overlap)
    return chunks

def _load_kb_files():
    _maybe_seed_kb()
    docs = []
    for p in sorted(KB_DIR.glob("*.md")):
        docs.append((p.name, p.read_text()))
    return docs

def _build_index_openai(model="text-embedding-3-small"):
    if OpenAI is None or faiss is None:
        raise RuntimeError("OpenAI/FAISS not available")
    client = OpenAI()
    metas, texts = [], []
    for fname, txt in _load_kb_files():
        for i, ch in enumerate(_chunk_text(txt)):
            metas.append({"source": fname, "chunk_id": i})
            texts.append(ch)
    # Embed in batches
    import numpy as np
    vecs = []
    B = 64
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        emb = client.embeddings.create(model=model, input=batch).data
        vecs.extend([np.array(e.embedding, dtype="float32") for e in emb])
    mat = np.vstack(vecs)
    # Cosine via inner-product on normalized vectors
    mat = mat / ( (mat**2).sum(axis=1, keepdims=True)**0.5 + 1e-12 )
    idx = faiss.IndexFlatIP(mat.shape[1])
    idx.add(mat.astype("float32"))
    faiss.write_index(idx, str(EMBED_INDEX))
    EMBED_META.write_text(json.dumps({"metas": metas, "texts": texts}, indent=2))
    return "faiss"

def _build_index_tfidf():
    if TfidfVectorizer is None:
        raise RuntimeError("scikit-learn not installed for TF-IDF fallback")
    metas, texts = [], []
    for fname, txt in _load_kb_files():
        for i, ch in enumerate(_chunk_text(txt)):
            metas.append({"source": fname, "chunk_id": i})
            texts.append(ch)
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(texts).toarray()
    import joblib, numpy as np
    joblib.dump(vec, TFIDF_VECT)
    np.save(TFIDF_MAT, X)
    EMBED_META.write_text(json.dumps({"metas": metas, "texts": texts}, indent=2))
    return "tfidf"

def ensure_index() -> str:
    """Returns 'faiss' or 'tfidf' depending on what is ready/possible."""
    if EMBED_INDEX.exists() and EMBED_META.exists():
        return "faiss"
    if TFIDF_VECT.exists() and TFIDF_MAT.exists() and EMBED_META.exists():
        return "tfidf"
    # Try FAISS+OpenAI first, then TF-IDF
    try:
        return _build_index_openai()
    except Exception:
        return _build_index_tfidf()

def retrieve(query: str, k: int = 4) -> List[Dict]:
    kind = ensure_index()
    meta = json.loads(EMBED_META.read_text())
    texts, metas = meta["texts"], meta["metas"]
    import numpy as np
    if kind == "faiss":
        client = OpenAI()
        q = client.embeddings.create(model="text-embedding-3-small", input=[query]).data[0].embedding
        q = np.array(q, dtype="float32")
        q = q / (np.linalg.norm(q) + 1e-12)
        idx = faiss.read_index(str(EMBED_INDEX))
        D, I = idx.search(q.reshape(1, -1), k)
        hits = []
        for score, i in zip(D[0], I[0]):
            hits.append({"text": texts[int(i)], "meta": metas[int(i)], "score": float(score)})
        return hits
    else:
        import joblib
        vec = joblib.load(TFIDF_VECT)
        X = __import__("numpy").load(TFIDF_MAT)
        qv = vec.transform([query]).toarray()
        sims = cosine_similarity(qv, X)[0]
        idx = sims.argsort()[::-1][:k]
        return [{"text": texts[int(i)], "meta": metas[int(i)], "score": float(sims[int(i)])} for i in idx]

_SYSTEM_PROMPT = (
    "You are a navigation safety assistant. Answer concisely using the retrieved CONTEXT. "
    "Focus on distance, hazard, and ecological impact. If unsure, say so. Cite sources (file names)."
)

def answer_llm(query: str, hits: List[Dict]) -> str:
    """Calls OpenAI Chat if available; otherwise, returns a concise extractive fallback."""
    ctx = "\n\n".join([f"[{h['meta']['source']} #{h['meta']['chunk_id']}] {h['text']}" for h in hits])
    try:
        client = OpenAI()  # requires OPENAI_API_KEY
        model = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": _SYSTEM_PROMPT},
                      {"role": "user",   "content": f"CONTEXT:\n{ctx}\n\nQUESTION: {query}"}],
            temperature=0.2,
            max_tokens=500
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        # Fallback text if no OpenAI or no key
        return ("LLM not available. Showing retrieved context instead:\n\n"
                + ctx[:1200])
