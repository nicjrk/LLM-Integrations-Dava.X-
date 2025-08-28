import os
import re
from pathlib import Path

import chromadb
from openai import OpenAI
from dotenv import load_dotenv

# --- Config & init ---
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# OpenAI client (ia cheia din env)
client = OpenAI()  # încarcă automat OPENAI_API_KEY din .env

# Chroma: folosește persistent storage (folder local "chroma_db/")
chroma_client = chromadb.PersistentClient(path=str(ROOT / "chroma_db"))
collection = chroma_client.get_or_create_collection(name="books")

# --- Load corpus ---
md_path = ROOT / "book_summaries.md"
if not md_path.exists():
    raise FileNotFoundError(f"Nu găsesc {md_path}. Rulează scriptul din folderul proiectului sau pune fișierul acolo.")

text = md_path.read_text(encoding="utf-8")

# Regex: titlu + bloc până la următorul titlu
pattern = r"## Title: (.+?)\n(.*?)(?=\n## Title:|\Z)"
matches = re.findall(pattern, text, flags=re.S)

if not matches:
    raise ValueError("Nu am găsit intrări. Verifică formatul: '## Title: <titlu>\\n<rezumat>\\n**Teme:** ...'")

# --- Indexare în Chroma ---
# Dacă rerulezi scriptul, evită dubluri: goliți colecția sau folosește ID-uri unice
existing_count = collection.count()

docs, metas, ids = [], [], []
for i, (title, summary) in enumerate(matches):
    doc = summary.strip()
    docs.append(doc)
    metas.append({"title": title})
    ids.append(f"{existing_count + i}")  # id-uri continue

# Creează embeddings cu un singur call (mai eficient)
embs = client.embeddings.create(
    input=docs,
    model="text-embedding-3-small"
).data
vectors = [row.embedding for row in embs]

collection.add(ids=ids, embeddings=vectors, documents=docs, metadatas=metas)
print(f"Încărcat {len(docs)} documente (total în colecție: {collection.count()}).")

# --- Test rapid de căutare semantică ---
def search(query: str, top_k: int = 3):
    q_emb = client.embeddings.create(input=query, model="text-embedding-3-small").data[0].embedding
    res = collection.query(query_embeddings=[q_emb], n_results=top_k, include=["documents", "metadatas"])
    print(f"\n Query: {query}\n")
    for rank, (doc, meta) in enumerate(zip(res["documents"][0], res["metadatas"][0]), start=1):
        title = meta.get("title", "N/A")
        preview = doc.replace("\n", " ")[:200]
        print(f"{rank}. {title}\n   {preview}...\n")

# Exemplu de query; 
search("vreau o carte despre prietenie și magie")
