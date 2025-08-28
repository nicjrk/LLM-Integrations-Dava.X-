import os
import re
from pathlib import Path
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

# --- Setup ---
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Persistent Chroma (aceeași colecție creată la Pasul 2)
chroma = chromadb.PersistentClient(path=str(ROOT / "chroma_db"))
col = chroma.get_or_create_collection("books")

# ---- Sursa locală pentru tool: citim din book_summaries.md ----
md_path = ROOT / "book_summaries.md"
TEXT = md_path.read_text(encoding="utf-8")

# Extrage toate intrările ca (title, content)
ENTRIES = re.findall(r"## Title: (.+?)\n(.*?)(?=\n## Title:|\Z)", TEXT, flags=re.S)
BOOK_MAP = {title.strip(): content.strip() for title, content in ENTRIES}

def get_summary_by_title(title: str) -> str:
    """Întoarce rezumatul complet pentru titlul exact (din fișierul local)."""
    return BOOK_MAP.get(title, "Nu am găsit rezumatul pentru acest titlu în corpus.")

# ---- Definim tool-ul pentru OpenAI (function calling) ----
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_summary_by_title",
            "description": "Returnează rezumatul complet pentru titlul exact al unei cărți.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Titlul exact al cărții"}
                },
                "required": ["title"],
                "additionalProperties": False
            },
        },
    }
]

SYSTEM_PROMPT = (
    "Ești Smart Librarian. Primești o întrebare despre preferințe de lectură.\n"
    "- Ai primit candidați (titluri) dintr-un vector store local.\n"
    "- Alege UN TITLU din listă (cel mai potrivit întrebării) și explică pe scurt de ce.\n"
    "- Apoi, apelează tool-ul get_summary_by_title(title) CU TITLUL EXACT ales.\n"
    "- NU inventa titluri care nu sunt în listă."
)

def retrieve_candidates(query: str, top_k: int = 4):
    """Caută semantic: întoarce listă [(title, preview-doc), ...]."""
    q_emb = client.embeddings.create(input=query, model="text-embedding-3-small").data[0].embedding
    res = col.query(query_embeddings=[q_emb], n_results=top_k, include=["documents", "metadatas"])
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    return [(m["title"], d) for m, d in zip(metas, docs)]

def run_chat(query: str):
    # 1) RAG: căutăm candidații
    cands = retrieve_candidates(query, top_k=4)
    cand_titles = [t for t, _ in cands]

    if not cands:
        print(f"Ne pare rau, dar nu am gasit nicio carte relevanta pentru: \"{query}\".\n")
        return

    # 2) Primul apel LLM cu tool-uri înregistrate
    user_prompt = (
        f"Întrebare utilizator: {query}\n\n"
        f"Candidați disponibili (alege doar din aceștia): {cand_titles}\n"
        f"Răspunde scurt: recomandarea + de ce. Apoi APELEAZĂ tool-ul cu titlul exact."
    )

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.3,
    )

    msg = completion.choices[0].message

    # 3) Dacă modelul vrea să apeleze tool-ul, îl executăm local și facem al doilea apel
    if msg.tool_calls:
        tool_call = msg.tool_calls[0]  # așteptăm un singur tool call
        if tool_call.function.name == "get_summary_by_title":
            import json
            args = json.loads(tool_call.function.arguments or "{}")
            title = args.get("title", "")
            summary = get_summary_by_title(title)

            # Al 2-lea apel: trimitem rezultatul tool-ului înapoi modelului ca "role=tool"
            final = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    msg,  # mesajul cu tool_calls
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": "get_summary_by_title",
                        "content": summary,
                    },
                ],
                temperature=0.3,
            )
            answer = final.choices[0].message.content
            print("\n--- Răspunsul asistentului ---\n")
            print(answer)
            return
    else:
        # Fallback: dacă nu apelează tool-ul, măcar dăm recomandarea
        print("\n--- Recomandare (fără tool) ---\n")
        print(msg.content)

def main():
    print("Smart Librarian CLI — scrie 'exit' ca să ieși.\n")
    while True:
        q = input("Tu: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        run_chat(q)

if __name__ == "__main__":
    main()
