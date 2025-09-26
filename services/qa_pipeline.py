# rag_service.py
import os
from typing import Dict, Any, List
from django.db import transaction

from dotenv import load_dotenv
load_dotenv()  # loads .env at project root

from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec  # <-- v3 client

from ResearchApp.models import Study, StudyDocument, ChatSession, ChatMessage

# ---------- ENV / CONST ----------
INDEX_NAME = os.getenv("PINECONE_INDEX", "psygen-studies")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))  # text-embedding-3-small
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_METRIC = os.getenv("PINECONE_METRIC", "cosine")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")         # e.g., "aws"
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1") # e.g., "us-east-1"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

print("PINECONE_API_KEY:", "set" if PINECONE_API_KEY else "NOT SET")

# ---------- Pinecone v3 helpers ----------
def _get_pc() -> Pinecone:
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is not set")
    return Pinecone(api_key=PINECONE_API_KEY)

def _ensure_index_exists():
    pc = _get_pc()
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    return pc

def _get_index():
    pc = _ensure_index_exists()
    return pc.Index(INDEX_NAME)

def _init_vectorstore():
    index = _get_index()  # pc.Index
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return PineconeVectorStore(index=index, embedding=embeddings)

# ---------- Utility formatters ----------
def _trim_text(s: str, max_chars: int = 8000) -> str:
    if not s:
        return ""
    return s if len(s) <= max_chars else s[:max_chars] + " …[truncated]"

def _format_history(history_qas: List[ChatMessage], max_chars: int = 4000) -> str:
    pieces = [f"Q: {m.question}\nA: {m.answer}" for m in history_qas]
    return _trim_text("\n".join(pieces), max_chars=max_chars)

def _sources_from_docs(docs) -> List[Dict[str, Any]]:
    out = []
    for d in docs:
        md = d.metadata or {}
        out.append({
            "study_id": md.get("study_id"),
            "title": md.get("title"),
            "pmid": md.get("pmid"),
            "DOI": md.get("DOI"),
            "journal_name": md.get("journal_name"),
            "year": md.get("year"),
            "countries": md.get("countries", []),
            "disorder": md.get("disorder", []),
            "article_type": md.get("article_type", []),
            "biological_modalities": md.get("biological_modalities", []),
            "genetic_source_materials": md.get("genetic_source_materials", []),
            "page": md.get("page"),
            "score": getattr(d, "score", None),
        })
    dedup = {}
    for s in out:
        key = (s.get("study_id"), s.get("page"))
        if key not in dedup or (s.get("score") or 0) > (dedup[key].get("score") or 0):
            dedup[key] = s
    return list(dedup.values())

def get_top_images(request, query_text: str) -> List[Dict[str, Any]]:
    return []

# ---------- Main RAG entry ----------
@transaction.atomic
def continue_chat_rag(
    request,
    email: str,
    question: str,
    session_id: int | None = None,
    filters: Dict[str, Any] | None = None,
) -> Dict[str, Any]:

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    vectorstore = _init_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 12,
            "score_threshold": 0.40,
            "filter": filters or None,
        },
    )

    # session
    if session_id:
        session = ChatSession.objects.filter(id=session_id).first()
        if not session:
            session = ChatSession.objects.create(email=email)
    else:
        session = ChatSession.objects.create(email=email)

    # history
    history_qas = session.messages.order_by("created_at")
    history_prompt = _format_history(history_qas)

    # retrieve
    docs = retriever.get_relevant_documents(question)

    # context cap
    max_context_chars = 8000
    context_chunks, running = [], 0
    for d in docs:
        t = d.page_content or ""
        if running + len(t) > max_context_chars:
            remaining = max_context_chars - running
            if remaining > 50:
                context_chunks.append(t[:remaining])
            break
        context_chunks.append(t)
        running += len(t)
    context = "\n".join(f"- {c}" for c in context_chunks)
    context_prompt = f"Relevant study context:\n{context}" if context else "Relevant study context:\n(none retrieved)"

    sources = _sources_from_docs(docs)
    summary_context = ""

    prompt = f"""
You are ỌpọlọAI, a psychiatric genomics research assistant. Answer strictly from the provided context (research study chunks).
- Do NOT fabricate facts.
- Prefer precise details (genes/variants/sample sizes/statistics) when present.
- If the answer is not clearly present, say: "This information is not explicitly available in the retrieved studies."
- Keep tone: scholarly, concise, and well-structured.
- Cite studies by PMID/DOI when relevant.
- follow-up questions sholud be informative and relevant to psychiatric genomics.


--- Previous Q&A ---
{history_prompt}

--- Retrieved Study Excerpts ---
{context_prompt}

--- Study Summaries ---
{summary_context}

Now answer:
Q: {question}
A:
""".strip()

    chat = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are ỌpọlọAI, a helpful research assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=500,
    )
    answer = chat.choices[0].message.content

    # suggestions
    try:
        sugg = client.chat_completions.create(  # or client.chat.completions.create, depending on your openai lib version
            model="gpt-4o",
            messages=[{"role": "user", "content": f"Based on this answer, suggest 3 concise follow-up research questions:\n\n{answer}"}],
            max_tokens=80,
            temperature=0.4,
        )
        suggestions = [s.strip("- •").strip() for s in sugg.choices[0].message.content.split("\n") if s.strip()]
        suggestions = suggestions[:3]
    except Exception:
        suggestions = []

    markers = [
        "not explicitly available", "no relevant study context", "unable to find",
        "no information found", "insufficient information",
    ]
    images = [] if any(m in (answer or "").lower() for m in markers) else get_top_images(request, question)

    ChatMessage.objects.create(
        session=session,
        question=question,
        answer=answer,
        image_results=images,
        source_studies=sources,
    )

    if not session.title:
        try:
            title_resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": f"Give a short 5-word title for this: {question}"}],
                max_tokens=10,
                temperature=0.3,
            )
            session.title = title_resp.choices[0].message.content
            session.save(update_fields=["title"])
        except Exception:
            pass

    return {
        "session_id": session.id,
        "title": session.title,
        "question": question,
        "answer": answer,
        "images": images,
        "sources": sources,
        "suggested_questions": suggestions,
    }
