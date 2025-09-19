# rag_service.py
import os
from typing import Dict, Any, List
from django.db import transaction
from django.utils import timezone

from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import pinecone

from .models import Study, StudyDocument, ChatSession, ChatMessage

INDEX_NAME = os.getenv("PINECONE_INDEX", "psygen-studies")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # e.g. "gcp-starter" / region string

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# you already wrote this for indexing; re-use same keys
client = OpenAI(api_key=OPENAI_API_KEY)

def _init_vectorstore():
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    index = pinecone.Index(INDEX_NAME)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return PineconeVectorStore(index=index, embedding=embeddings)

def _trim_text(s: str, max_chars: int = 8000) -> str:
    if not s:
        return ""
    return s if len(s) <= max_chars else s[:max_chars] + " …[truncated]"

def _format_history(history_qas: List[ChatMessage], max_chars: int = 4000) -> str:
    pieces = []
    for msg in history_qas:
        pieces.append(f"Q: {msg.question}\nA: {msg.answer}")
    joined = "\n".join(pieces)
    return _trim_text(joined, max_chars=max_chars)

def _sources_from_docs(docs) -> List[Dict[str, Any]]:
    """Flatten relevant metadata for UI display and citation."""
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
            "score": getattr(d, "score", None),  # set by retriever when available
        })
    # de-dup by (study_id,page) while keeping highest score
    dedup = {}
    for s in out:
        key = (s.get("study_id"), s.get("page"))
        if key not in dedup or (s.get("score") or 0) > (dedup[key].get("score") or 0):
            dedup[key] = s
    return list(dedup.values())

def get_top_images(request, query_text: str) -> List[Dict[str, Any]]:
    """Stub: hook your existing image search; kept simple for now."""
    # You can keep your existing vector image search; here we just return empty
    return []

@transaction.atomic
def continue_chat_rag(
    request,
    email: str,
    question: str,
    session_id: int | None = None,
    # optional metadata filters you may pass from UI:
    filters: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    LangChain-based retrieval + OpenAI generation. No manual .encode() or raw index.query().
    """
    vectorstore = _init_vectorstore()

    # Build a retriever with score threshold (similarity)
    # Use "similarity_score_threshold" to apply threshold, and raise k/fetch_k for recall
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 12,                    # return up to 12 chunks
            "score_threshold": 0.40,    # your previous threshold
            # Optional: metadata filters, e.g. {"year": {"$gte": 2018}}
            "filter": filters or None
        },
    )

    # Session
    if session_id:
        session = ChatSession.objects.filter(id=session_id).first()
        if not session:
            session = ChatSession.objects.create(email=email)
    else:
        session = ChatSession.objects.create(email=email)

    # History (for memory in prompt only; we still do RAG fresh each question)
    history_qas = session.messages.order_by("created_at")
    history_prompt = _format_history(history_qas)

    # Retrieve documents
    docs = retriever.get_relevant_documents(question)  # returns LangChain Documents with .page_content & .metadata

    # Build context (cap the context to avoid overlong prompts)
    max_context_chars = 8000
    context_chunks = []
    running = 0
    for d in docs:
        t = d.page_content or ""
        if running + len(t) > max_context_chars:
            # add what fits
            remaining = max_context_chars - running
            if remaining > 50:
                context_chunks.append(t[:remaining])
                running = max_context_chars
            break
        context_chunks.append(t)
        running += len(t)
    context = "\n".join(f"- {c}" for c in context_chunks)
    context_prompt = f"Relevant study context:\n{context}" if context else "Relevant study context:\n(none retrieved)"

    # Sources list for UI
    sources = _sources_from_docs(docs)

    # (Optional) summaries section — if you have precomputed summaries, inject here; else leave empty
    summary_context = ""

    # Compose final prompt
    prompt = f"""
You are ỌpọlọAI, a psychiatric genomics research assistant. Answer strictly from the provided context (research study chunks). 
- Do NOT fabricate facts.
- Prefer precise details (genes/variants/sample sizes/statistics) when present.
- If the answer is not clearly present, say: "This information is not explicitly available in the retrieved studies."
- Keep tone: scholarly, concise, and well-structured.

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

    # Generate
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

    # Suggestions
    try:
        sugg = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": f"Based on this answer, suggest 3 concise follow-up research questions:\n\n{answer}"}],
            max_tokens=80,
            temperature=0.4,
        )
        suggestions = [s.strip("- •").strip() for s in sugg.choices[0].message.content.split("\n") if s.strip()]
        suggestions = [s for s in suggestions if s][:3]
    except Exception:
        suggestions = []

    # Images (skip if answer is clearly "no info")
    markers = [
        "not explicitly available",
        "no relevant study context",
        "unable to find",
        "no information found",
        "insufficient information",
    ]
    images = [] if any(m in (answer or "").lower() for m in markers) else get_top_images(request, question)

    # Save message
    ChatMessage.objects.create(
        session=session,
        question=question,
        answer=answer,
        image_results=images,
        source_studies=sources,
    )

    # Title the session if empty
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
