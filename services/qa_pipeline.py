from openai import OpenAI
from ResearchApp.models import Study, StudyImage, ChatSession, ChatMessage
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF
import json

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
index = pc.Index("psygen-qa-index")
# retriever = SentenceTransformer("all-mpnet-base-v2")
retriever = SentenceTransformer("multi-qa-mpnet-base-dot-v1")



def get_top_images(request, query_embedding, max_results=8, similarity_threshold=0.6):
    images = StudyImage.objects.exclude(embedding__isnull=True)
    results = []

    for img in images:
        try:
            distance = cosine(query_embedding, img.embedding)
            if distance < similarity_threshold:
                results.append((distance, img))
        except:
            continue

    results.sort(key=lambda x: x[0])
    return [
        {
            "id": img.id,
            "caption": img.caption,
            "image_url": request.build_absolute_uri(img.image.url),
            "study_id": img.study.id,
            "study_title": img.study.title
        }
        for _, img in results[:max_results]
    ]


def extract_text_from_pdf(path):
    doc = fitz.open(path)
    text = "\n".join([page.get_text() for page in doc])
    doc.close()
    return text

def generate_structured_summary_for_study(study):
    if not hasattr(study, 'document'):
        return None

    text = study.document.extracted_text or extract_text_from_pdf(study.document.pdf_file.path)
    if not text or len(text) < 500:
        return None

    prompt = f"""
You are a summarization expert. Given this research paper content, extract and return a structured JSON summary with these sections:
- background
- methodology
- findings
- conclusion
- limitations

Text:
{text[:10000]}  # Truncate to avoid token limit

Respond in valid JSON only.
"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        summary_text = response.choices[0].message.content
        return json.loads(summary_text)
    except Exception as e:
        print(f"❌ Summary error for {study.title}: {e}")
        return None


# def continue_chat(request, email, question, session_id=None):
#     query_vector = retriever.encode(question).tolist()

#     # 🔍 Retrieve or create session
#     session = ChatSession.objects.filter(id=session_id).first() if session_id else ChatSession.objects.create(email=email)

#     # 📚 Collect prior Q&As as context
#     history = session.messages.order_by("created_at")
#     history_prompt = "\n".join([f"Q: {msg.question}\nA: {msg.answer}" for msg in history])

#     # 🔍 Retrieve context chunks from Pinecone
#     # results = index.query(vector=query_vector, top_k=3, include_metadata=True)
#     # context = "\n".join([f"- {match['metadata']['content']}" for match in results['matches']])
#     raw_results = index.query(vector=query_vector, top_k=20, include_metadata=True)

#     SIMILARITY_THRESHOLD = 0.4

#     filtered_matches = [
#     match for match in raw_results["matches"]
#     if match["score"] >= SIMILARITY_THRESHOLD]

#     context = "\n".join([
#     f"- {match['metadata']['content']}" for match in filtered_matches
#     ])

#     context_prompt = f"Relevant study context:\n{context}"


#     prompt = f"""
# You are ỌpọlọAI, a psychiatric genomics research assistant. You are to answer user questions strictly based on the context provided from the research studies below.

# Instructions:
# - Do not make up any facts not present in the context.
# - Use specific data, genetic findings, or participant details where available.
# - If the answer is not clearly in the context, say: "This information is not explicitly available in the retrieved studies."
# - Prioritize depth, precision, and research-style language.

# Previous questions and answers:
# {history_prompt}

# Relevant study context:
# {context_prompt}

# Now answer:
# Q: {question}
# A:"""


#     # 🔥 Generate answer with GPT-4o
#     response = openai_client.chat.completions.create(
#         model="gpt-4o",
#         messages=[
#             {"role": "system", "content": "You are ỌpọlọAI, a helpful research assistant."},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0.3,
#         max_tokens=400
#     )
#     answer = response.choices[0].message.content

#     # 🤖 Generate follow-up suggestions
#     try:
#         suggestion_response = openai_client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "user", "content": f"Based on this answer, suggest 2 relevant follow-up questions:\n\n{answer}"}
#             ],
#             max_tokens=30,
#             temperature=0.4
#         )
#         suggestions = suggestion_response.choices[0].message.content.strip().split("\n")
#         suggestions = [s.lstrip("-1234567890. ").strip() for s in suggestions if s.strip()]
#     except Exception:
#         suggestions = []


#     # 📦 Get source studies
#     # study_ids = {m["metadata"].get("study_id") for m in filtered_matches if "study_id" in m["metadata"]}
#     # # study_ids = {m["metadata"].get("study_id") for m in results["matches"] if "study_id" in m["metadata"]}
#     # matched_studies = Study.objects.filter(id__in=study_ids)
#     # sources = [
#     #     {
#     #         "id": s.id,
#     #         "title": s.title,
#     #         "journal": s.journal_name,
#     #         "year": s.year,
#     #         "doi": s.DOI,
#     #         "lead_author": s.lead_author,
#     #         "pdf_url": request.build_absolute_uri(s.document.pdf_file.url) if getattr(s, "document") and s.document.pdf_file else None
#     #     }
#     #     for s in matched_studies
#     # ]

#     # 📦 Get source studies
#     study_ids = {m["metadata"].get("study_id") for m in filtered_matches if "study_id" in m["metadata"]}
#     matched_studies = Study.objects.filter(id__in=study_ids)

#     sources = []
#     for s in matched_studies:
#         summary = generate_structured_summary_for_study(s)
#         sources.append({
#             "id": s.id,
#             "title": s.title,
#             "journal": s.journal_name,
#             "year": s.year,
#             "doi": s.DOI,
#             "lead_author": s.lead_author,
#             "pdf_url": request.build_absolute_uri(s.document.pdf_file.url) if getattr(s, "document") and s.document.pdf_file else None,
#             "summary": summary  # 🆕 this adds the structured summary
#         })


#     # 🖼️ Top related images
#     images = []
#     irrelevant_markers = [
#     "not explicitly available",
#     "no relevant study context",
#     "unable to find",
#     "no information found",
#     "based on the context provided, there is no",
#     "insufficient information"
#     ]
#     if not any(marker in answer.lower() for marker in irrelevant_markers):
#         images = get_top_images(request, query_vector)

#     # 💾 Save message
#     ChatMessage.objects.create(
#         session=session,
#         question=question,
#         answer=answer,
#         image_results=images,
#         source_studies=sources
#     )

#     # 🏷️ Title the session if it's empty
#     if not session.title:
#         try:
#             short_title = openai_client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=[{"role": "user", "content": f"Give a short 5-word title for this: {question}"}],
#                 max_tokens=10,
#                 temperature=0.3
#             )
#             session.title = short_title.choices[0].message.content
#             session.save()
#         except:
#             pass

#     return {
#         "session_id": session.id,
#         "title": session.title,
#         "question": question,
#         "answer": answer,
#         "images": images,
#         "sources": sources,
#         "suggested_questions": suggestions
#     }

def continue_chat(request, email, question, session_id=None):
    query_vector = retriever.encode(question).tolist()

    # 🔍 Retrieve or create session
    session = ChatSession.objects.filter(id=session_id).first() if session_id else ChatSession.objects.create(email=email)

    # 📚 Collect prior Q&As as context
    history = session.messages.order_by("created_at")
    history_prompt = "\n".join([f"Q: {msg.question}\nA: {msg.answer}" for msg in history])

    # 🔍 Retrieve context chunks from Pinecone
    raw_results = index.query(vector=query_vector, top_k=10, include_metadata=True)
    SIMILARITY_THRESHOLD = 0.4
    filtered_matches = [match for match in raw_results["matches"] if match["score"] >= SIMILARITY_THRESHOLD]

    context = "\n".join([f"- {match['metadata']['content']}" for match in filtered_matches])
    context_prompt = f"Relevant study context:\n{context}"

    # 📦 Get source studies
    study_ids = {m["metadata"].get("study_id") for m in filtered_matches if "study_id" in m["metadata"]}
    matched_studies = Study.objects.filter(id__in=study_ids)

    sources = []
    for s in matched_studies:
        summary = generate_structured_summary_for_study(s)
        sources.append({
            "id": s.id,
            "title": s.title,
            "journal": s.journal_name,
            "year": s.year,
            "doi": s.DOI,
            "lead_author": s.lead_author,
            "pdf_url": request.build_absolute_uri(s.document.pdf_file.url) if getattr(s, "document") and s.document.pdf_file else None,
            "summary": summary
        })

    # ✨ Create combined prompt
    summary_context = "\n\n".join([
        f"""STUDY: {s['title']}\nBackground: {s['summary'].get('background', 'N/A')}\nMethodology: {s['summary'].get('methodology', 'N/A')}\nFindings: {s['summary'].get('findings', 'N/A')}\nConclusion: {s['summary'].get('conclusion', 'N/A')}\nLimitations: {s['summary'].get('limitations', 'N/A')}"""
        for s in sources if s.get("summary")
    ])

    prompt = f"""
You are ỌpọlọAI, a psychiatric genomics research assistant. You are to answer user questions strictly based on the context provided from the research studies below.

Instructions:
- Do not make up any facts not present in the context.
- Use specific data, genetic findings, or participant details where available.
- If the answer is not clearly in the context, say: \"This information is not explicitly available in the retrieved studies.\"
- Prioritize depth, precision, and research-style language.

--- Previous Q&A ---
{history_prompt}

--- Retrieved Study Excerpts ---
{context_prompt}

--- Study Summaries ---
{summary_context}

Now answer:
Q: {question}
A:"""

    # 🔥 Generate answer with GPT-4o
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are ỌpọlọAI, a helpful research assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=400
    )
    answer = response.choices[0].message.content

    # 🤖 Generate follow-up suggestions
    try:
        suggestion_response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": f"Based on this answer, suggest 3 relevant follow-up questions:\n\n{answer}"}
            ],
            max_tokens=60,
            temperature=0.4
        )
        suggestions = suggestion_response.choices[0].message.content.strip().split("\n")
        suggestions = [s.lstrip("-1234567890. ").strip() for s in suggestions if s.strip()]
    except Exception:
        suggestions = []

    # 🖼️ Top related images
    images = []
    irrelevant_markers = [
        "not explicitly available",
        "no relevant study context",
        "unable to find",
        "no information found",
        "based on the context provided, there is no",
        "insufficient information"
    ]
    if not any(marker in answer.lower() for marker in irrelevant_markers):
        images = get_top_images(request, query_vector)

    # 💾 Save message
    ChatMessage.objects.create(
        session=session,
        question=question,
        answer=answer,
        image_results=images,
        source_studies=sources
    )

    # 🏷️ Title the session if it's empty
    if not session.title:
        try:
            short_title = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": f"Give a short 5-word title for this: {question}"}],
                max_tokens=10,
                temperature=0.3
            )
            session.title = short_title.choices[0].message.content
            session.save()
        except:
            pass

    return {
        "session_id": session.id,
        "title": session.title,
        "question": question,
        "answer": answer,
        "images": images,
        "sources": sources,
        "suggested_questions": suggestions
    }






