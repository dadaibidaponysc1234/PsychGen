from openai import OpenAI
from ResearchApp.models import Study, StudyImage, ChatSession, ChatMessage
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
load_dotenv()

import os

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))

index = pc.Index("psygen-qa-index")
retriever = SentenceTransformer("all-mpnet-base-v2")


def get_top_images(query_embedding, max_results=8, similarity_threshold=0.6):
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
            "image_url": img.image.url,
            "study_id": img.study.id,
            "study_title": img.study.title
        }
        for _, img in results[:max_results]
    ]


def continue_chat(email, question, session_id=None):
    query_vector = retriever.encode(question).tolist()

    # 🔍 Retrieve or create session
    session = ChatSession.objects.filter(id=session_id).first() if session_id else ChatSession.objects.create(email=email)

    # 📚 Collect prior Q&As as context
    history = session.messages.order_by("created_at")
    history_prompt = "\n".join([f"Q: {msg.question}\nA: {msg.answer}" for msg in history])

    # 🔍 Retrieve context chunks from Pinecone
    results = index.query(vector=query_vector, top_k=3, include_metadata=True)
    context = "\n".join([f"- {match['metadata']['content']}" for match in results['matches']])
    context_prompt = f"Relevant study context:\n{context}"

    # 🧠 Final GPT prompt
    prompt = f"""
{history_prompt}

{context_prompt}

Now answer the new question:
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

    # 📦 Get source studies
    study_ids = {m["metadata"].get("study_id") for m in results["matches"] if "study_id" in m["metadata"]}
    matched_studies = Study.objects.filter(id__in=study_ids)
    sources = [
        {
            "id": s.id,
            "title": s.title,
            "journal": s.journal_name,
            "year": s.year,
            "doi": s.DOI,
            "lead_author": s.lead_author,
            "pdf_url": s.document.pdf_file.url if hasattr(s, "document") and s.document.pdf_file else None
        }
        for s in matched_studies
    ]

    # 🖼️ Top related images
    images = get_top_images(query_vector)

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
        "sources": sources
    }

# Initialize clients
# openai_client = OpenAI(api_key=OPENAI_API_KEY)
# pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
# index = pc.Index("psygen-qa-index")

# # Initialize retriever (sentence transformer model)
# retriever = SentenceTransformer("all-mpnet-base-v2")


# def get_top_images(query_embedding, max_results=8, similarity_threshold=0.6):
#     """
#     Return up to `max_results` images whose captions are semantically similar to the question.
#     Return an empty list if no matches pass the threshold.
#     """
#     all_images = StudyImage.objects.exclude(embedding__isnull=True)
#     scored_images = []

#     for img in all_images:
#         try:
#             distance = cosine(query_embedding, img.embedding)
#             if distance < similarity_threshold:  # lower is better
#                 scored_images.append((distance, img))
#         except Exception:
#             continue

#     scored_images.sort(key=lambda x: x[0])  # Best match first
#     top_images = [img for _, img in scored_images[:max_results]]

#     return [
#         {
#             "id": img.id,
#             "caption": img.caption,
#             "image_url": img.image.url,
#             "study_id": img.study.id,
#             "study_title": img.study.title
#         }
#         for img in top_images
#     ]


# def ask_question(query, top_k=3):
#     # Step 1: Embed the user question
#     query_vector = retriever.encode(query).tolist()

#     # Step 2: Query Pinecone for similar study chunks
#     results = index.query(
#         vector=query_vector,
#         top_k=top_k,
#         include_metadata=True
#     )

#     # Step 3: Collect relevant context and source tracking
#     context_chunks = []
#     study_ids = set()

#     for match in results["matches"]:
#         context_chunks.append(match["metadata"]["content"])
#         if "study_id" in match["metadata"]:
#             study_ids.add(match["metadata"]["study_id"])

#     context = "\n".join([f"- {chunk}" for chunk in context_chunks])
#     prompt = f"""You are ỌpọlọAI, a helpful AI assistant trained on African psychiatric genomics research.

# Context:
# {context}

# Question: {query}
# Answer:"""

#     # Step 4: Use GPT-4o to generate a natural language answer
#     response = openai_client.chat.completions.create(
#         model="gpt-4o",
#         messages=[
#             {"role": "system", "content": "You are ỌpọlọAI, a research assistant specialized in African psychiatric genomics."},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0.3,
#         max_tokens=400
#     )

#     answer = response.choices[0].message.content

#     # Step 5: Gather source study metadata
#     matched_studies = Study.objects.filter(id__in=study_ids)
#     sources = []
#     for study in matched_studies:
#         sources.append({
#             "id": study.id,
#             "title": study.title,
#             "journal": study.journal_name,
#             "year": study.year,
#             "doi": study.DOI,
#             "lead_author": study.lead_author,
#             "pdf_url": (
#                 study.document.pdf_file.url
#                 if hasattr(study, "document") and study.document.pdf_file
#                 else None
#             )
#         })

#     # Step 6: Retrieve up to 5 related images
#     images = get_top_images(query_vector, max_results=5)

#     return {
#         "question": query,
#         "answer": answer,
#         "images": images,   # ✅ Included here
#         "sources": sources
#     }
