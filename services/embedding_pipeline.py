import os
import fitz  # PyMuPDF
import textwrap
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion, Metric
from ResearchApp.models import StudyDocument

load_dotenv()

# === Configuration ===
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "psygen-qa-index"
EMBEDDING_DIM = 768

# === Initialize Pinecone ===
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# === Lazy-loading SentenceTransformer ===
_retriever = None
def get_retriever():
    global _retriever
    if _retriever is None:
        from sentence_transformers import SentenceTransformer
        _retriever = SentenceTransformer("all-mpnet-base-v2")
    return _retriever

# === Ensure Pinecone Index Exists ===
def ensure_index_exists():
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric=Metric.COSINE,
            spec=ServerlessSpec(
                cloud=CloudProvider.AWS,
                region=AwsRegion.US_EAST_1
            )
        )

# === Process and Index PDF ===
def process_study_pdf(study_document: StudyDocument):
    """Extract text from a PDF, split it into chunks, embed, and store in Pinecone."""
    ensure_index_exists()
    index = pc.Index(INDEX_NAME)

    pdf_path = study_document.pdf_file.path
    doc = fitz.open(pdf_path)
    full_text = "\n".join([page.get_text() for page in doc])

    study_document.extracted_text = full_text
    study_document.save()

    # Chunk text (approx. 512 characters per chunk)
    chunks = textwrap.wrap(full_text, width=512)
    if not chunks:
        return

    retriever = get_retriever()
    embeddings = retriever.encode(chunks).tolist()

    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append((
            f"{study_document.study.id}-{i}",
            embedding,
            {
                "content": chunk,
                "study_id": study_document.study.id,
                "title": study_document.study.title
            }
        ))

    index.upsert(vectors=vectors)
