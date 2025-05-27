# Init
#API_KEY = "pcsk_5nAXxD_2BPaYcqPc4pJS6kJMo4zFvJFD4UafQXS3ZuhkyU78achEUbf9FrUZ5iirJdw83j"
#pc = Pinecone(api_key=API_KEY, environment="us-east-1")
#index = pc.Index("abstractive-question-answering")
# retriever = SentenceTransformer("flax-sentence-embeddings/all_datasets_v3_mpnet-base")
import fitz  # PyMuPDF
import textwrap
from ResearchApp.models import StudyDocument
from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion, Metric
import os
from dotenv import load_dotenv
load_dotenv()


# === CONFIG ===
# Pinecone setup (secure via environment or settings)
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_5nAXxD_2BPaYcqPc4pJS6kJMo4zFvJFD4UafQXS3ZuhkyU78achEUbf9FrUZ5iirJdw83j")
# PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENV", "us-east-1")
# Load Pinecone keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENV")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
INDEX_NAME = "psygen-qa-index"
# Initialize Pinecone (once)


# if INDEX_NAME not in pc.list_indexes():
#     pc.create_index(
#         name=INDEX_NAME,
#         dimension=768,
#         metric=Metric.COSINE,
#         spec=ServerlessSpec(
#             cloud=CloudProvider.AWS,
#             region=AwsRegion.US_EAST_1
#         )
#     )

def ensure_index_exists():
    if INDEX_NAME not in pc.list_indexes():
        pc.create_index(
            name=INDEX_NAME,
            dimension=768,
            metric=Metric.COSINE,
            spec=ServerlessSpec(
                cloud=CloudProvider.AWS,
                region=AwsRegion.US_EAST_1
            )
        )

index = pc.Index(INDEX_NAME)

# Lazy-loading transformer model
_retriever = None
def get_retriever():
    global _retriever
    if _retriever is None:
        from sentence_transformers import SentenceTransformer
        _retriever = SentenceTransformer("all-mpnet-base-v2")  # CPU-friendly
    return _retriever


def process_study_pdf(study_document: StudyDocument):
    """Extracts text from PDF, chunks it, embeds it, and indexes it in Pinecone."""
    ensure_index_exists()
    
    pdf_path = study_document.pdf_file.path
    doc = fitz.open(pdf_path)
    full_text = "\n".join([page.get_text() for page in doc])

    study_document.extracted_text = full_text
    study_document.save()

    # Chunk text (approx 512 tokens per chunk)
    chunks = textwrap.wrap(full_text, width=512)
    if not chunks:
        return

    # Embed chunks
    retriever = get_retriever()
    embeddings = retriever.encode(chunks).tolist()

    # Prepare Pinecone vectors
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append((
            f"{study_document.study.id}-{i}",
            embedding,
            {
                "content": chunk,
                "study_id": study_document.study.id,
                "title": study_document.study.title,
            }
        ))

    index.upsert(vectors=vectors)
