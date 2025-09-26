# utils_index.py
import os, json
from dotenv import load_dotenv
load_dotenv()  # loads .env at project root
from typing import Dict, Any, List
from django.db.models import QuerySet
from django.utils.timezone import now

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec  # <-- v3 API

from ResearchApp.models import StudyDocument, Study  # adjust if needed

# ---- Pinecone / embeddings constants ----
INDEX_NAME = os.getenv("PINECONE_INDEX", "psygen-studies")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))  # text-embedding-3-small = 1536; -large = 3072
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_METRIC = os.getenv("PINECONE_METRIC", "cosine")  # cosine | euclidean | dotproduct
# ----------------------------------------

def _qs_to_names(qs: QuerySet, attr: str) -> List[str]:
    if not qs:
        return []
    return list(qs.values_list(attr, flat=True))

def _safe_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)

def build_study_metadata(study: Study) -> Dict[str, Any]:
    design_name = study.study_designs.design_name if study.study_designs else ""
    meta = {
        "study_id": str(study.id),
        "pmid": _safe_str(study.pmid),
        "title": _safe_str(study.title),
        "abstract": _safe_str(study.abstract),
        "year": study.year if study.year is not None else None,
        "DOI": _safe_str(study.DOI),
        "journal_name": _safe_str(study.journal_name),
        "impact_factor": study.impact_factor if study.impact_factor is not None else None,
        "funding_source": _safe_str(study.funding_source),
        "lead_author": _safe_str(study.lead_author),

        # Relations (lists of names)
        "countries": _qs_to_names(study.countries.all(), "name"),
        "article_type": _qs_to_names(study.article_type.all(), "article_name"),
        "disorder": _qs_to_names(study.disorder.all(), "disorder_name"),
        "biological_modalities": _qs_to_names(study.biological_modalities.all(), "modality_name"),
        "genetic_source_materials": _qs_to_names(study.genetic_source_materials.all(), "material_type"),

        # Study details
        "phenotype": _safe_str(study.phenotype),
        "diagnostic_criteria_used": _safe_str(study.diagnostic_criteria_used),
        "study_design": _safe_str(design_name),
        "sample_size": _safe_str(study.sample_size),
        "age_range": _safe_str(study.age_range),
        "mean_age": _safe_str(study.mean_age),
        "male_female_split": _safe_str(study.male_female_split),
        "citation": study.citation if study.citation is not None else None,
        "keyword": _safe_str(study.keyword),
        "date": _safe_str(study.date),
        "pages": _safe_str(study.pages),
        "issue": _safe_str(study.issue),
        "volume": _safe_str(study.volume),
        "automatic_tags": _safe_str(study.automatic_tags),
        "authors_affiliations": _safe_str(study.authors_affiliations),
        "biological_risk_factor_studied": _safe_str(study.biological_risk_factor_studied),
        "biological_rationale_provided": _safe_str(study.biological_rationale_provided),
        "status_of_corresponding_gene": _safe_str(study.status_of_corresponding_gene),
        "technology_platform": _safe_str(study.technology_platform),
        "evaluation_method": _safe_str(study.evaluation_method),
        "statistical_model": _safe_str(study.statistical_model),
        "criteria_for_significance": _safe_str(study.criteria_for_significance),
        "validation_performed": _safe_str(study.validation_performed),
        "findings_conclusions": _safe_str(study.findings_conclusions),
        "generalisability_of_conclusion": _safe_str(study.generalisability_of_conclusion),
        "adequate_statistical_powered": _safe_str(study.adequate_statistical_powered),
        "comment": _safe_str(study.comment),
        "should_exclude": bool(study.should_exclude),

        # Book-keeping
        "indexed_at": now().isoformat(timespec="seconds"),
        "schema_version": "v1",
    }
    for k in ("abstract", "automatic_tags", "findings_conclusions", "authors_affiliations", "comment"):
        if meta.get(k) and len(meta[k]) > 8000:
            meta[k] = meta[k][:8000] + " …[truncated]"
    return meta

# ---- Pinecone helpers (v3) ----
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
# --------------------------------

def process_study_pdf_with_langchain(study_document: StudyDocument):
    """Read PDF, split, embed, and upsert into Pinecone with full Study metadata."""
    study = study_document.study
    pdf_path = study_document.pdf_file.path

    # 1) Load PDF
    loader = PyMuPDFLoader(pdf_path)
    base_docs = loader.load()

    # 2) Split smartly
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    docs = splitter.split_documents(base_docs)

    # Persist extracted text (optional)
    study_document.extracted_text = "\n".join(d.page_content for d in docs)
    study_document.save(update_fields=["extracted_text"])

    # 3) Init Pinecone + embeddings + vectorstore (v3)
    index = _get_index()  # pc.Index
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

    # 4) Build metadata
    study_meta = build_study_metadata(study)

    # 5) Prepare per-chunk payloads
    texts, metadatas, ids = [], [], []
    char_cursor = 0
    for i, d in enumerate(docs):
        text = d.page_content or ""
        page = d.metadata.get("page")
        chunk_id = f"{study.id}-p{page}-c{i}"
        md = {
            **study_meta,
            "page": int(page) if page is not None else None,
            "chunk_index": i,
            "chunk_char_len": len(text),
            "chunk_char_start": char_cursor,
            "chunk_char_end": char_cursor + len(text),
            "source": os.path.basename(pdf_path),
        }
        char_cursor += len(text)
        texts.append(text)
        metadatas.append(md)
        ids.append(chunk_id)

    # 6) Upsert
    vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    return {"chunks_indexed": len(texts), "study_id": str(study.id), "index": INDEX_NAME}
