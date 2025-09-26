# services/metadata_extractor.py
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------
# Logging (module-level). Django might also configure logging; this ensures
# we still see debug logs during dev if no config is present.
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# ---------------------------------------------------------------------
# Pydantic v1/v2 compatibility
# ---------------------------------------------------------------------
try:
    from pydantic import BaseModel, Field, field_validator as _field_validator  # type: ignore
    _PVD2 = True
    _PVD1 = False
except Exception:  # pragma: no cover
    from pydantic import BaseModel, Field, validator as _validator  # type: ignore
    _PVD1 = True
    _PVD2 = False

from openai import OpenAI
import fitz  # PyMuPDF
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------
class Author(BaseModel):
    name: str
    affiliation_numbers: List[str] = Field(default_factory=list)

class AuthorsAffiliations(BaseModel):
    authors: List[Author] = Field(default_factory=list)
    affiliations: Dict[str, str] = Field(default_factory=dict)

class ExtractedStudy(BaseModel):
    pmid: Optional[str] = None
    title: str
    abstract: str
    year: Optional[int] = None
    DOI: Optional[str] = ""
    journal_name: Optional[str] = None

    impact_factor: Optional[float] = None
    funding_source: Optional[str] = None
    lead_author: Optional[str] = None

    countries: List[str] = Field(default_factory=list)
    article_type: List[str] = Field(default_factory=list)
    disorder: List[str] = Field(default_factory=list)

    phenotype: Optional[str] = None
    diagnostic_criteria_used: Optional[str] = None
    study_designs: Optional[str] = None
    sample_size: Optional[str] = None
    age_range: Optional[str] = None
    mean_age: Optional[str] = None
    male_female_split: Optional[str] = None

    biological_modalities: List[str] = Field(default_factory=list)
    citation: Optional[int] = 0
    keyword: Optional[str] = None
    date: Optional[str] = None
    pages: Optional[str] = None
    issue: Optional[str] = None
    volume: Optional[str] = None
    automatic_tags: Optional[str] = None

    # NEW: enforce structured authors + affiliations
    authors_affiliations: Optional[AuthorsAffiliations] = None

    biological_risk_factor_studied: Optional[str] = None
    biological_rationale_provided: Optional[str] = None
    status_of_corresponding_gene: Optional[str] = None
    technology_platform: Optional[str] = None
    genetic_source_materials: List[str] = Field(default_factory=list)
    evaluation_method: Optional[str] = None
    statistical_model: Optional[str] = None
    criteria_for_significance: Optional[str] = None
    validation_performed: Optional[str] = None
    findings_conclusions: Optional[str] = None
    generalisability_of_conclusion: Optional[str] = None
    adequate_statistical_powered: Optional[str] = None
    comment: Optional[str] = None
    should_exclude: bool = False

    if _PVD2:
        @_field_validator("impact_factor", mode="before")
        def _ifloat(cls, v):
            if v in (None, "", "N/A"):
                return None
            try:
                return float(str(v).replace(",", "."))
            except Exception:
                return None
    else:
        @_validator("impact_factor", pre=True)  # type: ignore
        def _ifloat(cls, v):
            if v in (None, "", "N/A"):
                return None
            try:
                return float(str(v).replace(",", "."))
            except Exception:
                return None


# ---------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------
def extract_text_from_pdf(file_obj) -> str:
    """Read UploadedFile/bytes and return text; restore stream position if seekable."""
    pos = file_obj.tell() if hasattr(file_obj, "tell") else None
    data = file_obj.read()
    if pos is not None:
        try:
            file_obj.seek(pos)
        except Exception:
            pass
    doc = fitz.open(stream=data, filetype="pdf")
    texts: List[str] = []
    for p in doc:
        try:
            texts.append(p.get_text())
        except Exception as e:
            logger.warning("Failed to read page text: %s", e)
    return "\n".join(texts)


# ---------------------------------------------------------------------
# Prompt scaffolding
# ---------------------------------------------------------------------
SYSTEM = (
    "You extract structured bibliographic and methodological metadata from research articles. "
    "You MUST respond with one valid JSON object only—no prose, no code fences."
)

# ---------------------------------------------------------------------
# JSON repair helpers
# ---------------------------------------------------------------------
_SMART_QUOTES = {
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u2018": "'", "\u2019": "'", "\u2032": "'", "\u2033": '"',
}
_CTRL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

def _escape_unescaped_newlines_in_strings(s: str) -> str:
    """
    Replace literal newline characters inside quoted JSON strings with \\n.
    Keeps newlines outside of strings untouched.
    """
    out = []
    in_str = False
    escape = False
    for ch in s:
        if in_str:
            if escape:
                out.append(ch)
                escape = False
            else:
                if ch == '\\':
                    out.append(ch)
                    escape = True
                elif ch == '"':
                    out.append(ch)
                    in_str = False
                elif ch == '\n':
                    out.append('\\n')
                elif ch == '\r':
                    out.append('\\n')
                else:
                    out.append(ch)
        else:
            if ch == '"':
                out.append(ch)
                in_str = True
            else:
                out.append(ch)
    return ''.join(out)

def _normalize_jsonish(s: str) -> str:
    """Normalize common issues: smart quotes, control chars, dangling commas, code fences, newline-in-strings."""
    t = s.strip()

    # Strip markdown code fences
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.IGNORECASE | re.MULTILINE).strip()

    # Replace smart quotes
    for bad, good in _SMART_QUOTES.items():
        t = t.replace(bad, good)

    # Remove control characters
    t = _CTRL_CHARS_RE.sub("", t)

    # Remove trailing commas before } or ]
    t = re.sub(r",\s*(\})", r"\1", t)
    t = re.sub(r",\s*(\])", r"\1", t)

    # Ensure strings don't contain raw newlines (invalid JSON)
    t = _escape_unescaped_newlines_in_strings(t)
    return t

def _extract_first_json_object(s: str) -> str:
    """
    Find the first *balanced* top-level {...} JSON object in s, respecting quoted strings
    and escape sequences. Raises ValueError if not found.
    """
    t = _normalize_jsonish(s)
    start = t.find("{")
    if start == -1:
        raise ValueError("No '{' found")

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if escape:
                escape = False
            else:
                if ch == '\\':
                    escape = True
                elif ch == '"':
                    in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    candidate = t[start:i+1]
                    return candidate
    raise ValueError("Unbalanced braces; no complete JSON object")

# Save raw/normalized payloads for debugging
_LOG_DIR = Path("/tmp/ingest_logs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)

def _save_debug_payload(prefix: str, content: str) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    p = _LOG_DIR / f"{prefix}-{ts}.txt"
    try:
        p.write_text(content, encoding="utf-8")
        logger.debug("Saved debug payload: %s", p)
    except Exception as e:
        logger.warning("Failed to save debug payload %s: %s", p, e)
    return str(p)

def _list_model_fields() -> List[str]:
    """List model field names in a Pydantic v1/v2 compatible way."""
    try:
        # Pydantic v2
        return list(ExtractedStudy.model_json_schema()["properties"].keys())  # type: ignore
    except Exception:
        # Pydantic v1
        return list(ExtractedStudy.schema()["properties"].keys())  # type: ignore


# ---------------------------------------------------------------------
# Coercion helpers (fix common model-shape issues from LLM)
# ---------------------------------------------------------------------
_LIST_FIELDS = [
    "countries",
    "article_type",
    "disorder",
    "biological_modalities",
    "genetic_source_materials",
]

def _ensure_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if x is not None]
    if isinstance(v, str):
        v = v.strip()
        return [v] if v else []
    return [str(v)]

def _coerce_authors_affiliations(aa: Any) -> AuthorsAffiliations:
    """
    Normalize a variety of shapes into:
      {"authors":[{"name":..., "affiliation_numbers":[...]}], "affiliations":{"1":"...", ...}}
    """
    # Defaults
    authors: List[Dict[str, Any]] = []
    aff_map: Dict[str, str] = {}
    next_id = 1

    def _assign_aff_nums(affs: List[str]) -> List[str]:
        nonlocal next_id, aff_map
        nums: List[str] = []
        for aff in affs:
            aff = str(aff).strip()
            if not aff:
                continue
            # Try to reuse existing id for identical affiliation text
            found = None
            for k, v in aff_map.items():
                if v == aff:
                    found = k
                    break
            if found is None:
                k = str(next_id)
                aff_map[k] = aff
                nums.append(k)
                next_id += 1
            else:
                nums.append(str(found))
        return nums

    if aa is None:
        return AuthorsAffiliations(authors=[], affiliations={})

    # Already close to target shape
    if isinstance(aa, dict):
        # Normalize affiliations
        raw_aff = aa.get("affiliations")
        if isinstance(raw_aff, dict):
            aff_map = {str(k): str(v) for k, v in raw_aff.items()}
            # keep next_id > max existing
            try:
                max_key = max([int(k) for k in aff_map.keys()] or [0])
                next_id = max_key + 1
            except Exception:
                next_id = 1
        elif isinstance(raw_aff, list):
            for i, v in enumerate(raw_aff, start=1):
                aff_map[str(i)] = str(v)

        # Normalize authors
        raw_auth = aa.get("authors", [])
        if isinstance(raw_auth, list):
            for it in raw_auth:
                if isinstance(it, dict):
                    name = str(it.get("name", "")).strip()
                    nums = it.get("affiliation_numbers") or it.get("affiliations") or it.get("affiliation_ids") or it.get("affiliation_index")
                    if isinstance(nums, list):
                        nums_list = [str(n) for n in nums if str(n).strip() != ""]
                    elif isinstance(nums, (int, str)):
                        nums_list = [str(nums)]
                    else:
                        # if there's an 'affiliation' string, allocate an id
                        aff = it.get("affiliation")
                        if isinstance(aff, str) and aff.strip():
                            nums_list = _assign_aff_nums([aff])
                        elif isinstance(aff, list):
                            nums_list = _assign_aff_nums([str(x) for x in aff])
                        else:
                            nums_list = []
                    if name:
                        authors.append({"name": name, "affiliation_numbers": nums_list})
                elif isinstance(it, str):
                    nm = it.strip()
                    if nm:
                        authors.append({"name": nm, "affiliation_numbers": []})
        else:
            # if authors is a string
            if isinstance(raw_auth, str) and raw_auth.strip():
                authors.append({"name": raw_auth.strip(), "affiliation_numbers": []})

        return AuthorsAffiliations(
            authors=[Author(**a) for a in authors],
            affiliations=aff_map,
        )

    # List form: could be ["Name"], or [{"name":..., "affiliation":...}, ...]
    if isinstance(aa, list):
        for it in aa:
            if isinstance(it, dict):
                name = str(it.get("name", "")).strip()
                aff = it.get("affiliation")
                affs: List[str] = []
                if isinstance(aff, str) and aff.strip():
                    affs = [aff]
                elif isinstance(aff, list):
                    affs = [str(x) for x in aff if str(x).strip() != ""]
                nums = _assign_aff_nums(affs) if affs else []
                if name:
                    authors.append({"name": name, "affiliation_numbers": nums})
            elif isinstance(it, str):
                nm = it.strip()
                if nm:
                    authors.append({"name": nm, "affiliation_numbers": []})

        return AuthorsAffiliations(
            authors=[Author(**a) for a in authors],
            affiliations=aff_map,
        )

    # Single string fallback
    if isinstance(aa, str) and aa.strip():
        return AuthorsAffiliations(
            authors=[Author(name=aa.strip(), affiliation_numbers=[])],
            affiliations={},
        )

    return AuthorsAffiliations(authors=[], affiliations={})

def _coerce_to_model_shape(data: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(data or {})

    # Guarantee all model keys exist
    for f in _list_model_fields():
        d.setdefault(f, None)

    # Coerce list fields
    for f in _LIST_FIELDS:
        d[f] = _ensure_list(d.get(f))

    # authors_affiliations -> structured object
    d["authors_affiliations"] = _coerce_authors_affiliations(d.get("authors_affiliations"))

    # citation: int or parseable string
    cit = d.get("citation")
    if cit in (None, "", "N/A"):
        d["citation"] = None
    else:
        try:
            if isinstance(cit, str):
                m = re.search(r"\d+", cit.replace(",", ""))
                d["citation"] = int(m.group(0)) if m else None
            else:
                d["citation"] = int(cit)
        except Exception:
            d["citation"] = None

    # year: int or parseable string (reasonable window)
    yr = d.get("year")
    if yr in (None, "", "N/A"):
        d["year"] = None
    else:
        try:
            if isinstance(yr, str):
                m = re.search(r"(18|19|20|21)\d{2}", yr)
                d["year"] = int(m.group(0)) if m else None
            else:
                y = int(yr)
                d["year"] = y if 1800 <= y <= 2200 else None
        except Exception:
            d["year"] = None

    # impact_factor: leave to model validator, but normalize decimal comma
    ifn = d.get("impact_factor")
    if isinstance(ifn, str):
        d["impact_factor"] = ifn.replace(",", ".")

    # should_exclude: bool-ish
    se = d.get("should_exclude")
    if isinstance(se, str):
        d["should_exclude"] = se.strip().lower() in ("1", "true", "yes", "y")
    elif isinstance(se, (int, float)):
        d["should_exclude"] = bool(se)
    elif se is None:
        d["should_exclude"] = False

    # Ensure string fields are strings (trim)
    for f in ("title", "abstract", "DOI", "journal_name", "funding_source", "lead_author",
              "phenotype", "diagnostic_criteria_used", "study_designs", "sample_size",
              "age_range", "mean_age", "male_female_split", "keyword", "date", "pages",
              "issue", "volume", "automatic_tags", "biological_risk_factor_studied",
              "biological_rationale_provided", "status_of_corresponding_gene",
              "technology_platform", "evaluation_method", "statistical_model",
              "criteria_for_significance", "validation_performed", "findings_conclusions",
              "generalisability_of_conclusion", "adequate_statistical_powered",
              "comment", "pmid"):
        if d.get(f) is not None and not isinstance(d.get(f), str):
            d[f] = str(d[f])

    return d


# ---------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------
def extract_study_metadata_from_text(text: str) -> ExtractedStudy:
    """
    Ask the model for JSON; log and save raw outputs; normalize and parse robustly; coerce; validate via Pydantic.
    Raises exceptions if all parsing options fail (so your view can return a 500 with details).
    """
    MAX_CHARS = 50_000
    text = text[:MAX_CHARS]

    fields = _list_model_fields()

    prompt = f"""
Extract the study metadata from the article text below.

Rules (CRITICAL):
- Return a SINGLE valid JSON object and nothing else.
- Keep "abstract" as exact as it is in the paper.
- For list fields, return arrays of short strings (<= 10 items).
- For authors_affiliations, you MUST return an object with exactly:
    "authors": [{{"name": "Full Name", "affiliation_numbers": ["1","2"]}}, ...],
    "affiliations": {{"1": "Affiliation text 1", "2": "Affiliation text 2", ...}}
  Use numeric strings for affiliation_numbers, matching the keys in "affiliations".
- If something is unknown, set it to null or an empty string/array (but KEEP the key present).
- Use integers for "year" and "citation" when possible; floats for "impact_factor".

Return JSON with exactly these top-level keys (no extra keys, all present):
{fields}

Article text:
<<<
{text}
>>>
""".strip()

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]

    # -------------------- Try 1 — structured output with json_object --------------------
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
        max_tokens=5000,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or ""
    _save_debug_payload("llm_raw_try1", raw)
    logger.debug("Raw LLM output (try1) length=%s", len(raw))
    logger.debug("Raw LLM output (try1) head:\n%s", raw[:1200])

    # Parse with normalization and balanced-object extraction
    try:
        normalized = _normalize_jsonish(raw)
        _save_debug_payload("llm_norm_try1", normalized)
        first_obj = _extract_first_json_object(normalized)
        data = json.loads(first_obj)
    except Exception as e1:
        logger.exception("json.loads failed for try1 after extraction: %s", e1)

        # -------------------- Try 2 — stricter retry without response_format --------------------
        resp2 = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Return ONLY one valid JSON object, no extra text."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1200,
        )
        raw2 = resp2.choices[0].message.content or ""
        _save_debug_payload("llm_raw_try2", raw2)
        logger.debug("Raw LLM output (try2) length=%s", len(raw2))
        logger.debug("Raw LLM output (try2) head:\n%s", raw2[:1200])

        try:
            normalized2 = _normalize_jsonish(raw2)
            _save_debug_payload("llm_norm_try2", normalized2)
            first_obj2 = _extract_first_json_object(normalized2)
            data = json.loads(first_obj2)
        except Exception as e2:
            logger.exception("json.loads failed for try2 after extraction: %s", e2)

            # -------------------- Try 3 — Responses API with strict schema --------------------
            try:
                resp3 = client.responses.create(
                    model="gpt-4.1-mini",
                    input=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "ExtractedStudy",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "pmid": {"type": ["string", "null"]},
                                    "title": {"type": "string"},
                                    "abstract": {"type": "string"},
                                    "year": {"type": ["integer", "null"]},
                                    "DOI": {"type": ["string", "null"]},
                                    "journal_name": {"type": ["string", "null"]},
                                    "impact_factor": {"type": ["number", "null"]},
                                    "funding_source": {"type": ["string", "null"]},
                                    "lead_author": {"type": ["string", "null"]},
                                    "countries": {"type": "array", "items": {"type": "string"}},
                                    "article_type": {"type": "array", "items": {"type": "string"}},
                                    "disorder": {"type": "array", "items": {"type": "string"}},
                                    "phenotype": {"type": ["string", "null"]},
                                    "diagnostic_criteria_used": {"type": ["string", "null"]},
                                    "study_designs": {"type": ["string", "null"]},
                                    "sample_size": {"type": ["string", "null"]},
                                    "age_range": {"type": ["string", "null"]},
                                    "mean_age": {"type": ["string", "null"]},
                                    "male_female_split": {"type": ["string", "null"]},
                                    "biological_modalities": {"type": "array", "items": {"type": "string"}},
                                    "citation": {"type": ["integer", "null"]},
                                    "keyword": {"type": ["string", "null"]},
                                    "date": {"type": ["string", "null"]},
                                    "pages": {"type": ["string", "null"]},
                                    "issue": {"type": ["string", "null"]},
                                    "volume": {"type": ["string", "null"]},
                                    "automatic_tags": {"type": ["string", "null"]},
                                    "authors_affiliations": {
                                        "type": ["object", "null"],
                                        "required": ["authors", "affiliations"],
                                        "properties": {
                                            "authors": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "name": {"type": "string"},
                                                        "affiliation_numbers": {
                                                            "type": "array",
                                                            "items": {"type": "string"}
                                                        }
                                                    },
                                                    "required": ["name", "affiliation_numbers"],
                                                    "additionalProperties": False
                                                }
                                            },
                                            "affiliations": {
                                                "type": "object",
                                                "additionalProperties": {"type": "string"}
                                            }
                                        },
                                        "additionalProperties": False
                                    },
                                    "biological_risk_factor_studied": {"type": ["string", "null"]},
                                    "biological_rationale_provided": {"type": ["string", "null"]},
                                    "status_of_corresponding_gene": {"type": ["string", "null"]},
                                    "technology_platform": {"type": ["string", "null"]},
                                    "genetic_source_materials": {"type": "array", "items": {"type": "string"}},
                                    "evaluation_method": {"type": ["string", "null"]},
                                    "statistical_model": {"type": ["string", "null"]},
                                    "criteria_for_significance": {"type": ["string", "null"]},
                                    "validation_performed": {"type": ["string", "null"]},
                                    "findings_conclusions": {"type": ["string", "null"]},
                                    "generalisability_of_conclusion": {"type": ["string", "null"]},
                                    "adequate_statistical_powered": {"type": ["string", "null"]},
                                    "comment": {"type": ["string", "null"]},
                                    "should_exclude": {"type": "boolean"}
                                },
                                "required": [
                                    "title", "abstract", "countries", "article_type",
                                    "disorder", "biological_modalities",
                                    "genetic_source_materials", "should_exclude",
                                    "authors_affiliations"
                                ],
                                "additionalProperties": False
                            }
                        }
                    }
                )
                raw3 = getattr(resp3, "output_text", None) or ""
                _save_debug_payload("llm_raw_try3", raw3)
                data = json.loads(raw3)
            except Exception as e3:
                logger.exception("Strict schema call failed: %s", e3)
                # Last resort: raise so the caller returns a 500 with logs pointing to /tmp/ingest_logs
                raise

    # ---------- Coerce to the exact model shape before validation ----------
    data = _coerce_to_model_shape(data)

    # Validate and coerce types with Pydantic
    try:
        model = ExtractedStudy(**data)
    except Exception as e:
        _save_debug_payload("pydantic_failed_payload", json.dumps(data, ensure_ascii=False, indent=2))
        logger.exception("Pydantic validation failed: %s", e)
        raise
    return model


# ---------------------------------------------------------------------
# Convenience function: process a Django UploadedFile and return model
# ---------------------------------------------------------------------
def extract_study_from_uploaded_pdf(uploaded_file) -> ExtractedStudy:
    text = extract_text_from_pdf(uploaded_file)
    return extract_study_metadata_from_text(text)
