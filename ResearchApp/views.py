# from django.contrib.auth.models import User
# from rest_framework.authtoken.models import Token
# from rest_framework.authtoken.views import ObtainAuthToken

from rest_framework.views import APIView
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework import generics
import csv
import re
from io import StringIO
from django.core.exceptions import ValidationError
from django.db import models
from .models import (Study, Disorder, BiologicalModality,
                    GeneticSourceMaterial, ArticleType, StudyDesign,
                    Country,Visitor,StudyDocument, StudyImage, ChatSession,ChatMessage)
from .serializers import (StudySerializer,VisitorCountSerializer, StudyImageSerializer,
                            ChatSessionSerializer, ChatMessageSerializer, StudyUpsertSerializer)
from django.http import HttpResponse
import json
import logging
from django.db.models import Count
from datetime import timedelta
from django.utils import timezone
from rest_framework.permissions import IsAuthenticated, AllowAny

from django.contrib.auth import authenticate, login, logout
from rest_framework.decorators import permission_classes
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework_simplejwt.tokens import RefreshToken

from services.embedding_pipeline import process_study_pdf_with_langchain

from services.qa_pipeline import continue_chat_rag

from langchain_openai import OpenAIEmbeddings
# ResearchApp/views_ingest.py

from rest_framework.parsers import MultiPartParser, FormParser, JSONParser

from django.db import transaction

from .models import StudyDocument
from services.metadata_extractor import extract_text_from_pdf, extract_study_metadata_from_text
# from services.study_upsert import upsert_study_with_metadata
from typing import Any, Dict, List
from django.db import transaction
from django.db.models import Q
from rest_framework.views import APIView
from rest_framework import status, permissions, parsers
from rest_framework.response import Response


retriever = OpenAIEmbeddings(model="text-embedding-3-small")

# from sentence_transformers import SentenceTransformer

# retriever = SentenceTransformer("all-mpnet-base-v2")  # CPU-compatible

logger = logging.getLogger(__name__)

class LogoutView(APIView):
    permission_classes = [IsAuthenticated]  # Ensure authentication is required

    def post(self, request):
        try:
            # Extract the refresh token from the request data
            refresh_token = request.data.get('refresh_token')
            if not refresh_token:
                return Response({"error": "Refresh token is required"}, status=400)

            # Blacklist the refresh token
            token = RefreshToken(refresh_token)
            token.blacklist()

            return Response({"message": "Logged out successfully"}, status=200)

        except Exception as e:
            return Response({"error": "Invalid token"}, status=400)


class UploadCSVView(APIView):
    permission_classes = [IsAuthenticated]  # Restrict access to authenticated users only

    def post(self, request, format=None):
        file = request.FILES.get('file')
        if not file:
            return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)

        data = file.read().decode('ISO-8859-1')
        csv_data = csv.DictReader(StringIO(data))

        errors = []

        for row_number, row in enumerate(csv_data, start=1):
            if not any(row.values()):
                continue

            # Processing the impact factor
            impact_factor = row.get('Impact Factor', '').replace(',', '.').strip()
            try:
                impact_factor = float(impact_factor) if impact_factor else None
            except ValueError:
                impact_factor = None

            study_designs = row.get('Study Design', '').upper().strip()
            StudyDesign_name, _ = StudyDesign.objects.get_or_create(design_name=study_designs)

            # Prepare the data dictionary for the Study model
            row_data = {
                'pmid': row.get('PMID', '').strip(),
                'title': row.get('Title', '').strip(),
                'abstract': row.get('Abstract', '').strip(),
                'year': row.get('Year', None),
                'DOI': row.get('DOI', '').strip(),
                'journal_name': row.get('Journal Name', '').strip(),
                'impact_factor': impact_factor,
                'funding_source': row.get('Funding Source', '').strip(),
                'lead_author': row.get('Lead Author', '').strip(),
                'phenotype': row.get('Phenotype', '').strip(),
                'diagnostic_criteria_used': row.get('Diagnostic Criteria Used', '').strip(),
                'study_designs': StudyDesign_name,
                'sample_size': row.get('Sample Size', '').strip(),
                'age_range': row.get('Age Range', '').strip(),
                'mean_age': row.get('Mean Age', '').strip(),
                'male_female_split': row.get('Male/Female Split', '').strip(),
                'citation': row.get('Citation', '').strip(),
                'keyword': row.get('Keywords', '').strip(),
                'date': row.get('Date', '').strip(),
                'pages': row.get('Pages', '').strip(),
                'issue': row.get('Issue', None),
                'volume': row.get('Volume', None),
                'automatic_tags': row.get('Automatic Tags', '').strip(),
                'biological_risk_factor_studied': row.get('Biological Risk Factor Studied', '').strip(),
                'biological_rationale_provided': row.get('Biological Rationale Provided', '').strip(),
                'status_of_corresponding_gene': row.get('Status of Corresponding Gene', '').strip(),
                'technology_platform': row.get('Technology Platform', '').strip(),
                'evaluation_method': row.get('Evaluation Method', '').strip(),
                'statistical_model': row.get('Statistical Model', '').strip(),
                'criteria_for_significance': row.get('Criteria for Significance', '').strip(),
                'validation_performed': row.get('Validation Performed', '').strip(),
                'findings_conclusions': row.get('Findings/Conclusions', '').strip(),
                'generalisability_of_conclusion': row.get('Generalisability of Conclusion', '').strip(),
                'adequate_statistical_powered': row.get('Adequate Statistical Powered', '').strip(),
                'comment': row.get('Remark Comment', '').strip(),
            }

            # Handle authors/affiliations JSON field
            authors_affiliations = row.get('Authors/Affiliations', '').strip()
            try:
                authors_affiliations = json.loads(authors_affiliations) if authors_affiliations else None
            except json.JSONDecodeError:
                authors_affiliations = None
                errors.append({'row': row_number, 'error': 'Invalid JSON format in authors_affiliations field'})
            row_data['authors_affiliations'] = authors_affiliations

            # Handle exclusion based on 'Should Exclude?' column
            should_exclude = row.get('Should Exclude?')
            row_data['should_exclude'] = True if should_exclude and should_exclude.strip().lower() == 'yes' else False

            # Handling many-to-many relationships
            many_to_many_fields = {
                'countries': (Country, 'name', row.get('Countries', '').split(',')),
                'article_type': (ArticleType, 'article_name', row.get('Article Type', '').split(',')),
                'disorder': (Disorder, 'disorder_name', row.get('Disorder', '').split(',')),
                'biological_modalities': (BiologicalModality, 'modality_name', row.get('Biological Modality', '').split(',')),
                'genetic_source_materials': (GeneticSourceMaterial, 'material_type', row.get('Genetic Source Materials', '').split(','))
            }

            for key, (model, field, value_list) in many_to_many_fields.items():
                instances = []
                # field_data = []
                for item in value_list:
                    item = item.strip().upper()
                    if item:
                        instance, _ = model.objects.get_or_create(**{field: item})
                        instances.append(instance)
                        # field_data.append({"id": instance.id, field: getattr(instance, field)})
                        # field_data.append({field: getattr(instances, field)})
                row_data[key] = instances

            # Save the data using the serializer
            serializer = StudySerializer(data=row_data)
            if serializer.is_valid():
                try:
                    serializer.save()
                except Exception as e:
                    logger.error(f"Error saving row {row_number}: {str(e)}")
                    errors.append({'row': row_number, 'error': str(e)})
            else:
                logger.error(f"Validation error at row {row_number}: {serializer.errors}")
                errors.append({'row': row_number, 'error': serializer.errors})

        # Return success or errors
        if errors:
            return Response({"errors": errors}, status=status.HTTP_400_BAD_REQUEST)
        return Response({"message": "CSV file processed successfully."}, status=status.HTTP_201_CREATED)


class DownloadCSVExampleView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, format=None):
        # Define the response and CSV writer
        print(f"Authenticated user: {request.user}")  # Debugging
        if not request.user.is_authenticated:
            return Response({"error": "User is not authenticated"}, status=401)

        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="study_template.csv"'

        writer = csv.writer(response)

        # Write the CSV header
        writer.writerow([
            'PMID', 'Title', 'Abstract', 'Year', 'DOI', 'Journal Name', 'Countries', 'Impact Factor',
            'Article Type', 'Funding Source', 'Lead Author', 'Disorder', 'Phenotype',
            'Diagnostic Criteria Used', 'Study Design', 'Sample Size', 'Age Range', 'Mean Age',
            'Male/Female Split', 'Biological Modality','Citation','Keywords', 'Date', 'Pages', 'Issue', 'Volume','Automatic Tags',
            'Authors/Affiliations', 'Biological Risk Factor Studied','Biological Rationale Provided',
            'Status of Corresponding Gene', 'Technology Platform',
            'Genetic Source Materials', 'Evaluation Method', 'Statistical Model',
            'Criteria for Significance', 'Validation Performed', 'Findings/Conclusions',
            'Generalisability of Conclusion', 'Adequate Statistical Powered','Comment',
            'Should Exclude?'
        ])

        # Write a sample row
        writer.writerow([
            '123456', 'Example Study Title', 'Study abstract', '2024', '10.1234/doi-example',
            'Journal of Genetics', 'USA, Canada', '5.8', 'Original Research', 'NIH', 'John Doe',
            'Schizophrenia', 'DSM-5','Pre-diagnosed', 'Case-Control', '100', '18-65', '35', '50/50', 'Genomics',
            '0', 'Social anxiety disorder; Genetics; Serotonin; Dopamine; Temperament.',
            '2024-01-01', '123-135', '2', '12', 'Automatic tag', '[{"author": "John Doe", "affiliation": "University X"}]',
            'Gene XYZ', 'Epigenetics hypothesis', 'Active', 'PCR', 'Blood',
            'DNA methylation', 'Regression', 'p < 0.05', 'Internal Validation',
            'No significant difference', 'Limited due to small sample size', 'low power',
            'Some remarks', 'No'
        ])

        return response


class VisitorCountAPIView(APIView):
    permission_classes = [IsAuthenticated]
    def get(self, request):
        # Count unique visitors by IP address
        unique_visitors = Visitor.objects.values('ip_address').distinct().count()

        # Count total visits
        total_visits = Visitor.objects.count()

        # Get today's date and calculate the date 7 days ago
        today = timezone.now().date()
        last_7_days = today - timedelta(days=6)


        # Query for visit counts grouped by day for the last 7 days
        daily_visits = (
            Visitor.objects
            .filter(visit_date__date__range=[last_7_days, today])
            .annotate(day=models.functions.TruncDate('visit_date'))
            .values('day')
            .annotate(visit_count=Count('id'))
            .order_by('day')
        )

        # Prepare daily visits data as a list of dictionaries with date and visit count
        daily_visits_data = [
            {"date": day_data['day'], "visit_count": day_data['visit_count']}
            for day_data in daily_visits
        ]

        # Prepare data for the response
        data = {
            "unique_visitors": unique_visitors,
            "total_visits": total_visits,
            "daily_visits": daily_visits_data
        }

        serializer = VisitorCountSerializer(data)
        return Response(serializer.data, status=status.HTTP_200_OK)


# views.py
#==============================================================================================
#==============================================================================================
#==============================================================================================
class UploadPDFView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        study_id = request.data.get("study_id")
        file = request.FILES.get("file")

        if not study_id or not file:
            return Response({"error": "Missing study_id or file"}, status=400)

        try:
            study = Study.objects.get(id=study_id)
        except Study.DoesNotExist:
            return Response({"error": "Study not found"}, status=404)

        doc, created = StudyDocument.objects.get_or_create(study=study)
        doc.pdf_file = file
        doc.save()

        # process_study_pdf(doc)  # Extract, chunk, embed, upsert

        return Response({"message": "PDF uploaded and indexed successfully."})


class ContinueChatView(APIView):
    # permission_classes = [AllowAny]  # or IsAuthenticated if you want to protect it

    def post(self, request):
        """
        Body:
        {
          "email": "user@example.com",
          "question": "What variants are implicated in schizophrenia?",
          "session_id": 12,                        # optional
          "filters": { "disorder": {"$in": ["Schizophrenia"]}, "year": {"$gte": 2018} }   # optional Pinecone metadata filter
        }
        """
        email = request.data.get("email", "")
        question = request.data.get("question", "")
        session_id = request.data.get("session_id")
        filters = request.data.get("filters")

        if not email or not question:
            return Response({"error": "email and question are required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            payload = continue_chat_rag(request, email=email, question=question, session_id=session_id, filters=filters)
            return Response(payload, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ChatSessionListView(APIView):
    def get(self, request):
        email = request.query_params.get("email")
        if not email:
            return Response({"error": "Missing email query param."}, status=400)

        sessions = ChatSession.objects.filter(email=email).order_by('-created_at')
        data = ChatSessionSerializer(sessions, many=True).data
        return Response(data)



# ResearchApp/views.py
class StudyImageUploadView(generics.CreateAPIView):
    queryset = StudyImage.objects.all()
    serializer_class = StudyImageSerializer
    parser_classes = [MultiPartParser, FormParser]

    def perform_create(self, serializer):
        image_instance = serializer.save()
        if image_instance.caption:
            embedding = retriever.encode(image_instance.caption).tolist()
            image_instance.embedding = embedding
            image_instance.save()


class StudyImageListView(generics.ListAPIView):
    queryset = StudyImage.objects.all().order_by('-id')
    serializer_class = StudyImageSerializer


class StudyImageDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = StudyImage.objects.all()
    serializer_class = StudyImageSerializer


class DeleteChatSessionView(generics.DestroyAPIView):
    queryset = ChatSession.objects.all()
    serializer_class = ChatSessionSerializer
    lookup_field = "pk"

    def delete(self, request, *args, **kwargs):
        obj = self.get_object()
        email = request.query_params.get("email")
        if not email or obj.email != email:
            return Response({"error": "Unauthorized"}, status=403)
        return super().delete(request, *args, **kwargs)


class DeleteChatMessageView(generics.DestroyAPIView):
    queryset = ChatMessage.objects.all()
    serializer_class = ChatMessageSerializer
    lookup_field = "pk"

    def delete(self, request, *args, **kwargs):
        obj = self.get_object()
        session_email = obj.session.email
        email = request.query_params.get("email")
        if not email or session_email != email:
            return Response({"error": "Unauthorized"}, status=403)
        return super().delete(request, *args, **kwargs)

#================================================================================

# ---------- 1) Extract only: upload PDF, return JSON (no DB writes) ----------
class ExtractMetadataView(APIView):
    # permission_classes = [IsAuthenticated]  # or AllowAny for testing
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        """
        Accepts: multipart/form-data with 'file' (PDF)
        Returns: JSON metadata (no database writes)
        """
        file = request.FILES.get("file")
        if not file:
            return Response({"error": "Missing PDF `file`"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            text = extract_text_from_pdf(file)
            meta_model = extract_study_metadata_from_text(text)
            return Response({"metadata": meta_model.dict()}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


#================================================================================

# views_index.py
class IndexSinglePDFView(APIView):
    """
    POST /pinecone/index/<pk>/
    Index a single StudyDocument by primary key.
    """
    permission_classes = [IsAuthenticated]   # change to [AllowAny] if you want it open

    def post(self, request, pk: int):
        doc = get_object_or_404(StudyDocument, pk=pk)
        try:
            res = process_study_pdf_with_langchain(doc)
            return Response({"status": "ok", **res}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"status": "error", "study_id": str(doc.study.id), "detail": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class IndexAllPDFsView(APIView):
    """
    POST /pinecone/index/all/?offset=0&limit=200
    Index all StudyDocument rows (batched via offset/limit so you can call multiple times).
    """
    permission_classes = [IsAuthenticated]   # change to [AllowAny] if desired

    def post(self, request):
        try:
            offset = int(request.query_params.get("offset", 0))
            limit = int(request.query_params.get("limit", 200))
            qs = StudyDocument.objects.all().order_by("id")[offset:offset + limit]

            results = []
            for doc in qs:
                try:
                    res = process_study_pdf_with_langchain(doc)
                    results.append({"study_id": str(doc.study.id), "ok": True, **res})
                except Exception as e:
                    results.append({"study_id": str(doc.study.id), "ok": False, "error": str(e)})

            return Response(
                {
                    "count": len(results),
                    "offset": offset,
                    "limit": limit,
                    "next_offset": offset + limit,
                    "results": results,
                },
                status=status.HTTP_200_OK,
            )
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


########## views for upload ################
# ingest/views.py
# ----------------- helpers -----------------
def _ensure_list_str(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    s = str(v).strip()
    return [s] if s else []

def _upsert_m2m(model, values: List[str], attr: str) -> List[int]:
    ids: List[int] = []
    for val in values:
        obj, _ = model.objects.get_or_create(**{attr: val})
        ids.append(obj.id)
    return ids

def _clean_int(val, default=None):
    if val in (None, "", "N/A"):
        return default
    try:
        if isinstance(val, str):
            m = re.search(r"\d+", val.replace(",", ""))
            return int(m.group(0)) if m else default
        return int(val)
    except Exception:
        return default

def _normalize_authors_affiliations(aa: Any) -> Dict[str, Any]:
    # Ensure: {"authors":[{"name":..., "affiliation_numbers":[...]}], "affiliations":{"1":"...", ...}}
    if isinstance(aa, dict) and "authors" in aa and "affiliations" in aa:
        authors = []
        for it in aa.get("authors", []) or []:
            if isinstance(it, dict):
                name = str(it.get("name", "")).strip()
                nums = it.get("affiliation_numbers") or []
                if isinstance(nums, (str, int)):
                    nums = [str(nums)]
                elif isinstance(nums, list):
                    nums = [str(x) for x in nums]
                else:
                    nums = []
                if name:
                    authors.append({"name": name, "affiliation_numbers": nums})
            elif isinstance(it, str) and it.strip():
                authors.append({"name": it.strip(), "affiliation_numbers": []})
        affs = {}
        raw_affs = aa.get("affiliations") or {}
        if isinstance(raw_affs, dict):
            for k, v in raw_affs.items():
                affs[str(k)] = str(v)
        elif isinstance(raw_affs, list):
            for i, v in enumerate(raw_affs, start=1):
                affs[str(i)] = str(v)
        return {"authors": authors, "affiliations": affs}

    if isinstance(aa, list):
        authors = []
        for it in aa:
            if isinstance(it, dict) and "name" in it:
                nm = str(it.get("name", "")).strip()
                if nm:
                    nums = it.get("affiliation_numbers") or []
                    if isinstance(nums, (str, int)):
                        nums = [str(nums)]
                    elif isinstance(nums, list):
                        nums = [str(x) for x in nums]
                    else:
                        nums = []
                    authors.append({"name": nm, "affiliation_numbers": nums})
            elif isinstance(it, str) and it.strip():
                authors.append({"name": it.strip(), "affiliation_numbers": []})
        return {"authors": authors, "affiliations": {}}

    if isinstance(aa, str) and aa.strip():
        return {"authors": [{"name": aa.strip(), "affiliation_numbers": []}], "affiliations": {}}

    return {"authors": [], "affiliations": {}}

def _parse_payload(request) -> Dict[str, Any]:
    # multipart: payload is JSON string; JSON body: payload may be object or top-level fields
    data = request.data
    if "payload" in data:
        payload = data.get("payload")
        if isinstance(payload, (dict, list)):
            return payload
        if isinstance(payload, str) and payload.strip():
            try:
                return json.loads(payload)
            except Exception:
                pass
    payload = {}
    for k, v in data.items():
        if k == "pdf":
            continue
        payload[k] = v
    return payload

# ----------------- endpoint -----------------
class StudySaveView(APIView):
    """
    POST /api/studies/save/
      - Upsert Study (by study_id, else by unique (title, year, pmid))
      - Then create/update StudyDocument with the provided PDF (no text extraction)

    Accepts:
      - multipart/form-data: pdf=<file> (optional), payload=<JSON string>
      - application/json: {"payload": {...}} or top-level {...}

    Response:
    {
      "study_id": int,
      "created": bool,
      "document_id": int|null,
      "message": "Study saved"
    }
    """
    permission_classes = [permissions.AllowAny]
    parser_classes = [parsers.MultiPartParser, parsers.JSONParser, parsers.FormParser]

    @transaction.atomic
    def post(self, request, *args, **kwargs):
        ser = StudyUpsertSerializer(data=request.data)
        ser.is_valid(raise_exception=True)

        payload = _parse_payload(request)
        pdf_file = request.FILES.get("pdf")

        title = (payload.get("title") or "").strip()
        abstract = (payload.get("abstract") or "").strip()
        if not title or not abstract:
            return Response(
                {"error": "Both 'title' and 'abstract' are required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Upsert Study
        study = None
        created = False

        study_id = payload.get("study_id")
        if study_id:
            study = Study.objects.filter(id=study_id).first()

        if study is None:
            year = _clean_int(payload.get("year"))
            pmid = (payload.get("pmid") or None) or None
            study = Study.objects.filter(Q(title=title) & Q(year=year) & Q(pmid=pmid)).first()

        if study is None:
            study = Study(title=title, abstract=abstract)
            created = True

        # Scalars
        study.title = title
        study.abstract = abstract
        study.pmid = (payload.get("pmid") or None) or None
        study.year = _clean_int(payload.get("year"))
        study.DOI = (payload.get("DOI") or "")[:100]
        study.journal_name = payload.get("journal_name") or None

        impact_factor = payload.get("impact_factor")
        try:
            if isinstance(impact_factor, str):
                impact_factor = float(impact_factor.replace(",", "."))
            elif impact_factor is not None:
                impact_factor = float(impact_factor)
        except Exception:
            impact_factor = None
        study.impact_factor = impact_factor

        study.funding_source = payload.get("funding_source") or None
        study.lead_author = payload.get("lead_author") or None

        study.phenotype = payload.get("phenotype") or None
        study.diagnostic_criteria_used = payload.get("diagnostic_criteria_used") or None

        sd_name = (payload.get("study_designs") or "").strip()
        if sd_name:
            sd_obj, _ = StudyDesign.objects.get_or_create(design_name=sd_name)
            study.study_designs = sd_obj
        else:
            study.study_designs = None

        study.sample_size = payload.get("sample_size") or None
        study.age_range = payload.get("age_range") or None
        study.mean_age = payload.get("mean_age") or None
        study.male_female_split = payload.get("male_female_split") or None

        study.citation = _clean_int(payload.get("citation"), default=0) or 0
        study.keyword = payload.get("keyword") or None
        study.date = payload.get("date") or None
        study.pages = payload.get("pages") or None
        study.issue = payload.get("issue") or None
        study.volume = payload.get("volume") or None
        study.automatic_tags = payload.get("automatic_tags") or None

        study.authors_affiliations = _normalize_authors_affiliations(payload.get("authors_affiliations"))

        study.biological_risk_factor_studied = payload.get("biological_risk_factor_studied") or None
        study.biological_rationale_provided = payload.get("biological_rationale_provided") or None
        study.status_of_corresponding_gene = payload.get("status_of_corresponding_gene") or None
        study.technology_platform = payload.get("technology_platform") or None
        study.evaluation_method = payload.get("evaluation_method") or None
        study.statistical_model = payload.get("statistical_model") or None
        study.criteria_for_significance = payload.get("criteria_for_significance") or None
        study.validation_performed = payload.get("validation_performed") or None
        study.findings_conclusions = payload.get("findings_conclusions") or None
        study.generalisability_of_conclusion = payload.get("generalisability_of_conclusion") or None
        study.adequate_statistical_powered = payload.get("adequate_statistical_powered") or None
        study.comment = payload.get("comment") or None
        study.should_exclude = bool(payload.get("should_exclude", False))

        study.save()

        # M2M
        study.countries.set(_upsert_m2m(Country, _ensure_list_str(payload.get("countries")), "name"))
        study.article_type.set(_upsert_m2m(ArticleType, _ensure_list_str(payload.get("article_type")), "article_name"))
        study.disorder.set(_upsert_m2m(Disorder, _ensure_list_str(payload.get("disorder")), "disorder_name"))
        study.biological_modalities.set(_upsert_m2m(BiologicalModality, _ensure_list_str(payload.get("biological_modalities")), "modality_name"))
        study.genetic_source_materials.set(_upsert_m2m(GeneticSourceMaterial, _ensure_list_str(payload.get("genetic_source_materials")), "material_type"))

        # StudyDocument: only store PDF (no text extraction here)
        doc = None
        if pdf_file:
            doc, _ = StudyDocument.objects.get_or_create(study=study)
            doc.pdf_file = pdf_file  # just save the file
            # DO NOT touch doc.extracted_text here
            doc.save()
        else:
            # If no file was provided but a document exists, leave it untouched
            existing = StudyDocument.objects.filter(study=study).first()
            if existing:
                doc = existing

        return Response(
            {
                "study_id": study.id,
                "created": created,
                "document_id": doc.id if doc else None,
                "message": "Study saved",
            },
            status=status.HTTP_201_CREATED if created else status.HTTP_200_OK,
        )
