from rest_framework import generics
from ResearchApp.models import Study, Disorder, ResearchRegion, BiologicalModality, GeneticSourceMaterial,  ArticleType
from .serializers import StudySerializer,DisorderStudyCountSerializer,ResearchRegionStudyCountSerializer,BiologicalModalityStudyCountSerializer,GeneticSourceMaterialStudyCountSerializer,YearlyStudyCountSerializer
from django.db.models import Count, Q
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.pagination import PageNumberPagination
from rest_framework import status

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from django.db.models import

# the pagination class
class StudyListPagination(PageNumberPagination):
    page_size=10

class StudyListView(generics.ListCreateAPIView):
    serializer_class = StudySerializer
    pagination_class = StudyListPagination

    def get_queryset(self):
        queryset = Study.objects.all()
        
        title = self.request.GET.get('title')
        year = self.request.GET.get('year')
        research_regions = self.request.GET.get('research_regions')
        disorder = self.request.GET.get('disorder')
        article_type = self.request.GET.get('article_type')


        if title:
            queryset = queryset.filter(Q(title__icontains=title) | Q(lead_author__icontains=title))
        if disorder:
            queryset = queryset.filter(disorder__disorder_name__icontains=disorder)
        if research_regions:
            queryset = queryset.filter(research_regions__name__icontains=research_regions)
        if article_type:
            queryset = queryset.filter(article_type__article_name__icontains=article_type)
        if year:
            queryset = queryset.filter(year=year)

        return queryset

class StudyDeleteView(generics.DestroyAPIView):
    queryset = Study.objects.all()
    serializer_class = StudySerializer

    def delete(self, request, *args, **kwargs):
        instance = self.get_object()
        self.perform_destroy(instance)
        return Response(status=status.HTTP_204_NO_CONTENT)
        

class StudyBulkDeleteView(APIView):

    def post(self, request, *args, **kwargs):
        ids = request.data.get('ids', [])
        if not ids:
            return Response({"error": "No IDs provided"}, status=status.HTTP_400_BAD_REQUEST)
        
        studies = Study.objects.filter(id__in=ids)
        if not studies.exists():
            return Response({"error": "Studies not found"}, status=status.HTTP_404_NOT_FOUND)
        
        studies.delete()
        return Response({"message": "Studies deleted successfully"}, status=status.HTTP_200_OK)
#function for the recommender 
def recommend_similar_studies(study, top_n=5):
    # Get all studies except the current one
    all_studies = Study.objects.exclude(id=study.id)
    
    # Combine title and findings/conclusions to form the content
    study_content = [study.title + " " + (study.findings_conclusions or "")] + \
                    [s.title + " " + (s.findings_conclusions or "") for s in all_studies]
    
    # Calculate TF-IDF and cosine similarity
    tfidf = TfidfVectorizer(stop_words='english').fit_transform(study_content)
    cosine_sim = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    
    # Get the top N similar studies
    similar_indices = cosine_sim.argsort()[-top_n:][::-1]
    similar_studies = [all_studies[int(i)] for i in similar_indices]
    
    return similar_studies


class StudyDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Study.objects.all()
    serializer_class = StudySerializer

    def get_recommended_articles(self, study):
        return recommend_similar_studies(study)

    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()
        recommended_articles = self.get_recommended_articles(instance)
        serializer = self.get_serializer(instance)
        data = serializer.data
        data['recommended_articles'] = StudySerializer(recommended_articles, many=True).data
        return Response(data)


class DisorderStudyCountView(APIView):
    def get(self, request):
        # Group by disorder and count the number of studies
        disorder_counts = (
            Study.objects.values('disorder__disorder_name')  # Use 'disorder__disorder_name'
            .annotate(study_count=Count('id'))
            .order_by('-study_count')
        )

        # Serialize the data
        serializer = DisorderStudyCountSerializer(disorder_counts, many=True)
        return Response(serializer.data)
        
class ResearchRegionStudyCountView(APIView):
    def get(self, request):
        # Group by research region and count the number of studies
        research_region_counts = (
            Study.objects.values('research_regions__name')
            .annotate(study_count=Count('id'))
            .order_by('-study_count')
        )

        # Serialize the data
        serializer = ResearchRegionStudyCountSerializer(research_region_counts, many=True)
        return Response(serializer.data)
    
class BiologicalModalityStudyCountView(APIView):
    def get(self, request):
        # Annotate the count of studies for each biological modality
        biological_modality_counts = (
            Study.objects
            .values('biological_modalities__modality_name')  # Use the correct field name here
            .annotate(study_count=Count('id'))
            .order_by('-study_count')
        )

        # Prepare data for the serializer
        data = [
            {'modality_name': item['biological_modalities__modality_name'], 'study_count': item['study_count']}
            for item in biological_modality_counts
        ]

        # Serialize the data
        serializer = BiologicalModalityStudyCountSerializer(data, many=True)
        return Response(serializer.data)
    



class GeneticSourceMaterialStudyCountView(APIView):
    def get(self, request):
        # Annotate the count of studies for each genetic source material
        genetic_source_material_counts = (
            Study.objects
            .values('genetic_source_materials__material_type')
            .annotate(study_count=Count('id'))
            .order_by('-study_count')
        )

        # Prepare data for the serializer
        data = [
            {'material_type': item['genetic_source_materials__material_type'], 'study_count': item['study_count']}
            for item in genetic_source_material_counts
        ]

        # Serialize the data
        serializer = GeneticSourceMaterialStudyCountSerializer(data, many=True)
        return Response(serializer.data)
    

class YearlyStudyCountView(APIView):
    def get(self, request):
        # Annotate the count of studies for each year
        yearly_study_counts = (
            Study.objects
            .values('year')  # Group by year
            .annotate(study_count=Count('id'))  # Count the number of studies for each year
            .order_by('year')  # Order by year, descending
        )

        # Prepare data for the serializer
        data = [
            {'year': item['year'], 'study_count': item['study_count']}
            for item in yearly_study_counts
        ]

        # Serialize the data
        serializer = YearlyStudyCountSerializer(data, many=True)
        return Response(serializer.data)    