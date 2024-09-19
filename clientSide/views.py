from rest_framework import generics
from ResearchApp.models import Study, Disorder, BiologicalModality, GeneticSourceMaterial, ArticleType, StudyDesign, Country
from .serializers import StudySerializer,DisorderStudyCountSerializer,CountryStudyCountSerializer,BiologicalModalityStudyCountSerializer,GeneticSourceMaterialStudyCountSerializer,YearlyStudyCountSerializer,CountrySerializer
from django.db.models import Count, Q, Sum, Avg
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.pagination import PageNumberPagination
from rest_framework import status
# from django.db.models import Count, Q, Sum, Avg  # Add Sum and Avg here

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from django.db.models import
class AutoCompleteSuggestionView(APIView):
    def get(self, request):
        query = request.GET.get('query', '')
        
        if not query:
            return Response({"error": "No query provided"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Fetch suggestions including the ID
        study_suggestions = Study.objects.filter(title__icontains=query).values('id', 'title')[:5]
        disorder_suggestions = Disorder.objects.filter(disorder_name__icontains=query).values('id', 'disorder_name')[:5]
        author_suggestions = Study.objects.filter(lead_author__icontains=query).values('id', 'lead_author')[:5]

        # Combine suggestions for title, disorders, and authors with their IDs
        suggestions = {
            "study_titles": list(study_suggestions),
            "disorders": list(disorder_suggestions),
            "authors": list(author_suggestions),
        }

        return Response(suggestions, status=status.HTTP_200_OK)


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
        countries = self.request.GET.get('research_regions')
        disorder = self.request.GET.get('disorder')
        article_type = self.request.GET.get('article_type')

        if title:
            queryset = queryset.filter(Q(title__icontains=title) | Q(lead_author__icontains=title))
        else:
            queryset = Study.objects.all()
        if disorder:
            queryset = queryset.filter(disorder__disorder_name__icontains=disorder)
        if countries:
            queryset = queryset.filter(countries__name__icontains=countries)
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
        country_counts = (
            Study.objects.values('countries__name')
            .annotate(study_count=Count('id'))
            .order_by('-study_count')
        )

        # Serialize the data
        serializer = CountryStudyCountSerializer(country_counts, many=True)
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

        
        # Serialize the data
        serializer = BiologicalModalityStudyCountSerializer(biological_modality_counts, many=True)
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

        
        serializer = GeneticSourceMaterialStudyCountSerializer(genetic_source_material_counts, many=True)
        return Response(serializer.data)
    

class YearlyStudyCountView(APIView):
    def get(self, request):
        # Annotate the count of studies, total citations, and average impact factor for each year
        yearly_study_data = (
            Study.objects
            .values('year')  # Group by year
            .annotate(
                study_count=Count('id'),  # Count the number of studies for each year
                total_citations=Sum('citation'),  # Sum of citations for each year
                average_impact_factor=Avg('impact_factor')  # Average impact factor for each year
            )
            .order_by('year')  # Order by year ascending
        )

        # Prepare the data for the serializer
        data = [
            {
                'year': item['year'],
                'study_count': item['study_count'],
                'citation': item['total_citations'],
                'impact_factor': item['average_impact_factor']
            }
            for item in yearly_study_data
        ]

        # Serialize the data
        serializer = YearlyStudyCountSerializer(data, many=True)
        
        return Response(serializer.data, status=status.HTTP_200_OK)
