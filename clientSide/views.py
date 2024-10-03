from rest_framework import generics
from ResearchApp.models import Study, Disorder, BiologicalModality, GeneticSourceMaterial, ArticleType, StudyDesign, Country
from .serializers import StudySerializer,DisorderStudyCountSerializer,CountryStudyCountSerializer,BiologicalModalityStudyCountSerializer,GeneticSourceMaterialStudyCountSerializer,YearlyStudyCountSerializer,CountrySerializer
from django.db.models import Count, Q, Sum, Avg
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.pagination import PageNumberPagination
from rest_framework import status
from django.http import JsonResponse
from collections import defaultdict

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

# class StudyListView(generics.ListCreateAPIView):
#     serializer_class = StudySerializer
#     pagination_class = StudyListPagination

#     def get_queryset(self):
#         queryset = Study.objects.all()
        
#         title = self.request.GET.get('title')
#         year = self.request.GET.get('year')
#         countries = self.request.GET.get('research_regions')
#         disorder = self.request.GET.get('disorder')
#         article_type = self.request.GET.get('article_type')

#         if title:
#             queryset = queryset.filter(Q(title__icontains=title) | Q(abstract__icontains=title))
#         else:
#             queryset = Study.objects.all()
#         if disorder:
#             queryset = queryset.filter(disorder__disorder_name__icontains=disorder)
#         if countries:
#             queryset = queryset.filter(countries__name__icontains=countries)
#         if article_type:
#             queryset = queryset.filter(article_type__article_name__icontains=article_type)
#         if year:
#             queryset = queryset.filter(year=year)

#         return queryset


class StudyListView(generics.ListCreateAPIView):
    serializer_class = StudySerializer
    pagination_class = StudyListPagination

    def get_queryset(self):
        queryset = Study.objects.all()

        # Filter parameters
        title = self.request.GET.get('title')
        year = self.request.GET.get('year')
        countries = self.request.GET.get('research_regions')
        disorder = self.request.GET.get('disorder')
        article_type = self.request.GET.get('article_type')

        # Title filtering
        if title:
            queryset = queryset.filter(Q(title__icontains=title) | Q(abstract__icontains=title))

        # Other filters
        if disorder:
            queryset = queryset.filter(disorder__disorder_name__icontains=disorder)
        if countries:
            queryset = queryset.filter(countries__name__icontains=countries)
        if article_type:
            queryset = queryset.filter(article_type__article_name__icontains=article_type)
        if year:
            queryset = queryset.filter(year=year)

        return queryset

    def list(self, request, *args, **kwargs):
        queryset = self.get_queryset()

        # Count studies per disorder after filtering
        disorder_counts = (
            queryset.values('disorder__disorder_name')  # Use disorder name
            .annotate(study_count=Count('id'))  # Count studies per disorder
            .order_by('-study_count')
        )

        # Generate a response with the filtered studies and the disorder count
        response = super().list(request, *args, **kwargs)  # Get the serialized study list response
        
        # Add disorder counts to the response data
        response.data['disorder_study_counts'] = disorder_counts

        return Response(response.data)


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
    study_content = [study.title + " " + (study.findings_conclusions or "")+(study.abstract or "")] + \
                    [s.title + " " + (s.findings_conclusions or "") + (s.abstract or "") for s in all_studies]
    
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
        
# class ResearchRegionStudyCountView(APIView):
#     def get(self, request):
#         # Group by research region and count the number of studies
#         country_counts = (
#             Study.objects.values('countries__name')
#             .annotate(study_count=Count('id'))
#             .order_by('-study_count')
#         )

#         # Serialize the data
#         serializer = CountryStudyCountSerializer(country_counts, many=True)
#         return Response(serializer.data)
AFRICAN_COUNTRIES = [
    "ALGERIA", "ANGOLA", "BENIN", "BOTSWANA", "BURKINA FASO", "BURUNDI", "CABO VERDE", "CAMEROON",
    "CENTRAL AFRICAN REPUBLIC", "CHAD", "COMOROS", "DEMOCRATIC REPUBLIC OF THE CONGO", "REPUBLIC OF THE CONGO",
    "DJIBOUTI", "EGYPT", "EQUATORIAL GUINEA", "ERITREA", "ESWATINI", "ETHIOPIA", "GABON", "GAMBIA", "GHANA", "GUINEA",
    "GUINEA-BISSAU", "IVORY COAST", "KENYA", "LESOTHO", "LIBERIA", "LIBYA", "MADAGASCAR", "MALAWI", "MALI",
    "MAURITANIA", "MAURITIUS", "MOROCCO", "MOZAMBIQUE", "NAMIBIA", "NIGER", "NIGERIA", "RWANDA", "SÃO TOMÉ AND PRÍNCIPE",
    "SENEGAL", "SEYCHELLES", "SIERRA LEONE", "SOMALIA", "SOUTH AFRICA", "SOUTH SUDAN", "SUDAN", "TANZANIA", "TOGO",
    "TUNISIA", "UGANDA", "ZAMBIA", "ZIMBABWE"
]

class ResearchRegionStudyCountView(APIView):
    def get(self, request):
        # Filter studies by predefined African countries
        african_countries = Country.objects.filter(name__in=AFRICAN_COUNTRIES)
        print(african_countries)

        # Group by African countries and count the number of studies
        country_counts = (
            Study.objects.filter(countries__in=african_countries)  # Filter studies for African countries
            .values('countries__name')
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


class CountryCollaborationView(APIView):
    def get(self, request):
        try:
            # Step 1: Retrieve all studies and their associated countries
            studies = Study.objects.prefetch_related('countries').all()
            # studies = studies[69:]
            # print(f'studies: {studies}')

            # Initialize collaboration dictionary
            country_collaborations = defaultdict(lambda: defaultdict(int))
            # print(f'studies: {country_collaborations}')
            for study in studies:
                # Get all the countries related to this study
                countries = list(study.countries.values_list('name', flat=True))
                # countries = countries[1:]
                # print(f'studies: {countries}')
                if countries:
                    # Process the countries for collaboration tracking
                    for i, country1 in enumerate(countries):
                        for country2 in countries[i+1:]:
                            # Increment collaboration count between countries
                            country_collaborations[country1][country2] += 1
                            country_collaborations[country2][country1] += 1
                        country_collaborations[country1][country1] += 1
                        country_collaborations[country1][country1] += 1
                        # print(f'studies: {countries}--------> country_collaborations:{country_collaborations}')
                    # break
            # Step 2: Generate a list of unique countries
            all_countries = sorted(country_collaborations.keys())

            # Step 3: Create a matrix of collaborations
            matrix = []
            for country1 in all_countries:
                row = []
                for country2 in all_countries:
                    row.append(country_collaborations[country1].get(country2, 0))
                matrix.append(row)

            # Step 4: Prepare the response
            response_data = {
                'matrix': matrix,
                'countries': all_countries
            }

            return Response(response_data, status=status.HTTP_200_OK)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# from collections import defaultdict
# # from rest_framework.views import APIView
# # from django.http import JsonResponse
# # from clientSide.models import Study

# class CountryCollaborationView(APIView):
#     def get(self, request, *args, **kwargs):
#         try:
#             # Retrieve all studies from the database
#             study_details = Study.objects.all()

#             # Initialize a set for all countries and a dictionary for country co-occurrences
#             all_countries = set()
#             co_occurrences = defaultdict(lambda: defaultdict(int))

#             for study in study_details:
#                 # Get the list of countries related to this study
#                 countries = study.countries.all()

#                 # Extract the country names and clean them if necessary
#                 country_names = [country.name.strip() for country in countries]

#                 # Update the set of all countries
#                 all_countries.update(country_names)

#                 # Count co-occurrences between countries
#                 for i in range(len(country_names)):
#                     for j in range(i + 1, len(country_names)):
#                         country_a = country_names[i]
#                         country_b = country_names[j]
#                         co_occurrences[country_a][country_b] += 1
#                         co_occurrences[country_b][country_a] += 1  # Symmetric co-occurrence

#             # Convert all_countries to a sorted list
#             all_countries = sorted(list(all_countries))

#             # Initialize the connection matrix (n x n) for the chord diagram
#             connection_matrix = []
#             for country_a in all_countries:
#                 row = []
#                 for country_b in all_countries:
#                     row.append(co_occurrences[country_a].get(country_b, 0))
#                 connection_matrix.append(row)

#             # Prepare the response data
#             response_data = {
#                 'matrix': connection_matrix,
#                 'countries': all_countries
#             }

#             # Return the response as JSON
#             return JsonResponse(response_data, safe=False)

#         except Exception as e:
#             # Return an error response if something goes wrong
#             return JsonResponse({
#                 'status': 'error',
#                 'message': str(e)
#             }, status=500)
