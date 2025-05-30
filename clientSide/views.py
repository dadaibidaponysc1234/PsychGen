import csv
from rest_framework import generics
from ResearchApp.models import Study, Disorder, BiologicalModality, GeneticSourceMaterial, ArticleType, StudyDesign, Country
from .serializers import (StudySerializer,DisorderStudyCountSerializer,CountryStudyCountSerializer,
                            BiologicalModalityStudyCountSerializer,GeneticSourceMaterialStudyCountSerializer,
                            YearlyStudyCountSerializer,CountrySerializer,GetShortStudySerializer)
from django.db.models import Count, Q, Sum, Avg
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.pagination import PageNumberPagination
from rest_framework import status
from django.http import JsonResponse
from collections import defaultdict
from django.http import HttpResponse
from collections import Counter
from rest_framework.permissions import IsAuthenticated
from rest_framework.generics import ListAPIView

# from rest_framework.permissions import IsAuthenticated

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from django.db.models import


AFRICAN_COUNTRIES = [
    "ALGERIA", "ANGOLA", "BENIN", "BOTSWANA", "BURKINA FASO", "BURUNDI", "CABO VERDE", "CAMEROON",
    "CENTRAL AFRICAN REPUBLIC", "CHAD", "COMOROS", "DEMOCRATIC REPUBLIC OF THE CONGO", "REPUBLIC OF THE CONGO",
    "DJIBOUTI", "EGYPT", "EQUATORIAL GUINEA", "ERITREA", "ESWATINI", "ETHIOPIA", "GABON", "GAMBIA", "GHANA", "GUINEA",
    "GUINEA-BISSAU", "IVORY COAST", "KENYA", "LESOTHO", "LIBERIA", "LIBYA", "MADAGASCAR", "MALAWI", "MALI",
    "MAURITANIA", "MAURITIUS", "MOROCCO", "MOZAMBIQUE", "NAMIBIA", "NIGER", "NIGERIA", "RWANDA", "SÃO TOMÉ AND PRÍNCIPE",
    "SENEGAL", "SEYCHELLES", "SIERRA LEONE", "SOMALIA", "SOUTH AFRICA", "SOUTH SUDAN", "SUDAN", "TANZANIA", "TOGO",
    "TUNISIA", "UGANDA", "ZAMBIA", "ZIMBABWE"
]

class AutoCompleteSuggestionView(APIView):
    def get(self, request):
        query = request.GET.get('query', '')
        
        if not query:
            return Response({"error": "No query provided"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Fetch suggestions including the ID
        study_suggestions = Study.objects.filter(title__icontains=query).values('id','title')[:5]
        disorder_suggestions = Disorder.objects.filter(disorder_name__icontains=query).values('id','disorder_name')[:5]
        author_suggestions = Study.objects.filter(lead_author__icontains=query).values('id', 'lead_author')[:5]

        # Combine suggestions for title, disorders, and authors with their IDs
        suggestions = {
            "study_titles": list(study_suggestions),
            "disorders": list(disorder_suggestions),
            "author":list(author_suggestions)
        }

        return Response(suggestions, status=status.HTTP_200_OK)


# the pagination class
class StudyListPagination(PageNumberPagination):
    page_size=10


class CountryListView(APIView):
    def get(self, request, *args, **kwargs):
        countries = Country.objects.all().values('id', 'name')
        return Response(countries)

class DisorderListView(APIView):
    def get(self, request, *args, **kwargs):
        disorders = Disorder.objects.all().values('id', 'disorder_name')
        return Response(disorders)

class ArticleTypeListView(APIView):
    def get(self, request, *args, **kwargs):
        article_types = ArticleType.objects.all().values('id', 'article_name')
        return Response(article_types)

class BiologicalModalityListView(APIView):
    def get(self, request, *args, **kwargs):
        modalities = BiologicalModality.objects.all().values('id', 'modality_name')
        return Response(modalities)

class GeneticSourceMaterialListView(APIView):
    def get(self, request, *args, **kwargs):
        materials = GeneticSourceMaterial.objects.all().values('id', 'material_type')
        return Response(materials)

class StudyListView(generics.ListCreateAPIView):
    serializer_class = StudySerializer
    pagination_class = StudyListPagination
    # permission_classes = [IsAuthenticated]  # Restrict access to authenticated users only

    def get_queryset(self):
        queryset = Study.objects.all()

        # Get filter parameters from request
        title = self.request.GET.get('title')
        journal_name = self.request.GET.get('journal_name')
        keyword = self.request.GET.get('keyword')
        impact_factor_min = self.request.GET.get('impact_factor_min')  # Range filter for impact factor
        impact_factor_max = self.request.GET.get('impact_factor_max')

        year_min = self.request.GET.get('year_min')
        year_max = self.request.GET.get('year_max')
       
        year = self.request.GET.get('year')
        countries = self.request.GET.getlist('research_regions')  # Expect multiple countries
        disorder = self.request.GET.getlist('disorder')  # Expect multiple disorders
        article_type = self.request.GET.getlist('article_type')  # Expect multiple article types
        biological_modalities = self.request.GET.getlist('biological_modalities')  # Expect multiple modalities
        genetic_source_materials = self.request.GET.getlist('genetic_source_materials')  # Multiple materials
        
        # Build Q object to search across all fields
        search_query = Q()

        # Title and abstract filtering (searches both title and abstract)
        if title:
            search_query &= (Q(title__icontains=title) | Q(abstract__icontains=title))

        # Filter by year
        if year:
            search_query &= Q(year=year)

        # Filter by disorder name (ManyToManyField)
        if disorder:
            search_query &= Q(disorder__disorder_name__in=disorder)

        # Filter by article type (ManyToManyField)
        if article_type:
            search_query &= Q(article_type__article_name__in=article_type)

        # Filter by country name (ManyToManyField)
        if countries:
            search_query &= Q(countries__name__in=countries)
        
        # Filter by journal name
        if journal_name:
            search_query &= Q(journal_name__icontains=journal_name)

        # Filter by keyword
        if keyword:
            search_query &= Q(keyword__icontains=keyword)

        # Filter by impact factor range
        if impact_factor_min:
            search_query &= Q(impact_factor__gte=impact_factor_min)
        if impact_factor_max:
            search_query &= Q(impact_factor__lte=impact_factor_max)
        
        # Filter by impact factor range
        if year_min:
            search_query &= Q(year__gte=year_min)
        if year_max:
            search_query &= Q(year__lte=year_max)
        
        # Filter by biological modalities (ManyToManyField)
        if biological_modalities:
            search_query &= Q(biological_modalities__modality_name__in=biological_modalities)

        # Filter by genetic source materials (ManyToManyField)
        if genetic_source_materials:
            search_query &= Q(genetic_source_materials__material_type__in=genetic_source_materials)

        # Apply the search query to the queryset
        queryset = queryset.filter(search_query).distinct()

        return queryset

    def list(self, request, *args, **kwargs):
        queryset = self.get_queryset()

        #check if export to CSV is requested 
        export_format = request.GET.get('export', None)
        if export_format == 'csv':
            return self.export_to_csv(queryset)

        # Count studies per disorder after filtering
        disorder_counts = (
            queryset.values('disorder__disorder_name')  # Use disorder name
            .annotate(study_count=Count('id'))  # Count studies per disorder
            .order_by('-study_count')
        )

        # Count studies per year after filtering
        yearly_counts = (
            queryset.values('year')  # Group by year
            .annotate(study_count=Count('id'),
                    total_citations=Sum('citation'),
                    average_impact_factor=Avg('impact_factor'))  # Count studies per year
            .order_by('year')  # Optional: Order by year (ascending)
        )

        african_countries = Country.objects.filter(name__in=AFRICAN_COUNTRIES)
        african_study_counts = (
            queryset.filter(countries__in=african_countries)
            .values('countries__name')
            .annotate(study_count=Count('id'))
            .order_by('-study_count')
        )

        # Count studies for each biological modality
        biological_modality_counts = (
            queryset.values('biological_modalities__modality_name')
            .annotate(study_count=Count('id'))
            .order_by('-study_count')
        )

         # Count studies for each genetic source material
        genetic_source_material_counts = (
            queryset.values('genetic_source_materials__material_type')
            .annotate(study_count=Count('id'))
            .order_by('-study_count')
        )

        # ---- Country Collaboration Logic ---- #
        try:
            # Step 1: Retrieve all filtered studies and their associated countries
            studies = queryset.prefetch_related('countries').all()

            # Initialize collaboration dictionary
            country_collaborations = defaultdict(lambda: defaultdict(int))

            for study in studies:
                # Get all the countries related to this study
                countries = list(study.countries.values_list('name', flat=True))
                if countries:
                    # Process the countries for collaboration tracking
                    for i, country1 in enumerate(countries):
                        for country2 in countries[i+1:]:
                            # Increment collaboration count between countries
                            country_collaborations[country1][country2] += 1
                            country_collaborations[country2][country1] += 1
                        country_collaborations[country1][country1] += 1

            # Step 2: Generate a list of unique countries
            all_countries = sorted(country_collaborations.keys())

            # Step 3: Create a matrix of collaborations
            matrix = []
            for country1 in all_countries:
                row = []
                for country2 in all_countries:
                    row.append(country_collaborations[country1].get(country2, 0))
                matrix.append(row)

            # Prepare the collaboration response data
            collaboration_data = {
                'matrix': matrix,
                'countries': all_countries
            }

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Generate a response with the filtered studies and the disorder count
        response = super().list(request, *args, **kwargs)  # Get the serialized study list response

        # Add disorder counts to the response data
        response.data['disorder_study_counts'] = disorder_counts
        response.data['yearly_study_counts'] = yearly_counts
        response.data['african_study_counts'] = african_study_counts
        response.data['biological_modality_study_counts'] = biological_modality_counts
        response.data['genetic_source_material_study_counts'] = genetic_source_material_counts
        response.data['collaboration_data'] = collaboration_data
        return Response(response.data)
    def export_to_csv(self, queryset):
        """
        Exports the filtered queryset to CSV format.
        """
        # Define the fields to include in the CSV
        fields = [
            'pmid','lead_author', 'title', 'journal_name', 'year', 'impact_factor', 'countries', 'disorder', 
            'article_type', 'biological_modalities', 'genetic_source_materials'
        ]

        # Create the HttpResponse object with CSV header
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="studies.csv"'

        writer = csv.writer(response)
        # Write header
        writer.writerow(fields)

        # Write data rows
        for study in queryset:
            writer.writerow([
                study.pmid,
                study.lead_author,
                study.title,
                study.journal_name,
                study.year,
                study.impact_factor,
                ', '.join([country.name for country in study.countries.all()]),  # Convert ManyToManyField to string
                ', '.join([disorder.disorder_name for disorder in study.disorder.all()]),  # Convert ManyToManyField to string
                # study.study_design.design_name if study.study_design else 'N/A',  # ForeignKey (optional)
                ', '.join([article.article_name for article in study.article_type.all()]),  # Convert ManyToManyField to string
                ', '.join([modality.modality_name for modality in study.biological_modalities.all()]),  # Convert ManyToManyField to string
                ', '.join([material.material_type for material in study.genetic_source_materials.all()])  # Convert ManyToManyField to string
            ])

        return response    


class StudyDeleteView(generics.DestroyAPIView):
    permission_classes = [IsAuthenticated]
    queryset = Study.objects.all()
    serializer_class = StudySerializer

    def delete(self, request, *args, **kwargs):
        instance = self.get_object()
        self.perform_destroy(instance)
        return Response(status=status.HTTP_204_NO_CONTENT)
        

class StudyBulkDeleteView(APIView):
    permission_classes = [IsAuthenticated]

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



class TopFiveDisordersYearlyView(APIView):
    def get(self, request):
        # Step 1: Get the total study count per disorder and find the top five disorders
        top_disorders = (
            Study.objects.values('disorder__disorder_name')  # Group by disorder name
            .annotate(total_study_count=Count('id'))  # Get the total study count per disorder
            .order_by('-total_study_count')[:5]  # Limit to top five disorders
        )

        # Extract the names of the top five disorders
        top_disorder_names = [disorder['disorder__disorder_name'] for disorder in top_disorders]

        # Step 2: Get the yearly study count for these top five disorders
        disorder_yearly_counts = (
            Study.objects.filter(disorder__disorder_name__in=top_disorder_names)  # Filter for top disorders
            .values('disorder__disorder_name', 'year')  # Group by disorder and year
            .annotate(study_count=Count('id'))  # Count the studies for each year
            .order_by('year')  # Optional: order by year
        )

        # Step 3: Organize the results
        # Create a dictionary to store the yearly counts for each disorder
        result = {}
        for disorder_name in top_disorder_names:
            result[disorder_name] = {}

        for entry in disorder_yearly_counts:
            disorder_name = entry['disorder__disorder_name']
            year = entry['year']
            count = entry['study_count']

            # Append the study count to the appropriate disorder and year
            result[disorder_name][year] = count

        # Step 4: Return the response
        return Response(result)


class WordCloudView(APIView):
    def get(self, request):
        # Step 1: Get all keywords from studies
        keywords = Study.objects.values_list('keyword', flat=True)

        # Step 2: Split each keyword string into words and count occurrences
        word_counter = Counter()
        for keyword_string in keywords:
            if keyword_string:  # Check if the keyword string is not empty
                # Replace full stops, split by ';', strip whitespace, and convert to sentence case
                words = [
                    word.strip().replace('.', '').capitalize()  # Remove full stops and convert to sentence case
                    for word in keyword_string.replace(',', ';').split(';')
                ]
                word_counter.update(words)  # Update the count

        # Step 3: Structure the data for word cloud
        word_cloud_data = [
            {'text': word, 'size': count * 6}  # Scale size for visibility
            for word, count in word_counter.items()
        ]

        # Step 4: Return the response
        return Response(word_cloud_data)


# views.py
class ShortStudyListView(ListAPIView):
    queryset = Study.objects.all()
    serializer_class = GetShortStudySerializer
