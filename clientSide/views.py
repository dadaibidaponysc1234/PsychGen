from rest_framework import generics
from ResearchApp.models import Study, Disorder, ResearchRegion, BiologicalModality, GeneticSourceMaterial,  ArticleType
from .serializers import StudySerializer,DisorderStudyCountSerializer,ResearchRegionStudyCountSerializer,BiologicalModalityStudyCountSerializer,GeneticSourceMaterialStudyCountSerializer,YearlyStudyCountSerializer
from django.db.models import Count
from rest_framework.response import Response
from rest_framework.views import APIView

class StudyListView(generics.ListCreateAPIView):
    serializer_class = StudySerializer

    def get_queryset(self):
        queryset = Study.objects.all()
        # queryset = Study.objects.filter(should_exclude=True)
        title = self.request.GET.get('title')
        pmid = self.request.GET.get('pmid')
        year = self.request.GET.get('year')

        if title:
            queryset = queryset.filter(title__icontains=title)
        if pmid:
            queryset = queryset.filter(pmid__icontains=pmid)
        if year:
            queryset = queryset.filter(year=year)

        return queryset
    
class StudyDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Study.objects.all()
    serializer_class = StudySerializer

# class DisorderListView(generics.ListCreateAPIView):
#     queryset = Disorder.objects.all()
#     serializer_class = DisorderSerializer

# class DisorderDetailView(generics.RetrieveUpdateDestroyAPIView):
#     queryset = Disorder.objects.all()
#     serializer_class = DisorderSerializer

# Similarly, create views for other models as needed

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
            .order_by('-year')  # Order by year, descending
        )

        # Prepare data for the serializer
        data = [
            {'year': item['year'], 'study_count': item['study_count']}
            for item in yearly_study_counts
        ]

        # Serialize the data
        serializer = YearlyStudyCountSerializer(data, many=True)
        return Response(serializer.data)    