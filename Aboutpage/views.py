from rest_framework import generics
# from .models import AboutPage
# from .serializers import AboutPageSerializer
from rest_framework import generics
from .models import (
    AboutPage, Mission, Objective, KeyFeature, 
    TechnologyDevelopment, Vision
)
from .serializers import (
    AboutPageSerializer, MissionSerializer, ObjectiveSerializer,
    KeyFeatureSerializer, TechnologyDevelopmentSerializer, VisionSerializer
)

# Create or List About Pages
class AboutPageListCreateView(generics.ListCreateAPIView):
    queryset = AboutPage.objects.all()
    serializer_class = AboutPageSerializer

# Retrieve, Update, or Delete a Specific About Page
class AboutPageDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = AboutPage.objects.all()
    serializer_class = AboutPageSerializer

class MissionListCreateView(generics.ListCreateAPIView):
    queryset = Mission.objects.all()
    serializer_class = MissionSerializer

class ObjectiveListCreateView(generics.ListCreateAPIView):
    queryset = Objective.objects.all()
    serializer_class = ObjectiveSerializer

class KeyFeatureListCreateView(generics.ListCreateAPIView):
    queryset = KeyFeature.objects.all()
    serializer_class = KeyFeatureSerializer

class TechnologyDevelopmentListCreateView(generics.ListCreateAPIView):
    queryset = TechnologyDevelopment.objects.all()
    serializer_class = TechnologyDevelopmentSerializer

class VisionListCreateView(generics.ListCreateAPIView):
    queryset = Vision.objects.all()
    serializer_class = VisionSerializer
