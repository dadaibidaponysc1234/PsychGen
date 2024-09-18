from django.urls import path
from .views import (
    AboutPageListCreateView, AboutPageDetailView,
    MissionListCreateView, ObjectiveListCreateView,
    KeyFeatureListCreateView, TechnologyDevelopmentListCreateView,
    VisionListCreateView
)

urlpatterns = [
    path('about/', AboutPageListCreateView.as_view(), name='about-list-create'),
    path('about/<int:pk>/', AboutPageDetailView.as_view(), name='about-detail'),
    path('mission/', MissionListCreateView.as_view(), name='mission-list-create'),
    path('objective/', ObjectiveListCreateView.as_view(), name='objective-list-create'),
    path('key-feature/', KeyFeatureListCreateView.as_view(), name='keyfeature-list-create'),
    path('technology/', TechnologyDevelopmentListCreateView.as_view(), name='technology-list-create'),
    path('vision/', VisionListCreateView.as_view(), name='vision-list-create'),
]
