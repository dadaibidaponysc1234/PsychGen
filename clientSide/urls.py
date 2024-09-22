from django.urls import path
# from . import views
from .views import StudyListView, StudyDetailView,DisorderStudyCountView, ResearchRegionStudyCountView,BiologicalModalityStudyCountView,GeneticSourceMaterialStudyCountView,YearlyStudyCountView,StudyDeleteView,StudyBulkDeleteView,AutoCompleteSuggestionView, CountryCollaborationView

urlpatterns = [
    path('studies/', StudyListView.as_view(), name='study-list'),
    path('studies/<int:pk>/', StudyDetailView.as_view(), name='study-detail'),
    path('studies/delete/<int:pk>/', StudyDeleteView.as_view(), name='study-delete'),
    path('studies/delete-multiple/', StudyBulkDeleteView.as_view(), name='study-bulk-delete'),
    # path('disorders/', DisorderListView.as_view(), name='disorder-list'),
    # path('disorders/<int:pk>/', DisorderDetailView.as_view(), name='disorder-detail'),
    # Add other URL patterns as needed
    path('disorder-study-count/', DisorderStudyCountView.as_view(), name='disorder-study-count'),
    path('research-region-study-count/', ResearchRegionStudyCountView.as_view(), name='research-region-study-count'),
    path('biological-modality-study-count/', BiologicalModalityStudyCountView.as_view(), name='biological-modality-study-count'),
    path('genetic-source-material-study-count/', GeneticSourceMaterialStudyCountView.as_view(), name='genetic-source-material-study-count'),
    path('yearly-study-count/', YearlyStudyCountView.as_view(), name='yearly-study-count'),
    path('suggestions/', AutoCompleteSuggestionView.as_view(), name='autocomplete-suggestions'),
    path('country-collaboration/', CountryCollaborationView.as_view(), name='country-collaboration'),

]
