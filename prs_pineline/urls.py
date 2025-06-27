from django.urls import path
from .views import (
    AutoMapColumnsView, UploadZipView, SaveMappingView,
    DownloadMappedCleanedFileView, ApplySavedMappingView, 
)

urlpatterns = [
    path("upload/", UploadZipView.as_view(), name="upload-zip"),
    path("mapping/auto-map/", AutoMapColumnsView.as_view(), name="auto-map"),
    path("mapping/save/", SaveMappingView.as_view(), name="save-mapping"),
    path("download/mapped-cleaned/", DownloadMappedCleanedFileView.as_view(), name="download-cleaned"),
    path("mapping/apply/", ApplySavedMappingView.as_view(), name="apply-saved-mapping"),
]
