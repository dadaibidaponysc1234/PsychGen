from django.urls import path
from .views import UploadCSVView, DownloadCSVExampleView

urlpatterns = [
    path('upload-csv/', UploadCSVView.as_view(), name='upload-csv'),
    path('download-csv-example/', DownloadCSVExampleView.as_view(), name='download_csv_example')
]
