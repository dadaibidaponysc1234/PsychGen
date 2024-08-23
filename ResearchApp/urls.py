from django.urls import path
from .views import UploadCSVView, download_csv_example

urlpatterns = [
    path('upload-csv/', UploadCSVView.as_view(), name='upload-csv'),
    path('download-csv-example/', download_csv_example, name='download_csv_example')
]
