from django.urls import path
from .views import UploadCSVView, DownloadCSVExampleView,VisitorCountAPIView,LoginAPIView,LogoutAPIView

urlpatterns = [
    path('upload-csv/', UploadCSVView.as_view(), name='upload-csv'),
    path('download-csv-example/', DownloadCSVExampleView.as_view(), name='download_csv_example'),
    path('visitor-count/', VisitorCountAPIView.as_view(), name='visitor-count'),
    # path("login/", LoginAPIView.as_view(), name="login"),
    path('login/', LoginAPIView.as_view(), name='login'),
    path('logout/', LogoutAPIView.as_view(), name='logout'),

]
