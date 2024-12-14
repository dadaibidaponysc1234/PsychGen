from django.urls import path
from .views import UploadCSVView, DownloadCSVExampleView,VisitorCountAPIView,LogoutView
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

# ,LoginAPIView,LogoutAPIView
urlpatterns = [
    path('upload-csv/', UploadCSVView.as_view(), name='upload-csv'),
    path('download-csv-example/', DownloadCSVExampleView.as_view(), name='download_csv_example'),
    path('visitor-count/', VisitorCountAPIView.as_view(), name='visitor-count'),
    # path("login/", LoginAPIView.as_view(), name="login"),
    # path('login/', LoginAPIView.as_view(), name='login'),
    # path('logout/', LogoutAPIView.as_view(), name='logout'),
    path('login/', TokenObtainPairView.as_view(), name='login'),
    path('login/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('logout/', LogoutView.as_view(), name='token_logout'),

]
