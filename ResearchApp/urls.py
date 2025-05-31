from django.urls import path
from .views import (UploadCSVView, DownloadCSVExampleView,VisitorCountAPIView,
                    LogoutView, UploadPDFView,StudyImageUploadView,
                    StudyImageListView,StudyImageDetailView,ContinueChatView, 
                    ChatSessionListView, DeleteChatSessionView, DeleteChatMessageView)
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

# ,LoginAPIView,LogoutAPIView
urlpatterns = [
    path('upload-csv/', UploadCSVView.as_view(), name='upload-csv'),
    path('download-csv-example/', DownloadCSVExampleView.as_view(), name='download_csv_example'),
    path('visitor-count/', VisitorCountAPIView.as_view(), name='visitor-count'),
    
    path('login/', TokenObtainPairView.as_view(), name='login'),
    path('login/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('logout/', LogoutView.as_view(), name='token_logout'),

    #for the generative AI
    path('ai/upload-pdf/', UploadPDFView.as_view(), name='upload-pdf'),
    path('ai/images/', StudyImageListView.as_view(), name='image-list'),
    path('ai/images/upload/', StudyImageUploadView.as_view(), name='image-upload'),
    path('ai/images/<int:pk>/', StudyImageDetailView.as_view(), name='image-detail'),

    
    path("ai/chat/", ContinueChatView.as_view(), name="continue-chat"),
    path("ai/chat/sessions/", ChatSessionListView.as_view(), name="chat-sessions"),
    path("ai/chat/session/<int:pk>/delete/", DeleteChatSessionView.as_view(), name="delete-chat-session"),
    path("ai/chat/message/<int:pk>/delete/", DeleteChatMessageView.as_view(), name="delete-chat-message"),

]
