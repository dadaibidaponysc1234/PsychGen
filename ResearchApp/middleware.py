from .models import Visitor
from django.utils import timezone

class VisitorTrackingMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Retrieve the IP address of the visitor
        ip_address = request.META.get('REMOTE_ADDR')

        # Only create a visitor entry if the IP address is not None
        if ip_address:
            Visitor.objects.create(ip_address=ip_address, visit_date=timezone.now())

        response = self.get_response(request)
        return response
