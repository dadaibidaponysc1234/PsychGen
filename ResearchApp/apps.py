# ResearchApp/apps.py
from django.apps import AppConfig

class ResearchAppConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "ResearchApp"  # must match your package folder exactly

    def ready(self):
        # Ensure signal handlers are connected
        from . import signal  # noqa: F401
