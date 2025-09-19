# signals.py
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import StudyDocument
from .utils_index import process_study_pdf_with_langchain

@receiver(post_save, sender=StudyDocument)
def auto_index_pdf(sender, instance, created, **kwargs):
    if created:
        try:
            process_study_pdf_with_langchain(instance)
        except Exception as e:
            # log the error; avoid raising
            print(f"[auto-index] StudyDocument {instance.id} failed: {e}")
            # In production, use proper logging instead of print    