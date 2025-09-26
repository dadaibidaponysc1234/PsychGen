# signals.py
from django.db import transaction
from django.db.models.signals import pre_save, post_save
from django.dispatch import receiver
from .models import StudyDocument
from services.embedding_pipeline import process_study_pdf_with_langchain

@receiver(pre_save, sender=StudyDocument)
def _flag_pdf_change(sender, instance, **kwargs):
    if not instance.pk:
        instance._pdf_changed = True  # new object; treat as changed
        return
    old = sender.objects.filter(pk=instance.pk).only("pdf_file").first()
    instance._pdf_changed = bool(old and old.pdf_file != instance.pdf_file)

@receiver(post_save, sender=StudyDocument)
def auto_index_pdf(sender, instance, created, **kwargs):
    # run on create or when pdf file actually changed
    if created or getattr(instance, "_pdf_changed", False):
        def run():
            try:
                process_study_pdf_with_langchain(instance)
            except Exception as e:
                # replace with proper logging
                print(f"[auto-index] StudyDocument {instance.id} failed: {e}")
        # delay until DB commit to avoid reading a file before it’s persisted
        transaction.on_commit(run)
