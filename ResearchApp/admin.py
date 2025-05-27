from django.contrib import admin
from .models import (Study, Disorder, BiologicalModality, GeneticSourceMaterial, 
                        ArticleType, StudyDesign, Country,
                        StudyDocument, StudyImage, SavedResponse,
                        ChatSession,ChatMessage)


admin.site.register(Disorder)
admin.site.register(Study)
admin.site.register(Country)

admin.site.register(BiologicalModality)
admin.site.register(GeneticSourceMaterial)

admin.site.register(ArticleType)

admin.site.register(StudyDocument)
admin.site.register(StudyImage)
admin.site.register(SavedResponse)

admin.site.register(ChatSession)
admin.site.register(ChatMessage)