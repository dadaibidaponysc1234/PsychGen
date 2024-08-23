from django.contrib import admin
from .models import Disorder,ResearchRegion, BiologicalModality, GeneticSourceMaterial, Study,ArticleType,AuthorRegion


admin.site.register(Disorder)
admin.site.register(Study)
admin.site.register(ResearchRegion)
# admin.site.register(StudyDesign)
admin.site.register(BiologicalModality)
admin.site.register(GeneticSourceMaterial)
# admin.site.register(Remark)
admin.site.register(ArticleType)
admin.site.register(AuthorRegion)
# admin.site.register(Author)


# Register your models here.
