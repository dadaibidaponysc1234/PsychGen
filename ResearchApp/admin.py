from django.contrib import admin
from .models import Study, Disorder, BiologicalModality, GeneticSourceMaterial, ArticleType, StudyDesign, Country


admin.site.register(Disorder)
admin.site.register(Study)
admin.site.register(Country)
# admin.site.register(StudyDesign)
admin.site.register(BiologicalModality)
admin.site.register(GeneticSourceMaterial)
# admin.site.register(Remark)
admin.site.register(ArticleType)
# admin.site.register(AuthorRegion)
# admin.site.register(Author)


# Register your models here.
