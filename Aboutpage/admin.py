from django.contrib import admin
from .models import AboutPage, Mission, Objective, KeyFeature, TechnologyDevelopment, Vision

@admin.register(AboutPage)
class AboutPageAdmin(admin.ModelAdmin):
    list_display = ('title', 'last_updated')

admin.site.register(Mission)
admin.site.register(Objective)
admin.site.register(KeyFeature)
admin.site.register(TechnologyDevelopment)
admin.site.register(Vision)
