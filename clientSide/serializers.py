from rest_framework import serializers
from ResearchApp.models import Study, Disorder, BiologicalModality, GeneticSourceMaterial, ArticleType, StudyDesign, Country
from rest_framework import serializers
# from your_app.models import (
#     Study, Disorder, ResearchRegion, StudyDesign, BiologicalModality, 
#     GeneticSourceMaterial, ArticleType, Author, Remark
# )

class DisorderSerializer(serializers.ModelSerializer):
    class Meta:
        model = Disorder
        fields = '__all__'

class CountrySerializer(serializers.ModelSerializer):
    class Meta:
        model = Country
        fields = ['id', 'name']

# class StudyDesignSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = StudyDesign
#         fields = '__all__'

class BiologicalModalitySerializer(serializers.ModelSerializer):
    class Meta:
        model = BiologicalModality
        fields = '__all__'

class GeneticSourceMaterialSerializer(serializers.ModelSerializer):
    class Meta:
        model = GeneticSourceMaterial
        fields = '__all__'

class ArticleTypeSerializer(serializers.ModelSerializer):
    class Meta:
        model = ArticleType
        fields = '__all__'

class StudySerializer(serializers.ModelSerializer):
    disorder = DisorderSerializer(many=True, read_only=True)
    Country = CountrySerializer(many=True, read_only=True)
    biological_modalities = BiologicalModalitySerializer(many=True, read_only=True)
    genetic_source_materials = GeneticSourceMaterialSerializer(many=True, read_only=True)
    article_type = ArticleTypeSerializer(many=True, read_only=True)

    class Meta:
        model = Study
        fields = '__all__'


class DisorderStudyCountSerializer(serializers.Serializer):
    disorder__disorder_name = serializers.CharField()  # Use 'disorder__disorder_name'
    study_count = serializers.IntegerField()


class ResearchRegionStudyCountSerializer(serializers.Serializer):
    research_regions__name = serializers.CharField()
    study_count = serializers.IntegerField()

class BiologicalModalityStudyCountSerializer(serializers.Serializer):
    biological_modalities__modality_name = serializers.CharField()
    study_count = serializers.IntegerField()

class GeneticSourceMaterialStudyCountSerializer(serializers.Serializer):
    genetic_source_materials__material_type = serializers.CharField()
    study_count = serializers.IntegerField()

class YearlyStudyCountSerializer(serializers.Serializer):
    year = serializers.IntegerField()  # Assuming the year is stored as an integer
    study_count = serializers.IntegerField()    