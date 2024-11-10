from rest_framework import serializers
from .models import Study, Disorder, BiologicalModality, GeneticSourceMaterial, ArticleType, StudyDesign, Country


class CountrySerializer(serializers.ModelSerializer):
    class Meta:
        model = Country
        fields = ['id', 'name']

class StudySerializer(serializers.ModelSerializer):
    disorder = serializers.SlugRelatedField(slug_field='disorder_name', queryset=Disorder.objects.all(), many=True, required=False)
    countries = serializers.SlugRelatedField(slug_field='name', queryset=Country.objects.all(), many=True, required=False)
    study_designs = serializers.SlugRelatedField(slug_field='design_name', queryset=StudyDesign.objects.all(), required=False)
    biological_modalities = serializers.SlugRelatedField(slug_field='modality_name', queryset=BiologicalModality.objects.all(), many=True, required=False)
    genetic_source_materials = serializers.SlugRelatedField(slug_field='material_type', queryset=GeneticSourceMaterial.objects.all(), many=True, required=False)
    article_type = serializers.SlugRelatedField(slug_field='article_name', queryset=ArticleType.objects.all(), many=True, required=False)
    # countries = CountrySerializer(many=True, required=False)

    # country_ids = serializers.PrimaryKeyRelatedField(
    #     queryset=Country.objects.all(), source='countries', many=True, write_only=True
    # )

    class Meta:
        model = Study
        fields = '__all__'

    def create(self, validated_data):
        # Pop Many-to-Many fields
        disorder = validated_data.pop('disorder', [])
        countries = validated_data.pop('countries', [])
        study_designs = validated_data.pop('study_designs', None)
        biological_modalities = validated_data.pop('biological_modalities', [])
        genetic_source_materials = validated_data.pop('genetic_source_materials', [])
        article_type = validated_data.pop('article_type', [])

        # Create the study
        study = Study.objects.create(**validated_data)

        # Add Many-to-Many fields
        study.countries.set(countries)
        study.disorder.set(disorder)
        # study.author_regions.set(author_regions)
        if study_designs:
            study.study_designs = study_designs
        study.biological_modalities.set(biological_modalities)
        study.genetic_source_materials.set(genetic_source_materials)
        study.article_type.set(article_type)

        # for country in countries_data:
        #     country_obj, _ = Country.objects.get_or_create(**country)
        #     study.countries.add(country_obj)

        study.save()  # Save changes after setting many-to-many fields

        return study

class DailyVisitSerializer(serializers.Serializer):
    date = serializers.DateField()
    visit_count = serializers.IntegerField()

class VisitorCountSerializer(serializers.Serializer):
    unique_visitors = serializers.IntegerField()
    total_visits = serializers.IntegerField()
    daily_visits = DailyVisitSerializer(many=True)
