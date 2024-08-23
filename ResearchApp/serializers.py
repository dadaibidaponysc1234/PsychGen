from rest_framework import serializers
from .models import Study, Disorder, ResearchRegion, BiologicalModality, GeneticSourceMaterial, ArticleType,StudyDesign,AuthorRegion



class StudySerializer(serializers.ModelSerializer):
    disorder = serializers.SlugRelatedField(slug_field='disorder_name', queryset=Disorder.objects.all(), many=True, required=False)
    research_regions = serializers.SlugRelatedField(slug_field='name', queryset=ResearchRegion.objects.all(), many=True, required=False)
    author_regions = serializers.SlugRelatedField(slug_field='name', queryset=AuthorRegion.objects.all(), many=True, required=False)

    study_designs = serializers.SlugRelatedField(slug_field='design_name', queryset=StudyDesign.objects.all(), required=False)

    biological_modalities = serializers.SlugRelatedField(slug_field='modality_name', queryset=BiologicalModality.objects.all(), many=True, required=False)
    genetic_source_materials = serializers.SlugRelatedField(slug_field='material_type', queryset=GeneticSourceMaterial.objects.all(), many=True, required=False)
    article_type = serializers.SlugRelatedField(slug_field='article_name', queryset=ArticleType.objects.all(), many=True, required=False)
    

    class Meta:
        model = Study
        fields = '__all__'

    def create(self, validated_data):
        # Pop Many-to-Many fields
        disorder = validated_data.pop('disorder', [])
        research_regions = validated_data.pop('research_regions', [])
        author_regions = validated_data.pop('author_regions', [])
        # study_designs = validated_data.pop('study_designs', [])
        biological_modalities = validated_data.pop('biological_modalities', [])
        genetic_source_materials = validated_data.pop('genetic_source_materials', [])
        article_type = validated_data.pop('article_type', [])
        

        # Create the study
        study = Study.objects.create(**validated_data)

        # Add Many-to-Many fields
        study.disorder.set(disorder)
        study.research_regions.set(research_regions)
        study.author_regions.set(author_regions)
        # study.study_designs.set(study_designs)
        study.biological_modalities.set(biological_modalities)
        study.genetic_source_materials.set(genetic_source_materials)
        study.article_type.set(article_type)

        return study
