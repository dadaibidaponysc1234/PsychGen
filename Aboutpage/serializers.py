from rest_framework import serializers
from .models import AboutPage, Mission, Objective, KeyFeature, TechnologyDevelopment, Vision

class MissionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Mission
        fields = '__all__'

class ObjectiveSerializer(serializers.ModelSerializer):
    class Meta:
        model = Objective
        fields = '__all__'

class KeyFeatureSerializer(serializers.ModelSerializer):
    class Meta:
        model = KeyFeature
        fields = '__all__'

class TechnologyDevelopmentSerializer(serializers.ModelSerializer):
    class Meta:
        model = TechnologyDevelopment
        fields = '__all__'

class VisionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Vision
        fields = '__all__'

class AboutPageSerializer(serializers.ModelSerializer):
    mission = MissionSerializer(many=True, read_only=True)
    objectives = ObjectiveSerializer(many=True, read_only=True)
    key_features = KeyFeatureSerializer(many=True, read_only=True)
    technology = TechnologyDevelopmentSerializer(many=True, read_only=True)
    vision = VisionSerializer(many=True, read_only=True)

    class Meta:
        model = AboutPage
        fields = [
            'id', 'title', 'introduction', 
            'mission', 'objectives', 'key_features', 
            'technology', 'vision', 'last_updated'
        ]
