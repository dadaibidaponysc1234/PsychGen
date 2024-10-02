from django.db import models

class Disorder(models.Model):
    disorder_name = models.CharField(max_length=255)

    def __str__(self):
        return self.disorder_name

class ArticleType(models.Model):
    article_name = models.CharField(max_length=255)

    def __str__(self):
        return self.article_name

class Country(models.Model):  # Renamed to singular form
    name = models.CharField(max_length=255)

    def __str__(self):
        return self.name

class StudyDesign(models.Model):
    design_name = models.CharField(max_length=100)

    def __str__(self):
        return self.design_name

class BiologicalModality(models.Model):
    modality_name = models.CharField(max_length=255)

    def __str__(self):
        return self.modality_name

class GeneticSourceMaterial(models.Model):
    material_type = models.CharField(max_length=255)

    def __str__(self):
        return self.material_type

class Study(models.Model):
    pmid = models.CharField(max_length=50, null=True, blank=True)
    title = models.TextField(null=False, blank=False)
    abstract = models.TextField(null=False, blank=False)
    year = models.IntegerField(null=True, blank=True)
    DOI = models.CharField(max_length=100, blank=True, default='')  # Added unique constraint
    journal_name = models.CharField(max_length=255, null=True, blank=True)

    countries = models.ManyToManyField(Country, blank=True)

    impact_factor = models.FloatField(null=True, blank=True)
    article_type = models.ManyToManyField(ArticleType, blank=True)
    funding_source = models.CharField(max_length=1000, null=True, blank=True)
    lead_author = models.CharField(max_length=100, null=True, blank=True)

    disorder = models.ManyToManyField(Disorder, blank=True)
    
    phenotype = models.TextField(null=True, blank=True)
    diagnostic_criteria_used = models.TextField(null=True, blank=True)
    study_designs = models.ForeignKey(StudyDesign, on_delete=models.SET_NULL, null=True, blank=True)
    sample_size = models.CharField(max_length=255, null=True, blank=True)
    age_range = models.CharField(max_length=255, null=True, blank=True)
    mean_age = models.CharField(max_length=255, null=True, blank=True)
    male_female_split = models.CharField(max_length=400, null=True, blank=True)
    biological_modalities = models.ManyToManyField(BiologicalModality, blank=True)
    citation = models.IntegerField(default=0)
    keyword = models.CharField(max_length=1000, null=True, blank=True)
    date = models.CharField(max_length=100, null=True, blank=True)
    pages = models.CharField(max_length=100, null=True, blank=True)
    issue = models.CharField(max_length=10, null=True, blank=True)
    volume = models.CharField(max_length=10, null=True, blank=True)
    automatic_tags = models.CharField(max_length=1200, null=True, blank=True)
    authors_affiliations = models.JSONField()  # Store authors as JSON array
    biological_risk_factor_studied = models.TextField(null=True, blank=True)
    biological_rationale_provided = models.TextField(null=True, blank=True)
    status_of_corresponding_gene = models.TextField(null=True, blank=True)
    technology_platform = models.CharField(max_length=1000, null=True, blank=True)
    genetic_source_materials = models.ManyToManyField(GeneticSourceMaterial, blank=True)
    evaluation_method = models.CharField(max_length=1000, null=True, blank=True)
    statistical_model = models.CharField(max_length=1000, null=True, blank=True)
    criteria_for_significance = models.CharField(max_length=500, null=True, blank=True)
    validation_performed = models.CharField(max_length=1000, null=True, blank=True)
    findings_conclusions = models.TextField(null=True, blank=True)
    generalisability_of_conclusion = models.CharField(max_length=500, null=True, blank=True)
    adequate_statistical_powered = models.CharField(max_length=255, null=True, blank=True)
    comment = models.TextField(null=True, blank=True)
    should_exclude = models.BooleanField(default=False)

    class Meta:
        verbose_name = "Study"
        verbose_name_plural = "Studies"
        constraints = [
            models.UniqueConstraint(
                fields=['title', 'year', 'pmid'],
                name='unique_study'
            )
        ]

    def __str__(self):
        return self.title
