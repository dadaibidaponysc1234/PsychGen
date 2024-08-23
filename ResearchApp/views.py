from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import csv
from io import StringIO
from django.core.exceptions import ValidationError
from .models import Study, Disorder, ResearchRegion, BiologicalModality, GeneticSourceMaterial,  ArticleType,StudyDesign,AuthorRegion
from .serializers import StudySerializer
import re
from django.http import HttpResponse



class UploadCSVView(APIView):
    def post(self, request, format=None):
        file = request.FILES.get('file')
        if not file:
            return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)
        
        data = file.read().decode('ISO-8859-1')
        csv_data = csv.DictReader(StringIO(data))

        errors = []

        for row_number, row in enumerate(csv_data, start=1):
            if not any(row.values()):
                continue

            # Example: Processing the impact factor
            impact_factor = row.get('Impact Factor', '').replace(',', '.').strip()
            try:
                impact_factor = float(impact_factor) if impact_factor else None
            except ValueError:
                impact_factor = None

            study_designs = row.get('Study Design','').upper().strip()
            StudyDesign_name, _ = StudyDesign.objects.get_or_create(design_name=study_designs)
            

            # Prepare the data dictionary for the Study model
            row_data = {
                'title': row.get('Title', '').strip(),
                'year': row.get('Year', None),
                'journal_name': row.get('Journal Name', '').strip(),
                'impact_factor': impact_factor,
                'pmid': row.get('PMID', '').strip(),
                'funding_source': row.get('Funding Source', '').strip(),
                'lead_author': row.get('Lead Author', '').strip(),
                'phenotype': row.get('Phenotype', '').strip(),
                'diagnostic_criteria_used': row.get('Diagnostic Criteria Used', '').strip(),
                'sample_size': row.get('Sample Size', '').strip(),
                'age_range': row.get('Age Range', '').strip(),
                'mean_age': row.get('Mean Age', '').strip(),
                'male_female_split': row.get('Male/Female Split', '').strip(),
                'biological_risk_factor_studied': row.get('Biological Risk Factor Studied', '').strip(),
                'biological_rationale_provided': row.get('Biological Rationale Provided', '').strip(),
                'status_of_corresponding_gene': row.get('Status of Corresponding Gene', '').strip(),
                'technology_platform': row.get('Technology Platform', '').strip(),
                'evaluation_method': row.get('Evaluation Method', '').strip(),
                'statistical_model': row.get('Statistical Model', '').strip(),
                'criteria_for_significance': row.get('Criteria for Significance', '').strip(),
                'validation_performed': row.get('Validation Performed', '').strip(),
                'findings_conclusions': row.get('Findings/Conclusions', '').strip(),
                'generalisability_of_conclusion': row.get('Generalisability of Conclusion', '').strip(),
                'adequate_statistical_powered': row.get('Adequate Statistical Powered', '').strip(),
                # 'evaluation_method_statistical_model': row.get('Evaluation method/statistical model', '').strip(),
                'comment': row.get('Remark Comment', '').strip(),
                # 'study_designs': row.get('Study design','').strip()
                'study_designs': StudyDesign_name
            }

            # Handle Remark data
            should_exclude = row.get('Should Exclude?')

            if should_exclude and should_exclude.strip().lower()== 'yes':
                row_data['should_exclude'] = True
            else:
                row_data['should_exclude'] = False

            many_to_many_fields = {
                'disorder': (Disorder, 'disorder_name', row.get('Disorder', '').split(',')),
                'research_regions': (ResearchRegion, 'name', row.get('Region Data', '').split(',')),
                'author_regions': (AuthorRegion, 'name', row.get('Region Authors', '').split(',')),
                'biological_modalities': (BiologicalModality, 'modality_name', row.get('Biological Modality', '').split(',')),
                'genetic_source_materials': (GeneticSourceMaterial, 'material_type', row.get('Genetic Source Material', '').split(',')),
                'article_type': (ArticleType, 'article_name', row.get('Article Type', '').split(',')),
            }

            for key, (model, field, value_list) in many_to_many_fields.items():
                instances = []
                for item in value_list:
                    item = item.strip().upper()
                    if item:  # Only process non-empty strings
                        instance, _ = model.objects.get_or_create(**{field: item})
                        instances.append(instance)
                row_data[key] = instances

            serializer = StudySerializer(data=row_data)
            if serializer.is_valid():
                study = serializer.save()
            else:
                errors.append({'row': row_number, 'error': serializer.errors})

        if errors:
            return Response({"errors": errors}, status=status.HTTP_400_BAD_REQUEST)
        return Response({"message": "CSV file processed successfully."}, status=status.HTTP_201_CREATED)



def download_csv_example(request):
    # Create the HttpResponse object with the appropriate CSV header.
    response = HttpResponse(
        content_type='text/csv',
        headers={'Content-Disposition': 'attachment; filename="example_study_format.csv"'},
    )

    writer = csv.writer(response)
    
     # Write the CSV headers according to the model fields
    writer.writerow([
        'Title', 'Year', 'Journal Name', 'Impact Factor', 'PMID', 'Article Type', 'Funding Source', 
        'Lead Author', 'Research Regions', 'Author Regions', 'Disorder', 'Phenotype', 
        'Diagnostic Criteria Used', 'Study Design', 'Sample Size', 'Age Range', 
        'Mean Age', 'Male/Female Split', 'Biological Modalities', 'Biological Risk Factor Studied', 
        'Biological Rationale Provided', 'Status of Corresponding Gene', 'Technology Platform', 
        'Genetic Source Materials', 'Evaluation Method', 'Statistical Model', 'Criteria for Significance', 
        'Validation Performed', 'Findings/Conclusions', 'Generalisability of Conclusion', 
        'Adequate Statistical Powered', 'Remark Comment', 'Should Exclude?'
    ])

    # Write a few example rows
    writer.writerow([
        'Example Study Title', '2024', 'Journal of Genetics', '5.8', '12345678', 'Original Research', 'NIH', 
        'John doe', 'North America, South Africa', 'South Africa, United States of America & Canada', 'Schizophrenia, PTSD, Depression', 
        'DSM-5','Edinburgh Postnatal Depression Scale (EPDS) and the Beck Depression Inventory-II (BDI-II)', 'Case-Control', '100', '18-65', '35', '50/50', 'Genomics', 
        'Gene XYZ', 'Prenatal depression has been associated with adverse birth and child development outcomes. Epigenetics has been hypothesized to play a role in this association.', 'Active', 'PCR', 'Blood', 'DNA methylation measurement at individual CpG sites and differentially methylated regions', 
        'Regression', 'p < 0.05', 'Yes', 'The study found a significant association between Gene XYZ and Schizophrenia.', 
        'High', 'Yes', 'Some remarks here', 'No'
    ])


    return response
