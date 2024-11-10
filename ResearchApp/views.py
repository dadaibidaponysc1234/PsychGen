# from django.contrib.auth.models import User
# from rest_framework.authtoken.models import Token
# from rest_framework.authtoken.views import ObtainAuthToken

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import csv
from io import StringIO
from django.core.exceptions import ValidationError
from django.db import models
from .models import Study, Disorder, BiologicalModality, GeneticSourceMaterial, ArticleType, StudyDesign, Country,Visitor
from .serializers import StudySerializer,VisitorCountSerializer
from django.http import HttpResponse
import json
import logging
from django.db.models import Count
from datetime import timedelta
from django.utils import timezone
from rest_framework.permissions import IsAuthenticated

from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import permission_classes

class LoginAPIView(APIView):
    def post(self, request):
        username = request.data.get('username')
        password = request.data.get('password')
        
        # Authenticate the user
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)  # Log the user in
            return Response({"message": "Login successful","username":user.username}, status=status.HTTP_200_OK)
        else:
            return Response({"error": "Invalid credentials"}, status=status.HTTP_401_UNAUTHORIZED)


class LogoutAPIView(APIView):
    permission_classes = [IsAuthenticated]  # Only authenticated users can log out

    def post(self, request):
        logout(request)  # Log the user out
        return Response({"message": "Logout successful"}, status=status.HTTP_200_OK)



logger = logging.getLogger(__name__)

class UploadCSVView(APIView):
    permission_classes = [IsAuthenticated]  # Restrict access to authenticated users only

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

            # Processing the impact factor
            impact_factor = row.get('Impact Factor', '').replace(',', '.').strip()
            try:
                impact_factor = float(impact_factor) if impact_factor else None
            except ValueError:
                impact_factor = None

            study_designs = row.get('Study Design', '').upper().strip()
            StudyDesign_name, _ = StudyDesign.objects.get_or_create(design_name=study_designs)

            # Prepare the data dictionary for the Study model
            row_data = {
                'pmid': row.get('PMID', '').strip(),
                'title': row.get('Title', '').strip(),
                'abstract': row.get('Abstract', '').strip(),
                'year': row.get('Year', None),
                'DOI': row.get('DOI', '').strip(),
                'journal_name': row.get('Journal Name', '').strip(),
                'impact_factor': impact_factor,
                'funding_source': row.get('Funding Source', '').strip(),
                'lead_author': row.get('Lead Author', '').strip(),
                'phenotype': row.get('Phenotype', '').strip(),
                'diagnostic_criteria_used': row.get('Diagnostic Criteria Used', '').strip(),
                'study_designs': StudyDesign_name,
                'sample_size': row.get('Sample Size', '').strip(),
                'age_range': row.get('Age Range', '').strip(),
                'mean_age': row.get('Mean Age', '').strip(),
                'male_female_split': row.get('Male/Female Split', '').strip(),
                'citation': row.get('Citation', '').strip(),
                'keyword': row.get('Keywords', '').strip(),
                'date': row.get('Date', '').strip(),
                'pages': row.get('Pages', '').strip(),
                'issue': row.get('Issue', None),
                'volume': row.get('Volume', None),
                'automatic_tags': row.get('Automatic Tags', '').strip(),
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
                'comment': row.get('Remark Comment', '').strip(),
            }

            # Handle authors/affiliations JSON field
            authors_affiliations = row.get('Authors/Affiliations', '').strip()
            try:
                authors_affiliations = json.loads(authors_affiliations) if authors_affiliations else None
            except json.JSONDecodeError:
                authors_affiliations = None
                errors.append({'row': row_number, 'error': 'Invalid JSON format in authors_affiliations field'})
            row_data['authors_affiliations'] = authors_affiliations

            # Handle exclusion based on 'Should Exclude?' column
            should_exclude = row.get('Should Exclude?')
            row_data['should_exclude'] = True if should_exclude and should_exclude.strip().lower() == 'yes' else False

            # Handling many-to-many relationships
            many_to_many_fields = {
                'countries': (Country, 'name', row.get('Countries', '').split(',')),
                'article_type': (ArticleType, 'article_name', row.get('Article Type', '').split(',')),
                'disorder': (Disorder, 'disorder_name', row.get('Disorder', '').split(',')),
                'biological_modalities': (BiologicalModality, 'modality_name', row.get('Biological Modality', '').split(',')),
                'genetic_source_materials': (GeneticSourceMaterial, 'material_type', row.get('Genetic Source Materials', '').split(','))
            }
            
            for key, (model, field, value_list) in many_to_many_fields.items():
                instances = []
                # field_data = []
                for item in value_list:
                    item = item.strip().upper()
                    if item:
                        instance, _ = model.objects.get_or_create(**{field: item})
                        instances.append(instance)
                        # field_data.append({"id": instance.id, field: getattr(instance, field)})
                        # field_data.append({field: getattr(instances, field)})
                row_data[key] = instances

            # Save the data using the serializer
            serializer = StudySerializer(data=row_data)
            if serializer.is_valid():
                try:
                    serializer.save()
                except Exception as e:
                    logger.error(f"Error saving row {row_number}: {str(e)}")
                    errors.append({'row': row_number, 'error': str(e)})
            else:
                logger.error(f"Validation error at row {row_number}: {serializer.errors}")
                errors.append({'row': row_number, 'error': serializer.errors})

        # Return success or errors
        if errors:
            return Response({"errors": errors}, status=status.HTTP_400_BAD_REQUEST)
        return Response({"message": "CSV file processed successfully."}, status=status.HTTP_201_CREATED)


class DownloadCSVExampleView(APIView):
    permission_classes = [IsAuthenticated]  # Restrict access to authenticated users only
    def get(self, request, format=None):
        # Define the response and CSV writer
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="study_template.csv"'

        writer = csv.writer(response)

        # Write the CSV header
        writer.writerow([
            'PMID', 'Title', 'Abstract', 'Year', 'DOI', 'Journal Name', 'Countries', 'Impact Factor',
            'Article Type', 'Funding Source', 'Lead Author', 'Disorder', 'Phenotype', 
            'Diagnostic Criteria Used', 'Study Design', 'Sample Size', 'Age Range', 'Mean Age', 
            'Male/Female Split', 'Biological Modality','Citation','Keywords', 'Date', 'Pages', 'Issue', 'Volume','Automatic Tags', 
            'Authors/Affiliations', 'Biological Risk Factor Studied','Biological Rationale Provided',
            'Status of Corresponding Gene', 'Technology Platform', 
            'Genetic Source Materials', 'Evaluation Method', 'Statistical Model', 
            'Criteria for Significance', 'Validation Performed', 'Findings/Conclusions', 
            'Generalisability of Conclusion', 'Adequate Statistical Powered','Comment', 
            'Should Exclude?'
        ])

        # Write a sample row
        writer.writerow([
            '123456', 'Example Study Title', 'Study abstract', '2024', '10.1234/doi-example', 
            'Journal of Genetics', 'USA, Canada', '5.8', 'Original Research', 'NIH', 'John Doe', 
            'Schizophrenia', 'DSM-5','Pre-diagnosed', 'Case-Control', '100', '18-65', '35', '50/50', 'Genomics', 
            '0', 'Social anxiety disorder; Genetics; Serotonin; Dopamine; Temperament.', 
            '2024-01-01', '123-135', '2', '12', 'Automatic tag', '[{"author": "John Doe", "affiliation": "University X"}]',
            'Gene XYZ', 'Epigenetics hypothesis', 'Active', 'PCR', 'Blood', 
            'DNA methylation', 'Regression', 'p < 0.05', 'Internal Validation', 
            'No significant difference', 'Limited due to small sample size', 'low power', 
            'Some remarks', 'No'
        ])

        return response


class VisitorCountAPIView(APIView):
    # permission_classes = [IsAuthenticated]  # Restrict access to authenticated users only

    def get(self, request):
        # Count unique visitors by IP address
        unique_visitors = Visitor.objects.values('ip_address').distinct().count()
        
        # Count total visits
        total_visits = Visitor.objects.count()

        # Get today's date and calculate the date 7 days ago
        today = timezone.now().date()
        last_7_days = today - timedelta(days=6)


        # Query for visit counts grouped by day for the last 7 days
        daily_visits = (
            Visitor.objects
            .filter(visit_date__date__range=[last_7_days, today])
            .annotate(day=models.functions.TruncDate('visit_date'))
            .values('day')
            .annotate(visit_count=Count('id'))
            .order_by('day')
        )

        # Prepare daily visits data as a list of dictionaries with date and visit count
        daily_visits_data = [
            {"date": day_data['day'], "visit_count": day_data['visit_count']}
            for day_data in daily_visits
        ]

        # Prepare data for the response
        data = {
            "unique_visitors": unique_visitors,
            "total_visits": total_visits,
            "daily_visits": daily_visits_data
        }

        serializer = VisitorCountSerializer(data)
        return Response(serializer.data, status=status.HTTP_200_OK)
