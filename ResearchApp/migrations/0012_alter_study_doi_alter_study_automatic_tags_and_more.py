# Generated by Django 5.1 on 2024-10-01 19:34

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ResearchApp', '0011_alter_study_automatic_tags_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='study',
            name='DOI',
            field=models.CharField(blank=True, default='', max_length=100),
        ),
        migrations.AlterField(
            model_name='study',
            name='automatic_tags',
            field=models.CharField(blank=True, max_length=1200, null=True),
        ),
        migrations.AlterField(
            model_name='study',
            name='keyword',
            field=models.CharField(blank=True, max_length=1000, null=True),
        ),
        migrations.AlterField(
            model_name='study',
            name='statistical_model',
            field=models.CharField(blank=True, max_length=1000, null=True),
        ),
        migrations.AlterField(
            model_name='study',
            name='technology_platform',
            field=models.CharField(blank=True, max_length=1000, null=True),
        ),
        migrations.AlterField(
            model_name='study',
            name='validation_performed',
            field=models.CharField(blank=True, max_length=1000, null=True),
        ),
    ]