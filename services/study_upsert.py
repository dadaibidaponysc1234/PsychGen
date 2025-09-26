# # services/study_upsert.py
# from typing import Any, Dict, List, Tuple
# from django.db import transaction
# from ResearchApp.models import (
#     Study, StudyDocument, StudyDesign, Country, ArticleType, Disorder,
#     BiologicalModality, GeneticSourceMaterial
# )

# def _goc_list(model, field_name: str, items: List[str]):
#     out = []
#     for raw in items or []:
#         val = (raw or "").strip()
#         if not val:
#             continue
#         obj, _ = model.objects.get_or_create(**{field_name: val})
#         out.append(obj)
#     return out

# @transaction.atomic
# def upsert_study_with_metadata(meta: Dict[str, Any]) -> Tuple[Study, bool]:
#     sd = None
#     sd_name = (meta.get("study_designs") or "").strip()
#     if sd_name:
#         sd, _ = StudyDesign.objects.get_or_create(design_name=sd_name)

#     title = (meta.get("title") or "").strip()
#     year = meta.get("year")
#     pmid = (meta.get("pmid") or "") or None

#     study, created = Study.objects.get_or_create(
#         title=title, year=year, pmid=pmid,
#         defaults={
#             "abstract": meta.get("abstract") or "",
#             "DOI": meta.get("DOI") or "",
#             "journal_name": meta.get("journal_name") or None,
#             "impact_factor": meta.get("impact_factor"),
#             "funding_source": meta.get("funding_source") or None,
#             "lead_author": meta.get("lead_author") or None,
#             "phenotype": meta.get("phenotype") or None,
#             "diagnostic_criteria_used": meta.get("diagnostic_criteria_used") or None,
#             "study_designs": sd,
#             "sample_size": meta.get("sample_size") or None,
#             "age_range": meta.get("age_range") or None,
#             "mean_age": meta.get("mean_age") or None,
#             "male_female_split": meta.get("male_female_split") or None,
#             "citation": meta.get("citation") or 0,
#             "keyword": meta.get("keyword") or None,
#             "date": meta.get("date") or None,
#             "pages": meta.get("pages") or None,
#             "issue": meta.get("issue") or None,
#             "volume": meta.get("volume") or None,
#             "automatic_tags": meta.get("automatic_tags") or None,
#             "authors_affiliations": meta.get("authors_affiliations") or None,
#             "biological_risk_factor_studied": meta.get("biological_risk_factor_studied") or None,
#             "biological_rationale_provided": meta.get("biological_rationale_provided") or None,
#             "status_of_corresponding_gene": meta.get("status_of_corresponding_gene") or None,
#             "technology_platform": meta.get("technology_platform") or None,
#             "evaluation_method": meta.get("evaluation_method") or None,
#             "statistical_model": meta.get("statistical_model") or None,
#             "criteria_for_significance": meta.get("criteria_for_significance") or None,
#             "validation_performed": meta.get("validation_performed") or None,
#             "findings_conclusions": meta.get("findings_conclusions") or None,
#             "generalisability_of_conclusion": meta.get("generalisability_of_conclusion") or None,
#             "adequate_statistical_powered": meta.get("adequate_statistical_powered") or None,
#             "comment": meta.get("comment") or None,
#             "should_exclude": bool(meta.get("should_exclude") or False),
#         }
#     )

#     if not created and sd and study.study_designs_id is None:
#         study.study_designs = sd
#         study.save(update_fields=["study_designs"])

#     study.countries.set(_goc_list(Country, "name", meta.get("countries")))
#     study.article_type.set(_goc_list(ArticleType, "article_name", meta.get("article_type")))
#     study.disorder.set(_goc_list(Disorder, "disorder_name", meta.get("disorder")))
#     study.biological_modalities.set(_goc_list(BiologicalModality, "modality_name", meta.get("biological_modalities")))
#     study.genetic_source_materials.set(_goc_list(GeneticSourceMaterial, "material_type", meta.get("genetic_source_materials")))

#     return study, created
