# utils/mapping.py
STANDARD_HEADERS = ["SNP", "A1", "A2", "P", "BETA"]

REQUIRED_COLUMNS = ["SNP", "A1", "A2", "BETA", "P"]

COLUMN_ALIASES = {
    "SNP": ["SNP", "rsID", "ID", "RS", "SNP_ID", "VARIANT_ID", "MARKERNAME"],
    "A1": ["A1", "ALLELE1", "EFFECT_ALLELE", "ALT", "ALTERNATE_ALLELE"],
    "A2": ["A2", "ALLELE2", "REFERENCE_ALLELE", "REF", "ALLELE0", "NONEFFECT_ALLELE"],
    "BETA": ["BETA", "B", "EFFECT", "LOG_ODDS", "EFFECT_SIZE"],
    "P": ["P", "PVAL", "P_VALUE", "PVALUE", "P_DGC", "P_WALD"]
}

def auto_map_columns(file_headers):
    mapping = {}
    used = set()
    for std_col, aliases in COLUMN_ALIASES.items():
        for col in file_headers:
            if col.upper() in [a.upper() for a in aliases] and std_col not in used:
                mapping[col] = std_col
                used.add(std_col)
                break
    return mapping


