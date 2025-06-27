import tempfile, zipfile, os

def extract_zip_to_temp(zip_file, tool_name):
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    # Check if the first extracted folder is the root (like "h3gwas_data")
    entries = os.listdir(temp_dir)
    if len(entries) == 1:
        root_path = os.path.join(temp_dir, entries[0])
        if os.path.isdir(root_path):
            return root_path  # return actual data folder (e.g., /tmp/xxxx/h3gwas_data)

    return temp_dir


def categorize_files(folder_path):
    return {
        "geno": os.listdir(os.path.join(folder_path, "geno")) if os.path.exists(os.path.join(folder_path, "geno")) else [],
        "pheno": os.listdir(os.path.join(folder_path, "pheno")) if os.path.exists(os.path.join(folder_path, "pheno")) else [],
        "sumstats": os.listdir(os.path.join(folder_path, "sumstats")) if os.path.exists(os.path.join(folder_path, "sumstats")) else [],
    }
