from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework import status
from .utils.preprocessing import extract_zip_to_temp, categorize_files
from rest_framework.response import Response
from .utils.mapping import auto_map_columns
import pandas as pd
import os
import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .utils.mapping import auto_map_columns
import os
import json
import zipfile
import tempfile
from django.http import FileResponse, Http404



class UploadZipView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        zip_file = request.FILES.get("zip_file")
        tool = request.data.get("tool")

        if not zip_file or not zip_file.name.endswith(".zip"):
            return Response({"error": "Please upload a valid .zip file"}, status=400)

        if tool not in ["prscsx", "prsice", "sdprx"]:
            return Response({"error": "Invalid or missing tool type."}, status=400)

        try:
            # ✅ Pass tool_name to the function
            temp_dir = extract_zip_to_temp(zip_file, tool)
            file_structure = categorize_files(temp_dir)

            request.session["upload_dir"] = temp_dir
            request.session["tool"] = tool

            return Response({
                "message": f"✅ Zip extracted successfully for tool: {tool}",
                "tool": tool,
                "file_summary": file_structure,
                "upload_dir": temp_dir  # <-- add this
})


        except Exception as e:
            return Response({"error": str(e)}, status=500)


class AutoMapColumnsView(APIView):
    """
    Maps columns to standard fields and returns:
    - JSON for frontend preview (all mapped data)
    - Column mapping info (standard_name → original_name)
    - TXT file for download (transposed format)
    - Cleaned .txt file per dataset
    """

    def get(self, request):
        upload_dir = request.GET.get("upload_dir")
        if not upload_dir or not os.path.exists(upload_dir):
            return Response({"error": "Missing or invalid upload_dir"}, status=400)

        sumstats_dir = os.path.join(upload_dir, "sumstats")
        if not os.path.exists(sumstats_dir):
            return Response({"error": "Sumstats folder not found."}, status=404)

        output_path = os.path.join(upload_dir, "cleaned_mapped.txt")
        cleaned_dir = os.path.join(upload_dir, "cleaned_files")
        os.makedirs(cleaned_dir, exist_ok=True)

        json_output = []

        with open(output_path, "w") as fout:
            for fname in os.listdir(sumstats_dir):
                if fname.endswith((".txt", ".csv", ".tsv", ".gz")):
                    try:
                        fpath = os.path.join(sumstats_dir, fname)
                        if fname.endswith(".gz"):
                            df = pd.read_csv(fpath, sep=r"\s+", compression="gzip")
                        else:
                            df = pd.read_csv(fpath, sep=None, engine="python")

                        # Auto-map columns
                        mapping = auto_map_columns(df.columns)
                        mapped = {
                            col: mapping[col]
                            for col in df.columns
                            if col in mapping and mapping[col] != "Unmapped"
                        }

                        if not mapped:
                            continue

                        # Reverse mapping (standardized → original)
                        reverse_map = {std: orig for orig, std in mapped.items()}

                        # Write transposed preview output
                        fout.write(f"# File: {fname}\n")
                        for std_col, orig_col in reverse_map.items():
                            fout.write(f"{std_col}:\t{','.join(map(str, df[orig_col]))}\n")
                        fout.write("\n")

                        # Write full cleaned file (normal tabular format)
                        cleaned_table_path = os.path.join(cleaned_dir, f"{fname.replace('.txt', '')}_mapped.txt")
                        df_cleaned = pd.DataFrame({
                            std_col: df[orig_col] for std_col, orig_col in reverse_map.items()
                        })
                        df_cleaned.to_csv(cleaned_table_path, sep="\t", index=False)

                        # Append to response
                        json_output.append({
                            "file": fname,
                            "mapped_data": {
                                std_col: df[orig_col].tolist()
                                for std_col, orig_col in reverse_map.items()
                            },
                            "column_mapping": reverse_map
                        })

                    except Exception as e:
                        json_output.append({
                            "file": fname,
                            "error": str(e)
                        })

        return Response({
            "results": json_output,
            "mapped_file": f"/api/download/mapped-columns/?upload_dir={upload_dir}"
        })

# views.py
class SaveMappingView(APIView):
    def post(self, request):
        upload_dir = request.data.get("upload_dir")
        mappings = request.data.get("mappings")  # Expect: { "filename.txt": { "ORIG": "TARGET", ... } }

        if not upload_dir or not mappings:
            return Response({"error": "Missing upload_dir or mappings."}, status=400)

        mapping_path = os.path.join(upload_dir, "final_column_mappings.json")
        try:
            with open(mapping_path, "w") as f:
                json.dump(mappings, f, indent=2)

            return Response({"message": "✅ Mappings saved successfully."})
        except Exception as e:
            return Response({"error": str(e)}, status=500)

# views.py (example)
class ApplySavedMappingView(APIView):
    def post(self, request):
        upload_dir = request.data.get("upload_dir")

        if not upload_dir:
            return Response({"error": "Missing upload_dir"}, status=400)

        mapping_file = os.path.join(upload_dir, "final_column_mappings.json")
        if not os.path.exists(mapping_file):
            return Response({"error": "Saved mapping not found."}, status=404)

        try:
            with open(mapping_file, "r") as f:
                mappings = json.load(f)

            sumstats_dir = os.path.join(upload_dir, "sumstats")
            output_dir = os.path.join(upload_dir, "cleaned_files")
            os.makedirs(output_dir, exist_ok=True)

            generated_files = []

            for fname, colmap in mappings.items():
                path = os.path.join(sumstats_dir, fname)
                if not os.path.exists(path):
                    continue

                if fname.endswith(".gz"):
                    df = pd.read_csv(path, sep=r"\s+", compression="gzip")
                else:
                    df = pd.read_csv(path, sep=None, engine="python")

                final_cols = {
                    std: orig for orig, std in colmap.items()
                    if std != "Unmapped" and orig in df.columns
                }

                if not final_cols:
                    continue

                cleaned_df = df[list(final_cols.values())].rename(columns={v: k for k, v in final_cols.items()})
                cleaned_filename = f"{os.path.splitext(fname)[0]}_cleaned.txt"
                cleaned_path = os.path.join(output_dir, cleaned_filename)
                cleaned_df.to_csv(cleaned_path, sep="\t", index=False)

                generated_files.append({
                    "file": cleaned_filename,
                    "columns": list(cleaned_df.columns),
                    "row_count": len(cleaned_df),
                    "data": cleaned_df.to_dict(orient="records")  # 🚀 full content here
                })

            return Response({
                "message": "✅ Structured cleaned files saved.",
                "files": generated_files
            })

        except Exception as e:
            return Response({"error": str(e)}, status=500)


class DownloadMappedCleanedFileView(APIView):
    def get(self, request):
        upload_dir = request.GET.get("upload_dir")
        filename = request.GET.get("filename")

        if not upload_dir or not filename:
            return Response({"error": "upload_dir and filename required"}, status=400)

        filepath = os.path.join(upload_dir, "cleaned_files", filename)
        if not os.path.exists(filepath):
            return Response({"error": "File not found"}, status=404)

        return FileResponse(open(filepath, "rb"), filename=filename)


class DownloadMappedColumnsView(APIView):
    def get(self, request):
        upload_dir = request.GET.get("upload_dir")
        if not upload_dir:
            raise Http404("Missing upload_dir")

        file_path = os.path.join(upload_dir, "cleaned_mapped.txt")
        if not os.path.exists(file_path):
            raise Http404("Mapped columns file not found.")

        return FileResponse(open(file_path, "rb"), as_attachment=True, filename="cleaned_mapped.txt")
