import pandas as pd
import requests
import io

headers = {"User-Agent": "Mozilla/5.0"}
data_url = "https://epoch.ai/data/all_ai_models.csv"

response = requests.get(data_url, headers=headers)
models_df = pd.read_csv(io.StringIO(response.text))

with open("data_inspection.txt", "w") as f:
    f.write("=== HEAD ===\n")
    f.write(models_df.head().to_string())
    f.write("\n\n=== COLUMNS ===\n")
    f.write("\n".join(models_df.columns.tolist()))

print("Saved to data_inspection.txt")