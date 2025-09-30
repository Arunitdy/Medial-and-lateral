import pandas as pd
import os

# File name (in same folder as script)
file_name = "VBM data.xlsx"

# Load Excel file
df = pd.read_excel(file_name)

# Extract all column titles
columns = df.columns.tolist()

# Prepare README content
readme_content = "Column Titles in VBM data:\n\n"
for col in columns:
    readme_content += f"- {col}\n"

# Save README.txt in the same folder
output_file = os.path.join(os.path.dirname(__file__), "README.txt")
with open(output_file, "w", encoding="utf-8") as f:
    f.write(readme_content)

print("README.txt created with column titles.")
